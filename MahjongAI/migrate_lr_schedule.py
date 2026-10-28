"""Switch a steps_cosine run (written before T-067) to the games_cosine_v1 LR schedule, in a NEW run dir (plan T-067).

Run from the repo root: python -m MahjongAI.migrate_lr_schedule <src_run_dir> <new_run_dir> [--lr-cycle-games N]

The source is only read: the source, the new run dir and its staging folder <new_run_dir>.migrating must not overlap
(canonical paths, symlinks resolved). The migration:
  1. creates <new_run_dir>.migrating holding an ownership record (lr_migration.json: this tool, the source, the new run
     dir); copies into it the source's metrics.jsonl, train.log (when it has one) and ONLY its latest checkpoint (older
     ones would stay steps_cosine): the snapshot;
  2. refuses unless that checkpoint is steps_cosine, it and the metrics header agree on every run-identity key
     (supervised_learning.config_mismatch) and the validation ids, and the run state has a games counter
     (supervised_learning.games_read_of_state); the new lr_schedule passes the corpus guard (games_cosine_schedule:
     the recorded LR_CYCLE_GAMES on its corpus only, else --lr-cycle-games N). A refusal removes the staging folder;
  3. completes the record with the provenance (the source checkpoint's path and sha256, the metrics snapshot's sha256)
     and the migration (the schedule, the history entry {"at_update", "games", "from", "to"}, the new configs), writes
     the checkpoint again with the new lr_schedule and that entry (atomic: *.tmp, then os.replace), then renames the
     folder to <new_run_dir>;
  4. writes the header the same way (the new config and the same history entry), atomic.
The trainer resumes only when the checkpoint and the header agree: a stop between 3 and 4 leaves "checkpoint new,
header old", which the trainer names as an interrupted migration. Running this again completes it from the record and
the snapshot, whatever the source did since (a newer checkpoint, appended rows); it refuses a new run dir whose record
names another source (or a source checkpoint changed since), and an existing staging folder whose ownership record is
not this migration's. A complete migration is left as it is; a staging folder of this migration is built again. An
unrelated mismatch still fails. The switch sets the LR from the steps_cosine value (0 at a finished run's end) to
lr_at(g, k) on purpose.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile

import torch

from MahjongAI import supervised_learning as SL

RECORD = "lr_migration.json"
TOOL = "MahjongAI.migrate_lr_schedule"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_atomic(state: dict, path: str):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def _write_record(record: dict, folder: str):
    path = os.path.join(folder, RECORD)
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(path + ".tmp", path)


def _read_record(folder: str):
    """The record in folder, or None when it has none (or one this tool did not write)."""
    try:
        with open(os.path.join(folder, RECORD)) as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) and record.get("tool") == TOOL else None


def _overlap(a: str, b: str) -> bool:
    return os.path.commonpath([a, b]) in (a, b)


class Migration:
    """The migration of one source run (its latest checkpoint) into new_run_dir: the path checks at construction,
    run() for the steps of the module docstring from whatever state an earlier stop left."""

    def __init__(self, src_run_dir: str, new_run_dir: str, lr_cycle_games: int = None):
        self.src, self.new = os.path.normpath(src_run_dir), os.path.normpath(new_run_dir)
        self.staging = SL.migration_staging(self.new)
        self.lr_cycle_games = lr_cycle_games
        self.src_real, self.new_real = os.path.realpath(self.src), os.path.realpath(self.new)
        for other in (self.new, self.staging):
            assert not _overlap(self.src_real, os.path.realpath(other)), (
                f"{self.src} and {other} overlap: the new run dir and its staging folder must be outside the source, "
                "and it outside them (the source is only read)")
        assert os.path.isdir(self.src), f"{self.src}: no run dir"
        self.owner = {"tool": TOOL, "src": self.src_real, "new": self.new_real}

    # ------------------------------------------------------------ step 2: the checks, on the snapshot
    def _record(self, checkpoint_path: str, metrics_path: str, src_checkpoint: str, step: int) -> dict:
        """The full record of a snapshot (its checkpoint and metrics.jsonl copies) of src_checkpoint, after the checks
        of step 2 (messages name the source files)."""
        src_metrics = os.path.join(self.src, "metrics.jsonl")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        header, _ = SL.Metrics(metrics_path).read()
        for where, config in ((src_checkpoint, checkpoint["config"]), (src_metrics, header["config"])):
            schedule = SL.with_legacy_run(config)["lr_schedule"]
            assert schedule == SL.LEGACY_LR_SCHEDULE, f"{where} is lr_schedule {schedule}, not steps_cosine"
        mismatch = SL.config_mismatch(checkpoint["config"], header["config"])
        assert not mismatch and checkpoint["val_ids"] == header["val_ids"], (
            f"{src_checkpoint} and {src_metrics} header disagree"
            + (f": {SL._mismatch_text(mismatch)}" if mismatch else ": validation games"))
        assert checkpoint["current_step"] == step, f"{src_checkpoint} holds step {checkpoint['current_step']}"
        games = SL.games_read_of_state(checkpoint)
        config = SL.with_legacy_run(checkpoint["config"])
        schedule = SL.games_cosine_schedule(config["train_files_sha256"], config["expected_skips"],
                                            self.lr_cycle_games)
        return {**self.owner, "src_checkpoint": os.path.realpath(src_checkpoint),
                "src_checkpoint_sha256": _sha256(checkpoint_path), "src_metrics_sha256": _sha256(metrics_path),
                "step": step, "schedule": schedule,
                "entry": {"at_update": step, "games": games, "from": SL.LEGACY_LR_SCHEDULE["name"],
                          "to": schedule["name"]},
                "checkpoint_config": {**config, "lr_schedule": schedule},
                "header_config": {**SL.with_legacy_run(header["config"]), "lr_schedule": schedule}}

    def _is_migrated_checkpoint(self, checkpoint: dict) -> bool:
        # config_mismatch: a trainer stop at this step re-saves it with its own total_steps
        r = self.record
        return (not SL.config_mismatch(checkpoint["config"], r["checkpoint_config"])
                and checkpoint.get("schedule_history") == [r["entry"]] and checkpoint["current_step"] == r["step"]
                and "scheduler_state_dict" not in checkpoint)

    def _is_migrated_header(self, header: dict) -> bool:
        return (not SL.config_mismatch(header["config"], self.record["header_config"])
                and header.get("schedule_history") == [self.record["entry"]])

    # ------------------------------------------------------------ steps 1, 3, 4
    def _migrate_checkpoint(self, path: str):
        """Step 3's rewrite: the new lr_schedule and the history entry; the LambdaLR state goes (lr_at needs none)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint.pop("scheduler_state_dict")
        checkpoint["config"] = self.record["checkpoint_config"]
        checkpoint["schedule_history"] = [self.record["entry"]]
        _save_atomic(checkpoint, path)

    def _migrate_header(self):
        """Step 4: the header with the new config and the history entry (Metrics._write: *.tmp, then os.replace)."""
        metrics = SL.Metrics(os.path.join(self.new, "metrics.jsonl"))
        header, rows = metrics.read()
        header["config"] = self.record["header_config"]
        header["schedule_history"] = [self.record["entry"]]
        metrics._write([header] + rows)

    def _make_staging(self):
        """Step 1's folder, created with its ownership record (in a temporary folder, then renamed): a staging folder
        without this tool's record is never one of its own. An existing one is removed only when its record is this
        migration's (same source, same new run dir)."""
        if os.path.lexists(self.staging):
            owner = None if os.path.islink(self.staging) else _read_record(self.staging)
            assert owner and all(owner.get(k) == v for k, v in self.owner.items()), (
                f"{self.staging} exists and is not a staging folder of the migration {self.src} -> {self.new} (no "
                f"{RECORD} of it): left as it is; move it away if it is not needed")
            shutil.rmtree(self.staging)  # a stop inside an earlier step 1-3 of this same migration
        parent = os.path.dirname(os.path.abspath(self.staging))
        os.makedirs(parent, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=parent, prefix=os.path.basename(self.staging) + ".tmp-")
        _write_record(self.owner, tmp)
        os.rename(tmp, self.staging)

    def _copy(self):
        """Steps 1-3: the snapshot and the checkpoint rewrite in the staging folder, then the rename (the new run dir
        appears with its record and its checkpoint migrated, never as a plain steps_cosine copy)."""
        step, src_checkpoint = SL.find_latest_checkpoint(os.path.join(self.src, "checkpoints"))
        assert src_checkpoint, f"{self.src}: no checkpoint to migrate"
        self._make_staging()
        try:
            os.makedirs(os.path.join(self.staging, "checkpoints"))
            shutil.copy2(os.path.join(self.src, "metrics.jsonl"), os.path.join(self.staging, "metrics.jsonl"))
            if os.path.exists(os.path.join(self.src, "train.log")):  # a CLI run has one (a test's: beside)
                shutil.copy2(os.path.join(self.src, "train.log"), os.path.join(self.staging, "train.log"))
            path = os.path.join(self.staging, "checkpoints", os.path.basename(src_checkpoint))
            shutil.copy2(src_checkpoint, path)
            self.record = self._record(path, os.path.join(self.staging, "metrics.jsonl"), src_checkpoint, step)
        except AssertionError:  # a refusal: nothing of this migration stays
            shutil.rmtree(self.staging)
            raise
        _write_record(self.record, self.staging)
        self._migrate_checkpoint(path)
        os.rename(self.staging, self.new)

    def _check_destination(self):
        """An existing new run dir: its record must name this source (and the source's recorded checkpoint, when it
        still exists, must be the one copied), the same schedule, and its checkpoint must be the migrated one."""
        record = _read_record(self.new)
        assert record and "entry" in record, f"{self.new} exists without {RECORD}: not a migration made by this tool"
        assert record["src"] == self.src_real, (
            f"{self.new} is the migration of {record['src']}, not of {self.src_real}")
        assert not os.path.exists(record["src_checkpoint"]) or _sha256(record["src_checkpoint"]) == record[
            "src_checkpoint_sha256"], f"{record['src_checkpoint']} changed since {self.new} was migrated from it"
        header = record["header_config"]
        schedule = SL.games_cosine_schedule(header["train_files_sha256"], header["expected_skips"],
                                            self.lr_cycle_games)
        assert schedule == record["schedule"], (
            f"{self.new} was migrated to {record['schedule']}, not {schedule}: give the same --lr-cycle-games")
        self.record = record
        path = os.path.join(self.new, "checkpoints", f"step_{record['step']}.pt")
        assert os.path.exists(path), f"{self.new} exists without step_{record['step']}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert self._is_migrated_checkpoint(checkpoint), (
            f"{path} is not the migrated checkpoint {self.new}/{RECORD} records (config, schedule_history or step "
            "differ)")

    def run(self) -> str:
        """Steps 1-4, from whatever state an earlier stop left; returns what was done."""
        if not os.path.exists(self.new):
            self._copy()
            done = "copied the checkpoint and migrated it"
        else:
            self._check_destination()
            done = ""
        header, _ = SL.Metrics(os.path.join(self.new, "metrics.jsonl")).read()
        if self._is_migrated_header(header):
            return done or "already migrated: nothing to do"
        # header not migrated: only the stop between steps 3 and 4 leaves this, with the snapshot as it was copied
        assert _sha256(os.path.join(self.new, "metrics.jsonl")) == self.record["src_metrics_sha256"], (
            f"{self.new}/metrics.jsonl is neither the migrated header nor the snapshot of the source's file")
        self._migrate_header()
        return (done + "; " if done else "") + "migrated the header"


def migrate(src_run_dir: str, new_run_dir: str, lr_cycle_games: int = None) -> str:
    """The module docstring's migration; what was done, also logged to the new run's train.log."""
    m = Migration(src_run_dir, new_run_dir, lr_cycle_games)
    done = m.run()
    r = m.record
    SL.RunLog(os.path.join(m.new, "train.log"))(
        f"lr schedule migration from {r['src']} (step_{r['step']}.pt, {r['entry']['games']} games read): {done}; "
        f"{r['entry']['from']} -> {r['schedule']}")
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch a steps_cosine run to games_cosine_v1 in a new run dir "
                                                 "(T-067).")
    parser.add_argument("src_run_dir", help="the run to switch (only read)")
    parser.add_argument("new_run_dir", help="the new run dir (must not exist, or hold an interrupted migration)")
    parser.add_argument("--lr-cycle-games", type=SL.positive_int, default=None, metavar="N",
                        help=f"games of one pass (default {SL.LR_CYCLE_GAMES:,}, only on the corpus it was counted on)")
    args = parser.parse_args()
    migrate(args.src_run_dir, args.new_run_dir, args.lr_cycle_games)
