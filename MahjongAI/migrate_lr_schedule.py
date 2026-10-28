"""Switch a steps_cosine run (written before T-067) to the games_cosine_v1 LR schedule, in a NEW run dir (plan T-067).

Run from the repo root: python -m MahjongAI.migrate_lr_schedule <src_run_dir> <new_run_dir> [--lr-cycle-games N]

The source is only read. The migration:
  1. refuses unless the source's latest checkpoint is steps_cosine, it and the source metrics header agree on every
     run-identity key (supervised_learning.config_mismatch) and the validation ids, and the run state has a games
     counter (supervised_learning.games_read_of_state); the new lr_schedule passes the corpus guard
     (games_cosine_schedule: the recorded LR_CYCLE_GAMES on its corpus only, else --lr-cycle-games N);
  2. copies the source's metrics.jsonl, train.log (when it has one) and ONLY its latest checkpoint (older ones would
     stay steps_cosine) into <new_run_dir>.migrating, writes that checkpoint again with the new lr_schedule and a
     schedule_history entry {"at_update", "games", "from", "to"} (atomic: *.tmp, then os.replace), then renames the
     folder to <new_run_dir>;
  3. writes the header the same way (the new config and the same history entry), atomic.
The trainer resumes only when the checkpoint and the header agree: a stop between 2 and 3 leaves "checkpoint new,
header old", which the trainer names as an interrupted migration; running this again completes it (idempotent: a
complete migration is left as it is, a stale <new_run_dir>.migrating is built again). An unrelated mismatch still
fails. The switch sets the LR from the steps_cosine value (0 at a finished run's end) to lr_at(g, k) on purpose.
"""
import argparse
import hashlib
import os
import shutil

import torch

from MahjongAI import supervised_learning as SL


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_atomic(state: dict, path: str):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


class Migration:
    """The migration of one source run (its latest checkpoint) into new_run_dir: the checks of step 1 at construction,
    the target checkpoint config, header config and history entry, and run() for steps 2 and 3."""

    def __init__(self, src_run_dir: str, new_run_dir: str, lr_cycle_games: int = None):
        self.src, self.new = os.path.normpath(src_run_dir), os.path.normpath(new_run_dir)
        assert self.src != self.new, f"{self.src}: the new run dir must be another folder (the source is only read)"
        self.step, self.src_checkpoint = SL.find_latest_checkpoint(os.path.join(self.src, "checkpoints"))
        assert self.src_checkpoint, f"{self.src}: no checkpoint to migrate"
        self.src_metrics = os.path.join(self.src, "metrics.jsonl")
        checkpoint = torch.load(self.src_checkpoint, map_location="cpu", weights_only=True)
        header, _ = SL.Metrics(self.src_metrics).read()
        assert header is not None, f"{self.src_metrics} is missing"
        for where, config in ((self.src_checkpoint, checkpoint["config"]), (self.src_metrics, header["config"])):
            schedule = SL.with_legacy_run(config)["lr_schedule"]
            assert schedule == SL.LEGACY_LR_SCHEDULE, f"{where} is lr_schedule {schedule}, not steps_cosine"
        mismatch = SL.config_mismatch(checkpoint["config"], header["config"])
        assert not mismatch and checkpoint["val_ids"] == header["val_ids"], (
            f"{self.src_checkpoint} and {self.src_metrics} header disagree"
            + (f": {SL._mismatch_text(mismatch)}" if mismatch else ": validation games"))
        assert checkpoint["current_step"] == self.step, f"{self.src_checkpoint} holds step {checkpoint['current_step']}"
        games = SL.games_read_of_state(checkpoint)
        config = SL.with_legacy_run(checkpoint["config"])
        self.schedule = SL.games_cosine_schedule(config["train_files_sha256"], config["expected_skips"], lr_cycle_games)
        self.entry = {"at_update": self.step, "games": games, "from": SL.LEGACY_LR_SCHEDULE["name"],
                      "to": self.schedule["name"]}
        self.checkpoint_config = {**config, "lr_schedule": self.schedule}
        self.header_config = {**SL.with_legacy_run(header["config"]), "lr_schedule": self.schedule}
        self.name = f"step_{self.step}.pt"

    def _is_migrated_checkpoint(self, checkpoint: dict) -> bool:
        # config_mismatch: a trainer stop at this step re-saves it with its own total_steps
        return (not SL.config_mismatch(checkpoint["config"], self.checkpoint_config)
                and checkpoint.get("schedule_history") == [self.entry] and checkpoint["current_step"] == self.step
                and "scheduler_state_dict" not in checkpoint)

    def _is_migrated_header(self, header: dict) -> bool:
        return (not SL.config_mismatch(header["config"], self.header_config)
                and header.get("schedule_history") == [self.entry])

    def _migrate_checkpoint(self, path: str):
        """Step 2's rewrite: the new lr_schedule and the history entry; the LambdaLR state goes (lr_at needs none)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint.pop("scheduler_state_dict")
        checkpoint["config"] = self.checkpoint_config
        checkpoint["schedule_history"] = [self.entry]
        _save_atomic(checkpoint, path)

    def _migrate_header(self):
        """Step 3: the header with the new config and the history entry (Metrics._write: *.tmp, then os.replace)."""
        metrics = SL.Metrics(os.path.join(self.new, "metrics.jsonl"))
        header, rows = metrics.read()
        header["config"] = self.header_config
        header["schedule_history"] = [self.entry]
        metrics._write([header] + rows)

    def _copy(self):
        """Step 2: the copy and the checkpoint rewrite in <new>.migrating, then the rename (the new run dir appears
        with its checkpoint migrated, never as a plain steps_cosine copy)."""
        staging = SL.migration_staging(self.new)
        if os.path.exists(staging):  # a stop inside an earlier step 2: only this module writes there
            shutil.rmtree(staging)
        os.makedirs(os.path.join(staging, "checkpoints"))
        shutil.copy2(self.src_metrics, os.path.join(staging, "metrics.jsonl"))
        if os.path.exists(os.path.join(self.src, "train.log")):  # a run started by the CLI has one (a test's: beside)
            shutil.copy2(os.path.join(self.src, "train.log"), os.path.join(staging, "train.log"))
        path = os.path.join(staging, "checkpoints", self.name)
        shutil.copy2(self.src_checkpoint, path)
        self._migrate_checkpoint(path)
        os.rename(staging, self.new)

    def run(self) -> str:
        """Steps 2 and 3, from whatever state an earlier stop left; returns what was done."""
        if not os.path.exists(self.new):
            self._copy()
            done = "copied the checkpoint and migrated it"
        else:
            path = os.path.join(self.new, "checkpoints", self.name)
            assert os.path.exists(path), f"{self.new} exists without {self.name}: not a migration of {self.src}"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert self._is_migrated_checkpoint(checkpoint), (
                f"{path} is not the migration of {self.src_checkpoint} (config, schedule_history or step differ)")
            done = ""
        header, _ = SL.Metrics(os.path.join(self.new, "metrics.jsonl")).read()
        if self._is_migrated_header(header):
            return done or "already migrated: nothing to do"
        # header not migrated: only the stop between steps 2 and 3 leaves this, with the source's file as it was
        assert _sha256(os.path.join(self.new, "metrics.jsonl")) == _sha256(self.src_metrics), (
            f"{self.new}/metrics.jsonl is neither the migrated header nor the source's file")
        self._migrate_header()
        return (done + "; " if done else "") + "migrated the header"


def migrate(src_run_dir: str, new_run_dir: str, lr_cycle_games: int = None) -> str:
    """Steps 1-3 (the module docstring); what was done, also logged to the new run's train.log."""
    m = Migration(src_run_dir, new_run_dir, lr_cycle_games)
    done = m.run()
    SL.RunLog(os.path.join(m.new, "train.log"))(
        f"lr schedule migration from {m.src} ({m.name}, {m.entry['games']} games read): {done}; "
        f"{m.entry['from']} -> {m.schedule}")
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
