"""Supervised training of encoder_conv on Tenhou logs, scored against a no-state prior (plan T-005).

Run from the repo root as a module: python -m MahjongAI.supervised_learning encoder_conv 12000 --run-dir runs/T-005

Batch modes (T-055): `games` (the default) feeds the model 2-game batches, each head capped at 2,250 sampled rows;
`positions` (--batch-mode positions) feeds it per-head steps from utils/position_batcher.py (one game per draw, rows
carried over): a step takes N_d discard rows and 0, N or 2 N rows of the during / post heads, by their buffers
(position_batcher.side_take). --device mps trains on the Apple GPU; validation always runs on the CPU (a CPU copy of
the model, float64 CE), so val numbers do not depend on the device. A resume must use the same mode, sizes and
device. The header records the git commit; the inputs and the feature code are assumed fixed within a run.

Step semantics: row s of metrics.jsonl = the model after s optimizer updates; row 0 = before any update. A step that
is both an eval and a save step evals (row written) first, then saves. The final step (total_steps) is always both;
a --stop-after-step stop at an earlier step keeps the schedule and adds a checkpoint of that step (no extra eval
row).

Run dir: train.log, metrics.jsonl (a header row, then one row per eval), checkpoints/step_<s>.pt. Starting again on
the same run dir resumes from its latest checkpoint: the run config and the val ids must match (total_steps may
change; a config written before T-010 has no architecture keys and reads as 64 x 1), rows of a later step are dropped,
and a checkpoint step that is an eval step without a row is evaluated first.
"""
import argparse
import copy
import cProfile
import glob
import hashlib
import io
import json
import math
import os
import pathlib
import pstats
import re
import subprocess
from datetime import datetime
from typing import NamedTuple

import torch
import torch.nn.functional as F

from MahjongAI.agents.encoder_conv import (EMBD_SIZE, LEGACY_ARCH, N_LAYERS, EncoderConvModel, PrebuiltGame,
                                           TransformerTensorProcessor, build_game, count_parameters)
from MahjongAI.utils.baselines import fit_bias_prior, greedy_hits, score_bias, shanten_best, uniform_legal
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster, UnexpectedSkip
from MahjongAI.utils.game_workers import GameWorkers, _transportable
from MahjongAI.utils.phase_probe import PhaseProbe, sync_for, sync_reader_for
from MahjongAI.utils.phase_timing import MeasureWindow, PhaseTimer, memory_for
from MahjongAI.utils.position_batcher import DEFAULT_SIZES, PositionBatcher

# Root of the MahjongAI package: data paths resolve the same from any working directory.
_PACKAGE_DIR = pathlib.Path(__file__).parent
DATA_ROOT = _PACKAGE_DIR / "data" / "processed"

HEADS = ("during", "discard", "post")
learning_rate = 1e-3
TRAIN_YEARS = (2021,)
# file order, year interleave and per-epoch reshuffle; the loader position is in the checkpoint, so a resume
# continues where it stopped
DATA_SEED = 0
MODEL_SEED = 0  # torch.manual_seed before the model is built
TORCH_THREADS = 4  # T-047 W2 (after M1 + M2, DATA_WORKERS 4): s/step 4 threads 0.159, 6: 0.162 (T-011 T2 chose 8 with
# 2 workers and the slower model)
VAL_YEAR = "2022"  # time-forward validation: the week after the training year
VAL_DAYS = tuple(f"202201{d:02d}" for d in range(1, 8))
VAL_GAMES = 100  # the first VAL_GAMES accepted games of VAL_DAYS in a seeded order, held in RAM
VAL_SEED = 0
PRIOR_GAMES = 4000  # training games the no-state prior is fitted on (2021, its own loader order)
PRIOR_SEED = 1
PRIOR_L2 = 1e-4
UNIFORM_ROW0_TOL = 0.05  # row 0 (init model) val CE must be within this of uniform-legal CE, per head
EXPECT_PRIOR_CE_TOL = 1e-4
# Exclusions by design (InvalidGameException reasons). Any other skip, a parse error or a tensor-level skip, stops
# the run after a checkpoint. "North round" is not here: 0 in 2.2M games (T-038), so one is unexpected.
EXPECTED_SKIPS = frozenset(f"invalid:{r}" for r in (
    "Three players", "BYE", "Against human", "Red tiles", "Kansaki",
    "Own-turn action after own discard",  # T-046: 7 malformed Tenhou logs in 2.2M games, rejected on purpose
    "Auto-played seat",  # T-043: a seat with a tsumogiri run >= 25 outside riichi, ~100 games a year
))
LOGGED_SKIPS = frozenset({"invalid:Own-turn action after own discard",
                          "invalid:Auto-played seat"})  # rare: each one is logged with its file
EVAL_EVERY_STEPS = 500
SAVE_EVERY_STEPS = 500
# T-011 D1: training games are parsed and tensor-built in DATA_WORKERS spawn processes, DATA_PREFETCH files ahead
# per loader (0 workers = in the trainer, as before). Exact: same games, tensors, skips and loader state.
# Timed s/step (10 threads): 0 workers 0.718, 1: 0.469, 2: 0.332, 3: 0.324 (within 2's spread) -> 2.
# T-047 W2 (drain thread, M1 + M2; workers/threads): 2/8 0.243, 3/6 0.184, 4/4 0.159, 4/6 0.162, 6/2 0.178, 6/4 0.168,
# 6/6 0.165, 8/4 0.170 -> 4 workers, 4 threads.
DATA_WORKERS = 4
DATA_PREFETCH = 8
# T-011 P: the prior's games are parsed and built by their own pool of PRIOR_WORKERS (0 = in the trainer), closed
# when the prior sample is read; the training workers have not started yet. Exact: same games, masks and labels.
# T-047 S1: the validation set is built the same way, by a pool of PRIOR_WORKERS closed when it is read (VAL_BUILD:
# PrebuiltGame plus the header's shanten step). The pool size comes from the core count (14 cores -> 12).
PRIOR_WORKERS = max(1, (os.cpu_count() or 4) - 2)
PRIOR_PREFETCH = 32
MAX_BATCHES_WITHOUT_UPDATE = 50  # consecutive empty / sample-less batches before the run aborts (data failure)
BATCH_MODES = ("games", "positions")
DEVICES = ("cpu", "mps")
# T-055: the run-identity keys of a config written before T-055 (2-game batches on the CPU)
LEGACY_RUN = {"batch_mode": "games", "position_sizes": None, "device": "cpu"}
MPS_MEMORY_EVERY = 100  # steps between the MPS memory log lines (T-055)


class RunLog:
    """Timestamped lines to stdout and to the run's train.log (opened only when a run starts, never at import)."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, message: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {message}"  # ms: T-055 step timing
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def head_tensors(processor: TransformerTensorProcessor, halfturns, tokens) -> dict:
    """{head: (mask (n, A) bool, y (n,), hand37 (n, 37) for the discard head else None)} for ONE game."""
    out = {}
    for head, t in zip(HEADS, processor.prepare_batches([halfturns], [tokens])):
        hand37 = t[1][0][:, 0].sum(dim=-1).round().long() if head == "discard" else None  # x1 plane 0, thermometer
        out[head] = (t[2].bool(), t[3].long(), hand37)
    return out


def _check_tensor_skips(processor: TransformerTensorProcessor, where: str, files=()):
    if processor.skipped:
        raise UnexpectedSkip(f"tensor-level skip {dict(processor.skipped)} in {where}: {[os.path.basename(f) for f in files]}")


def _next_game(loader: DataLoader):
    """(file, halfturns, tokens) of the next accepted game of a batch_size-1 loader."""
    assert loader.batch_size == 1
    halfturns, tokens = next(loader, ([], []))
    if not halfturns:
        raise RuntimeError(f"{loader} ran out of games")
    return loader.last_batch_files[0], halfturns[0], tokens[0]


class ValGame(PrebuiltGame):
    """A validation game: the PrebuiltGame plus the header's bias-free shanten step of its discard samples
    (baselines.shanten_best, bool (n, 37) numpy), or the error that step raised (raised by baseline_header)."""

    def __init__(self, game: PrebuiltGame, shanten_best=None, shanten_error=None):
        super().__init__(game.rounds, game.skipped, game.build_error)
        self.shanten_best, self.shanten_error = shanten_best, shanten_error


def build_val_game(halfturns, tokens, processor_cls=None) -> ValGame:
    """A validation worker's build (T-047 S1): build_game, then the shanten step of the discard samples. A game with a
    build error or a tensor-level skip gets no shanten step: ValSet raises for it first, as the serial path does."""
    game = build_game(halfturns, tokens, processor_cls)
    if game.build_error is not None or game.skipped:
        return ValGame(game)
    try:
        mask, y, hand37 = head_tensors(TransformerTensorProcessor(), game, None)["discard"]
        return ValGame(game, shanten_best(hand37, mask, y).numpy())
    except Exception as e:
        return ValGame(game, shanten_error=_transportable(e, "header"))


VAL_BUILD = build_val_game  # tests force a skip or a failure with a functools.partial of it


class ValSet:
    """The fixed validation games in RAM as PrebuiltGame objects (built once at load), with their per-head masks,
    labels and (discard) hands.

    ids: game file basenames, in evaluation order. Every per-game list (here and in metrics.jsonl) follows `ids`;
    a head with no sample in a game has n = 0 and stays in the list, so the lists of every row align.
    """

    def __init__(self, files, games, processor: TransformerTensorProcessor):
        self.ids = [os.path.basename(f) for f in files]
        assert len(set(self.ids)) == len(self.ids), "duplicate validation game ids"
        assert all(isinstance(g, PrebuiltGame) for g in games), "validation games are PrebuiltGame objects"
        self.games = games  # [PrebuiltGame]: unpacked by prepare_batches, never built again
        self.tensors = [head_tensors(processor, g, None) for g in games]  # per game: {head: (mask, y, hand37)}
        _check_tensor_skips(processor, "the validation set", files)

    @classmethod
    def load(cls, log: RunLog) -> "ValSet":
        """The first VAL_GAMES accepted games, parsed and built (VAL_BUILD) by a pool of PRIOR_WORKERS (0 = in the
        trainer), closed on every exit of the read."""
        workers = GameWorkers(PRIOR_WORKERS, VAL_BUILD, window=PRIOR_PREFETCH) if PRIOR_WORKERS else None
        try:
            loader = DataLoader(str(DATA_ROOT / VAL_YEAR), 1, log, days=VAL_DAYS, seed=VAL_SEED,
                                expected_skips=EXPECTED_SKIPS, logged_skips=LOGGED_SKIPS, workers=workers)
            assert loader.file_list, f"no validation games for {VAL_YEAR} {VAL_DAYS}"
            log(f"Validation: first {VAL_GAMES} accepted games of {len(loader.file_list)} files "
                f"({VAL_YEAR} {VAL_DAYS[0]}..{VAL_DAYS[-1]}, seed {VAL_SEED})")
            games = [_next_game(loader) for _ in range(VAL_GAMES)]
            log(f"  {VAL_GAMES} games from {loader.current_index} files; skipped {dict(loader.skipped)}")
        finally:
            if workers is not None:
                workers.close()
        built = [g if workers is not None else VAL_BUILD(g, t) for _, g, t in games]
        return cls([f for f, _, _ in games], built, TransformerTensorProcessor())

    def head_samples(self, head: str):
        """Per game (id, mask, y, hand37) of one head."""
        return [(gid, *t[head]) for gid, t in zip(self.ids, self.tensors)]

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        """{head: [(game_id, n, nll_sum, correct)]}, one PrebuiltGame per forward call (T-047 E1): the 468d71c
        evaluation bit for bit, without building the game again. A multi-game call is not used: the batch size moves
        logits by ~1e-7 (float reduction order), which can flip a top-1 tie."""
        model.eval()
        try:
            out = {h: [] for h in HEADS}
            for gid, game, tensors in zip(self.ids, self.games, self.tensors):
                results = model([game], [None], sample=False)
                for head in HEADS:
                    y = tensors[head][1]
                    if head not in results:
                        assert len(y) == 0, f"{gid}: no {head} output for {len(y)} samples"
                        out[head].append((gid, 0, 0.0, 0))
                        continue
                    r = results[head]
                    assert torch.equal(r.y.cpu(), y), f"{gid}: {head} labels differ from the stored validation labels"
                    nll = F.cross_entropy(r.logits.double(), r.y, reduction="sum").item()
                    out[head].append((gid, len(y), nll, int((r.logits.argmax(dim=1) == r.y).sum())))
            return out
        finally:
            model.train()


class PriorSample(NamedTuple):
    """The prior's training sample: per head, one mask / label tensor per accepted game, in loader order."""
    files: list
    masks: dict  # {head: [mask (n, A) bool per game]}
    labels: dict  # {head: [y (n,) per game]}
    files_read: int
    skipped: dict


def prior_sample(log: RunLog, workers=None) -> PriorSample:
    """Masks and labels of the first PRIOR_GAMES accepted 2021 games (loader seed PRIOR_SEED), the games parsed and
    built by `workers` when given."""
    loader = DataLoader(str(DATA_ROOT / "2021"), 1, log, seed=PRIOR_SEED, expected_skips=EXPECTED_SKIPS,
                        logged_skips=LOGGED_SKIPS, workers=workers)
    log(f"Prior: fitting on the first {PRIOR_GAMES} accepted games of {len(loader.file_list)} 2021 files (seed {PRIOR_SEED})")
    processor = TransformerTensorProcessor()
    files, masks, labels = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    for k in range(PRIOR_GAMES):  # one game at a time: keep only its masks / labels
        f, h, t = _next_game(loader)
        files.append(f)
        for head, (mask, y, _) in head_tensors(processor, h, t).items():
            masks[head].append(mask)
            labels[head].append(y)
        _check_tensor_skips(processor, "the prior's training games", [f])
        if (k + 1) % 500 == 0:
            log(f"  prior games {k + 1} / {PRIOR_GAMES} (files read {loader.current_index}, skipped {dict(loader.skipped)})")
    return PriorSample(files, masks, labels, loader.current_index, dict(loader.skipped))


def prior_bias(log: RunLog, sample: PriorSample) -> dict:
    """{head: bias} of the no-state prior fitted on `sample`."""
    bias = {}
    for head in HEADS:
        m, y = torch.cat(sample.masks[head]), torch.cat(sample.labels[head])
        bias[head] = fit_bias_prior(m, y, l2=PRIOR_L2)
        nll, _ = score_bias(bias[head], m, y)
        assert torch.isfinite(bias[head]).all() and torch.isfinite(nll).all(), f"prior {head}: non-finite bias or NLL"
        unseen = (torch.bincount(y, minlength=m.shape[1]) == 0).sum().item()
        log(f"  prior {head}: n={len(y)}, train CE {nll.mean().item():.4f}, slots never taken {unseen}")
    return bias


def fit_prior(log: RunLog) -> dict:
    """{head: bias} of the no-state prior, fitted on the first PRIOR_GAMES accepted 2021 games (loader seed PRIOR_SEED).
    The games come from a pool of PRIOR_WORKERS, closed on every exit of the sample read."""
    workers = GameWorkers(PRIOR_WORKERS, build_game, window=PRIOR_PREFETCH) if PRIOR_WORKERS else None
    try:
        sample = prior_sample(log, workers)
    finally:
        if workers is not None:
            workers.close()
    return prior_bias(log, sample)


def baseline_header(val: ValSet, bias: dict) -> dict:
    """Per head, per game: uniform-legal [id, n, nll_sum, p_correct_sum] and prior [id, n, nll_sum, correct]; and
    the discard shanten-greedy hits per game [id, n, correct] (ties broken by the prior's discard bias)."""
    out = {"uniform": {}, "prior": {}, "shanten_greedy": []}
    for head in HEADS:
        out["uniform"][head], out["prior"][head] = [], []
        for game, (gid, mask, y, hand37) in zip(val.games, val.head_samples(head)):
            n = len(y)
            if n == 0:
                out["uniform"][head].append([gid, 0, 0.0, 0.0])
                out["prior"][head].append([gid, 0, 0.0, 0])
                if head == "discard":
                    out["shanten_greedy"].append([gid, 0, 0])
                continue
            u_nll, u_p = uniform_legal(mask, y)
            p_nll, p_hit = score_bias(bias[head], mask, y)
            out["uniform"][head].append([gid, n, u_nll.sum().item(), u_p.sum().item()])
            out["prior"][head].append([gid, n, p_nll.sum().item(), int(p_hit.sum())])
            if head == "discard":
                if game.shanten_error is not None:
                    raise game.shanten_error
                best = torch.from_numpy(game.shanten_best)
                out["shanten_greedy"].append([gid, n, int(greedy_hits(best, y, bias[head]).sum())])
    return out


def _flat(ref) -> list:
    """Per-game rows of a header reference (a {head: rows} dict or a rows list), heads in HEADS order."""
    return [r for h in HEADS for r in ref[h]] if isinstance(ref, dict) else list(ref)


def pooled(rows, nll_col=2):
    """Sum-ratio mean over per-game rows [id, n, value, ...]: Σ value / Σ n."""
    n = sum(r[1] for r in rows)
    return sum(r[nll_col] for r in rows) / n if n else float("nan")


class Metrics:
    """metrics.jsonl: line 1 the header, then one JSON row per eval step, strictly increasing."""

    def __init__(self, path: str):
        self.path = path

    def read(self):
        if not os.path.exists(self.path):
            return None, []
        with open(self.path) as f:
            raw = [line for line in f if line.strip()]
        lines = []
        for i, line in enumerate(raw):
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                if i < len(raw) - 1 or not lines:
                    raise
                # a kill in the middle of an append leaves a partial LAST line: that row is lost, a resume re-runs it
        assert lines and lines[0]["type"] == "header", f"{self.path}: no header row"
        return lines[0], lines[1:]

    def _write(self, lines):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        os.replace(tmp, self.path)

    def start(self, header: dict):
        self._write([header])

    def truncate_after(self, step: int) -> list:
        """Drop rows of a step > `step` (written after the checkpoint a resume starts from); return the kept rows."""
        header, rows = self.read()
        kept = [r for r in rows if r["step"] <= step]
        self._write([header] + kept)
        return kept

    def append(self, row: dict):
        _, rows = self.read()
        assert not rows or rows[-1]["step"] < row["step"], f"row {row['step']} after row {rows[-1]['step']}"
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())


def train_files_sha256(cluster: DataLoaderCluster) -> str:
    """Hash of every training file path (relative to its loader root), in loader order. Called on fresh loaders,
    before a checkpoint's loader state is loaded, so it hashes the epoch-0 order: the run identity, whatever epoch
    a resume continues in."""
    h = hashlib.sha256()
    for dl in cluster.dataloaders:
        for f in dl.file_list:
            h.update(os.path.relpath(f, dl.path).encode() + b"\n")
    return h.hexdigest()


def batch_files_hash(files) -> str:
    """12 hex characters of the sha256 of the game file names (basenames), in order: the per-step log's record of the
    files an update consumed, so two runs' consumed-file sequences can be compared line by line."""
    return hashlib.sha256("\n".join(os.path.basename(f) for f in files).encode()).hexdigest()[:12]


def with_legacy_arch(config: dict) -> dict:
    """The config with the architecture keys of a pre-T-010 run (64 x 1 and its parameter count) when it has neither
    embd_size nor n_layers; any other config unchanged (a config WITH them must match exactly)."""
    if any(k in config for k in LEGACY_ARCH):
        return config
    return {**config, **LEGACY_ARCH, "n_params": count_parameters(**LEGACY_ARCH)}


def with_legacy_run(config: dict) -> dict:
    """with_legacy_arch, and the LEGACY_RUN value of each T-055 key the config does not have (games, cpu)."""
    return {**LEGACY_RUN, **with_legacy_arch(config)}


def config_mismatch(saved: dict, current: dict) -> dict:
    """{key: (saved, current)} of the run-identity keys that differ; total_steps may change on a resume (a run
    extended past its budget), every other key (torch_threads too: it changes the float reduction order) must not.
    A config without architecture keys reads as 64 x 1 (with_legacy_arch), one without T-055 keys as games mode on
    the CPU (with_legacy_run)."""
    saved, current = with_legacy_run(saved), with_legacy_run(current)
    keys = sorted((saved.keys() | current.keys()) - {"total_steps"})
    return {k: (saved.get(k, "<absent>"), current.get(k, "<absent>")) for k in keys
            if k not in saved or k not in current or saved[k] != current[k]}


def _mismatch_text(mismatch: dict) -> str:
    return ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in mismatch.items())


def git_commit() -> str:
    """The code's git commit (`-dirty` when tracked files changed), or None outside a git checkout."""
    def git(*args):
        return subprocess.run(["git", "-C", str(_PACKAGE_DIR), *args], capture_output=True, text=True, check=True).stdout
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no").strip()
        return git("rev-parse", "HEAD").strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def rng_state(device: torch.device) -> dict:
    """The torch RNG states a step draws from: the CPU generator, and the MPS one (dropout on the GPU)."""
    state = {"torch_rng_state": torch.get_rng_state()}
    if device.type == "mps":
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict, device: torch.device):
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps_rng_state"])


def make_workers(timing: bool = False, n: int = None, prefetch: int = None):
    """The run's data workers (started on first use), or None for 0 workers. timing: their T-057 clocks (exposed
    wait, busy time), for --phase-timing / --measure-window only. n / prefetch (T-061 --data-workers / --data-prefetch):
    None = DATA_WORKERS / DATA_PREFETCH."""
    n = DATA_WORKERS if n is None else n
    prefetch = DATA_PREFETCH if prefetch is None else prefetch
    return GameWorkers(n, build_game, window=prefetch, timing=timing) if n else None


def train_cluster(log, batch_size: int, workers=None) -> DataLoaderCluster:
    """The training loaders (one per TRAIN_YEARS year) and their cluster, served by `workers` when given."""
    dataloaders = [
        DataLoader(str(DATA_ROOT / str(yr)), batch_size, log, seed=DATA_SEED, expected_skips=EXPECTED_SKIPS,
                   logged_skips=LOGGED_SKIPS, workers=workers)
        for yr in TRAIN_YEARS
    ]
    return DataLoaderCluster(dataloaders, seed=DATA_SEED)


CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")


def find_latest_checkpoint(checkpoint_dir: str):
    """(step, path) of the highest-step checkpoint in the directory, or (0, None)."""
    found = [(int(m.group(1)), p) for p in glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
             if (m := CHECKPOINT_RE.search(p))]
    return max(found) if found else (0, None)


def train(model_name: str, total_steps: int, run_dir: str, log: RunLog, eval_every: int = EVAL_EVERY_STEPS,
          save_every: int = SAVE_EVERY_STEPS, expect_prior_ce=None, expect_shanten=None, embd_size: int = EMBD_SIZE,
          n_layers: int = N_LAYERS, row0_tol: float = None, batch_mode: str = "games", position_sizes: dict = None,
          device: str = "cpu", stop_after_step: int = None, phase_timing: bool = False, measure_window: tuple = None,
          data_workers: int = None, data_prefetch: int = None, torch_threads: int = None):
    """position_sizes: {head: step size N} of positions mode (default DEFAULT_SIZES): a step takes N discard rows and
    0, N or 2 N during / post rows (position_batcher.side_take); games mode takes none.
    stop_after_step: stop once this many updates are done (after that step's eval / save), with a checkpoint of that
    step; total_steps stays the LR horizon, so a resume continues the same schedule (T-056). None: run to total_steps.
    phase_timing: positions mode only, a PhaseProbe (utils/phase_probe.py) marks each operation of the step, eval and
    save, and the marks give each step's phase times, one `phase-time` line per step (utils/phase_timing.py, T-057); a
    mark syncs the device (MPS). No run-identity key: the training math is unchanged.
    measure_window: positions mode only, (W, M): an outer timer from the completion of update W to that of update
    W + M (right after each scheduler step), the rows trained and admitted in between, the workers' busy time, a
    memory sample at each end, and the one eval and save at step W + M timed apart (T-057). Either T-057 flag turns
    the data workers' clocks on (exposed wait, busy time); with both off they read none. A fresh run only; it needs
    stop_after_step = W + M and eval / save steps at W + M and none inside the window.
    data_workers / data_prefetch / torch_threads (T-061): None = DATA_WORKERS / DATA_PREFETCH / TORCH_THREADS; each
    >= 1. The workers' count and prefetch window change no batch (the loaders own the order) and are no run-identity
    key; torch_threads is one (the config records the value used; it changes the float reduction order). The startup
    pools (prior, validation) keep PRIOR_WORKERS / PRIOR_PREFETCH; the data workers keep one torch thread each."""
    row0_tol = UNIFORM_ROW0_TOL if row0_tol is None else row0_tol
    assert batch_mode in BATCH_MODES and device in DEVICES, (batch_mode, device)
    if batch_mode == "positions":
        position_sizes = {h: (position_sizes or DEFAULT_SIZES)[h] for h in HEADS}
    assert batch_mode == "positions" or position_sizes is None, "position sizes are a positions-mode setting"
    assert device != "mps" or torch.backends.mps.is_available(), "--device mps: this machine has no MPS backend"
    assert stop_after_step is None or stop_after_step > 0, f"stop_after_step {stop_after_step}: must be > 0"
    assert batch_mode == "positions" or not phase_timing, "phase timing is a positions-mode setting"
    assert batch_mode == "positions" or measure_window is None, "the measure window is a positions-mode setting"
    for name, value in (("data_workers", data_workers), ("data_prefetch", data_prefetch),
                        ("torch_threads", torch_threads)):
        assert value is None or (isinstance(value, int) and value >= 1), f"{name} {value!r}: must be an integer >= 1"
    torch_threads = TORCH_THREADS if torch_threads is None else torch_threads
    workers = make_workers(timing=phase_timing or measure_window is not None, n=data_workers, prefetch=data_prefetch)
    try:
        _train(model_name, total_steps, run_dir, log, eval_every, save_every, expect_prior_ce, expect_shanten, workers,
               embd_size, n_layers, row0_tol, batch_mode, position_sizes, torch.device(device),
               stop_after_step, phase_timing, measure_window, torch_threads)
    finally:  # every stop path: normal end, exception, KeyboardInterrupt
        if workers is not None:
            workers.close()


def _train(model_name: str, total_steps: int, run_dir: str, log: RunLog, eval_every: int, save_every: int,
           expect_prior_ce, expect_shanten, workers, embd_size: int, n_layers: int,
           row0_tol: float, batch_mode: str, position_sizes, device: torch.device,
           stop_after_step, phase_timing: bool = False, measure_window: tuple = None,
           torch_threads: int = TORCH_THREADS):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics = Metrics(os.path.join(run_dir, "metrics.jsonl"))
    torch.set_num_threads(torch_threads)

    if model_name == "encoder_conv":
        torch.manual_seed(MODEL_SEED)
        model = EncoderConvModel(n_layers=n_layers, embd_size=embd_size).to(device)  # moved before the optimizer
        batch_size = 2 if batch_mode == "games" else 1
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    # T-010 G0: the built model has the asked architecture (its real dims, not the arguments) and the parameter count
    # of that architecture, which the header records
    n_params = sum(p.numel() for p in model.parameters())
    assert (model.ln_f.normalized_shape, len(model.blocks)) == ((embd_size,), n_layers), (
        f"built ln_f {model.ln_f.normalized_shape} x {len(model.blocks)} blocks, asked {embd_size} x {n_layers}")
    expected_params = count_parameters(embd_size, n_layers)
    assert n_params == expected_params, f"{n_params} parameters, expected {expected_params}"
    log(f"model {model_name}: embd_size {embd_size}, n_layers {n_layers}, {n_params:,} parameters")
    log(f"batch mode {batch_mode}" + (f" {position_sizes}" if position_sizes else "") + f", device {device}")
    # validation runs on the CPU: the model itself on a CPU run, else a CPU copy (deepcopy: no RNG draw) whose weights
    # are refreshed before each eval
    eval_model = model if device.type == "cpu" else copy.deepcopy(model).to("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    warmup_steps = min(1000, total_steps // 10)  # a 300-step smoke must leave warmup; a 1,100-step run must not collapse

    def lr_lambda(step: int) -> float:  # linear warmup, then cosine decay to ~0 at total_steps
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / float(max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    cluster = train_cluster(log, batch_size, workers)
    dataloaders = cluster.dataloaders
    batcher = PositionBatcher(cluster, position_sizes) if batch_mode == "positions" else None
    val = ValSet.load(log)
    train_ids = {os.path.basename(f) for dl in dataloaders for f in dl.file_list}
    assert not train_ids & set(val.ids), f"train and validation games overlap: {sorted(train_ids & set(val.ids))[:5]}"

    config = {
        "model": model_name, "total_steps": total_steps, "batch_size": batch_size, "lr": learning_rate,
        "train_years": list(TRAIN_YEARS), "data_seed": DATA_SEED, "model_seed": MODEL_SEED,
        "train_files": sum(len(dl.file_list) for dl in dataloaders), "train_files_sha256": train_files_sha256(cluster),
        "val_year": VAL_YEAR, "val_days": list(VAL_DAYS), "val_games": VAL_GAMES, "val_seed": VAL_SEED,
        "prior_games": PRIOR_GAMES, "prior_seed": PRIOR_SEED, "prior_l2": PRIOR_L2,
        "expected_skips": sorted(EXPECTED_SKIPS), "torch_threads": torch_threads,
        "embd_size": embd_size, "n_layers": n_layers, "n_params": n_params,
        "batch_mode": batch_mode, "position_sizes": position_sizes, "device": device.type,
    }

    current_step, rows, header = 0, [], None
    train_sums = {h: [0.0, 0] for h in HEADS}  # train-mode batch loss x n, n; since the last eval row
    latest_step, latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
        mismatch = config_mismatch(checkpoint["config"], config)
        assert not mismatch, (f"{latest_checkpoint} is from another run config: {_mismatch_text(mismatch)}; "
                              "resume with the same config or start a new run dir")
        assert checkpoint["val_ids"] == val.ids, f"{latest_checkpoint}: validation games differ"
        if checkpoint["config"]["total_steps"] != total_steps:
            log(f"total_steps {checkpoint['config']['total_steps']} -> {total_steps} (cosine horizon recomputed)")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # moves its state to the parameters' device
        misplaced = {k for state in optimizer.state.values() for k, v in state.items()
                     if torch.is_tensor(v) and k != "step" and v.device.type != device.type}
        assert not misplaced, f"optimizer state {sorted(misplaced)} is not on {device} after the load"
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # The optimizer state restores the LR of the saved schedule; recompute it on THIS run's horizon
        # (total_steps may have changed) so the first resumed update already uses it.
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        cluster.load_state_dict(checkpoint["loader_state"])
        if batcher is not None:
            batcher.load_state_dict(checkpoint["batcher_state"])
        set_rng_state(checkpoint, device)
        current_step = checkpoint["current_step"]
        assert current_step == latest_step, f"{latest_checkpoint} holds step {current_step}"
        bias = {h: checkpoint["prior_bias"][h] for h in HEADS}
        train_sums = {h: list(checkpoint["train_sums"][h]) for h in HEADS}
        header, _ = metrics.read()
        assert header is not None, f"{metrics.path} is missing: cannot resume a run without its header"
        mismatch = config_mismatch(header["config"], config)
        assert not mismatch and header["val_ids"] == val.ids, (
            f"{metrics.path} header is from another run" + (f": {_mismatch_text(mismatch)}" if mismatch else ""))
        rows = metrics.truncate_after(current_step)
        if any(r["step"] == current_step for r in rows):
            # row current_step is on disk, so its interval is closed: a stop after the row's write and before the
            # resets (or inside them) saved that interval's train sums and buffer peaks
            peaks = None if batcher is None else dict(batcher.max_buffered)
            sums = {h: list(train_sums[h]) for h in HEADS}
            train_sums = {h: [0.0, 0] for h in HEADS}
            if batcher is not None:
                batcher.reset_max_buffered()
            if any(sums[h][1] for h in HEADS) or (batcher is not None and peaks != batcher.max_buffered):
                log(f"row {current_step} is on disk: its interval closed (train sums {sums} -> 0"
                    + ("" if batcher is None else f", buffer peaks {peaks} -> {batcher.max_buffered}") + ")")
        log(f"Loaded checkpoint {latest_checkpoint}: resuming at step {current_step}; prior loaded "
            f"(no refit); data loader at {checkpoint['loader_state']}; metrics rows kept {[r['step'] for r in rows]}")
    else:
        log("No checkpoint found. Starting from scratch.")
        bias = fit_prior(log)

    baselines = baseline_header(val, bias)
    if header is not None:  # a resume: the rows to come must be paired with the same per-game references
        changed = {k: sorted({a[0] for a, b in zip(_flat(header[k]), _flat(baselines[k])) if a != b})
                   for k in ("uniform", "prior", "shanten_greedy")}
        assert not any(changed.values()), (
            f"validation references differ from the {metrics.path} header (games {changed}): the val tensors changed "
            "since the run started (parser / tensor fix?); restart from step 0 with a new run dir")
        log("resume check: per-game uniform / prior / shanten-greedy references equal the header's")
    prior_ce = {h: pooled(baselines["prior"][h]) for h in HEADS}
    uniform_ce = {h: pooled(baselines["uniform"][h]) for h in HEADS}
    sg = baselines["shanten_greedy"]
    shanten_hits, discard_n = sum(r[2] for r in sg), sum(r[1] for r in sg)
    log("prior val CE " + " | ".join(f"{h} {prior_ce[h]:.6f}" for h in HEADS)
        + " (uniform-legal " + " | ".join(f"{h} {uniform_ce[h]:.4f}" for h in HEADS) + ")")
    log(f"shanten-greedy discard hits {shanten_hits} / {discard_n} = {shanten_hits / discard_n:.5f} (ties by the prior)")
    log(f"freeze: --expect-prior-ce {','.join(f'{prior_ce[h]:.6f}' for h in HEADS)} --expect-shanten {shanten_hits}")
    if expect_prior_ce is not None:
        diffs = {h: abs(prior_ce[h] - e) for h, e in zip(HEADS, expect_prior_ce)}
        assert max(diffs.values()) <= EXPECT_PRIOR_CE_TOL, (
            f"prior val CE {prior_ce} != expected {dict(zip(HEADS, expect_prior_ce))} (|diff| {diffs}): the val set, "
            "the prior sample or the parser changed; stop and re-plan")
        log(f"pre-run check: prior val CE equals --expect-prior-ce within {EXPECT_PRIOR_CE_TOL}")
    if expect_shanten is not None:
        assert shanten_hits == expect_shanten, (
            f"shanten-greedy hits {shanten_hits} != expected {expect_shanten}: the val set, the prior or the parser "
            "changed; stop and re-plan")
        log("pre-run check: shanten-greedy hits equal --expect-shanten")

    if not latest_checkpoint:
        if os.path.exists(metrics.path):
            log(f"{metrics.path} exists without a checkpoint: overwritten")
        metrics.start({"type": "header", "config": config, "val_ids": val.ids, "started": datetime.now().isoformat(),
                       "git_commit": git_commit(), "prior_bias": {h: bias[h].tolist() for h in HEADS}, **baselines})

    def is_eval_step(s: int) -> bool:
        return s % eval_every == 0 or s == total_steps

    def is_save_step(s: int) -> bool:
        return s % save_every == 0 or s == total_steps

    # --phase-timing (T-057): one probe mark after each operation, in the order of the T-059 mark list; the timer reads
    # each mark's time and times the spans from timer.begin(<mark>), called right before each operation. Every probe
    # call and its arguments sit under `if probe is not None` (every timer call under `if timer is not None`), so a run
    # without the flag does no extra work (no call, no allocation).
    probe = timer = None
    if phase_timing:
        timer = PhaseTimer(log, None if workers is None else (lambda: workers.exposed_wait_s), sync_for(device))
        probe = PhaseProbe(log, sync_reader_for(device), is_eval_step, is_save_step, timer=timer)
    # --measure-window (T-057): every call sits under `if window is not None` (or `window is None`): none without it
    window = None
    if measure_window is not None:
        warmup, measure = measure_window
        last = warmup + measure
        inside = [s for s in range(1, last) if is_eval_step(s) or is_save_step(s)]
        assert not latest_checkpoint, "--measure-window needs a fresh run dir (a resume shifts the window)"
        assert stop_after_step == last, f"--measure-window {warmup},{measure} needs --stop-after-step {last}"
        assert not inside and is_eval_step(last) and is_save_step(last), (
            f"--measure-window {warmup},{measure}: eval / save steps {inside[:5]} inside the window, or none at step "
            f"{last} (use --eval-every {last} --save-every {last})")
        window = MeasureWindow(log, warmup, measure, sync_for(device), batcher,
                               lambda: {h: train_sums[h][1] for h in HEADS}, workers, memory_for(device))

    def phase_stop():
        """The last phase mark (--phase-timing)."""
        if probe is not None:
            probe.mark("stop", step=current_step)

    def log_mps_memory(when: str = ""):
        """One MPS memory sample. when: "" for the every-MPS_MEMORY_EVERY-steps line (before that step's eval / save),
        " (post-eval/save)" right after an eval or a save, " (final)" after the last save of the run (T-056)."""
        if device.type == "mps":
            log(f"mps memory at step {current_step}{when}: driver {torch.mps.driver_allocated_memory() / 2**20:.0f} "
                f"MB, current {torch.mps.current_allocated_memory() / 2**20:.0f} MB")

    def after_eval_save():
        """The memory line after an eval / save, then (--phase-timing) the boundary mark."""
        log_mps_memory(" (post-eval/save)")
        if probe is not None:
            probe.mark("boundary")

    def evaluate():
        """Row `current_step`: val per head and game, mean train CE since the last row. Row 0 is written only if
        it passes the uniform-legal check, so a restart after that stop checks it again."""
        if eval_model is not model:
            eval_model.load_state_dict(model.state_dict())
        if probe is not None:
            probe.mark("eval_weights")
        per_game = val.evaluate(eval_model)
        heads = {}
        for h in HEADS:
            g = per_game[h]
            n = sum(r[1] for r in g)
            heads[h] = {
                "ce": pooled(g), "top1": sum(r[3] for r in g) / n if n else float("nan"), "n": n,
                "games": [list(r) for r in g],
                "train_ce": train_sums[h][0] / train_sums[h][1] if train_sums[h][1] else None,
                "train_n": train_sums[h][1],
            }
        log(f"step {current_step} | val " + " | ".join(
            f"{h} CE {heads[h]['ce']:.4f} (uniform {uniform_ce[h]:.4f}, prior {prior_ce[h]:.4f}) top1 {heads[h]['top1']:.3f} "
            f"n={heads[h]['n']}" + (f" train CE {heads[h]['train_ce']:.4f}" if heads[h]["train_ce"] is not None else "")
            for h in HEADS))
        if current_step == 0:
            gaps = {h: heads[h]["ce"] - uniform_ce[h] for h in HEADS}
            log("row 0 gap to uniform-legal CE: " + " | ".join(f"{h} {gaps[h]:+.4f}" for h in HEADS)
                + f" (|gap| <= {row0_tol} required)")
            assert all(abs(v) <= row0_tol for v in gaps.values()), f"row 0 is not near uniform-legal: {gaps}"
        bad = {h: heads[h]["ce"] for h in HEADS if not math.isfinite(heads[h]["ce"])}
        assert not bad, f"non-finite val CE at step {current_step}: {bad}"
        row = {"type": "eval", "step": current_step, "total_steps": total_steps,
               "lr": optimizer.param_groups[0]["lr"], "heads": heads}
        if batcher is not None:
            row["positions"] = batcher.report()
            log(f"step {current_step} | positions: games admitted {row['positions']['games_admitted']}, buffered "
                f"{row['positions']['buffered']}, max buffered since the last row {row['positions']['max_buffered']}")
        metrics.append(row)
        for h in HEADS:  # reset only once the row is on disk: a stop before keeps the sums in the checkpoint
            train_sums[h] = [0.0, 0]
        if batcher is not None:
            batcher.reset_max_buffered()
        if probe is not None:
            probe.mark("eval")

    def save_checkpoint():
        """Everything a resume needs, written atomically (*.tmp, then os.replace). A stop between a positions-mode
        step's forward and its commit saves the RNG state from before that forward (step_rng): with the step's rows
        still buffered, the resume repeats the step as an unstopped run does."""
        path = os.path.join(checkpoint_dir, f"step_{current_step}.pt")
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "current_step": current_step,
            "loader_state": cluster.state_dict(),
            "prior_bias": bias,
            **(step_rng or rng_state(device)),
            "train_sums": train_sums,
            "config": config,
            "val_ids": val.ids,
        }
        if batcher is not None:
            state["batcher_state"] = batcher.state_dict()
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
        processor = model.tensor_processor if batcher is None else batcher.processor
        log(f"Checkpoint saved at {path}; skipped games: {dict(cluster.skipped())}; "
            f"skipped samples: {dict(processor.skipped)}")
        if probe is not None:
            probe.mark("save")

    # total_steps counts optimizer UPDATES: a batch without samples does not consume the budget, so the
    # cosine horizon and the number of updates agree. A persistent data failure aborts.
    batches_without_update = 0
    consistent = True  # False from optimizer.step() until the step count, RNG snapshot and train sums match the update
    step_rng = None  # positions mode: the RNG state before the forward of a step not committed yet
    if probe is not None:
        probe.mark("init", step=current_step)
    try:
        if is_eval_step(current_step) and not any(r["step"] == current_step for r in rows):
            evaluate()  # row 0 of a fresh run, or the row a stop / kill lost after its checkpoint
            after_eval_save()
        consumed = []  # the files of the batches drawn since the last update, in order (a no-sample batch included)
        while current_step < total_steps:
            if stop_after_step is not None and current_step >= stop_after_step:
                break  # at the top of the loop: the last update, its eval row and its save are done
            if probe is not None:
                probe.mark("top", step=current_step + 1)
            if batcher is None:
                games, tokens = next(cluster)
                consumed += cluster.last_batch_files
                results = model(games, tokens, sample=True) if games else None  # no games: every one was skipped
                _check_tensor_skips(model.tensor_processor, f"the training batch after step {current_step}",
                                    cluster.last_batch_files)
                if not results:
                    batches_without_update += 1
                    if batches_without_update >= MAX_BATCHES_WITHOUT_UPDATE:
                        raise RuntimeError(
                            f"{batches_without_update} consecutive batches gave no training sample at step "
                            f"{current_step} (skipped games: {dict(cluster.skipped())}); check the data"
                        )
                    continue
                batches_without_update = 0
            else:  # positions mode: the step's rows stay buffered until commit, after optimizer.step()
                if timer is not None:
                    timer.begin("peek")
                batches = batcher.peek()
                if probe is not None:
                    probe.mark("peek", heads=tuple(batches))
                if timer is not None:
                    timer.begin("rng")
                step_rng = rng_state(device)
                if probe is not None:
                    probe.mark("rng")
                results = model.forward_heads(batches, mark=None if probe is None else probe.mark)
                if probe is not None:
                    probe.mark("results")  # the previous step's outputs are released by this assignment

            if timer is not None:
                timer.begin("loss")
            total_loss = sum(out.loss for out in results.values())
            if probe is not None:
                probe.mark("loss")
            if timer is not None:
                timer.begin("finite")
            if not torch.isfinite(total_loss):  # before backward: the weights a stop saves are clean
                per_head = {h: f"{out.loss.item():.4f}" for h, out in results.items()}
                raise RuntimeError(f"NaN or Inf in total_loss at update {current_step + 1} (per-head: {per_head})")
            if probe is not None:
                probe.mark("finite")

            if timer is not None:
                timer.begin("zero_grad")
            optimizer.zero_grad()
            if probe is not None:
                probe.mark("zero_grad")
            if timer is not None:
                timer.begin("backward")
            total_loss.backward()
            if probe is not None:
                probe.mark("backward")
            # a finite loss can still give non-finite gradients: raise before optimizer.step, the weights stay clean
            if timer is not None:
                timer.begin("clip")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            if probe is not None:
                probe.mark("clip")
            consistent = False
            if timer is not None:
                timer.begin("opt_step")
            optimizer.step()
            if probe is not None:
                probe.mark("opt_step")
            if timer is not None:
                timer.begin("sched_step")
            scheduler.step()
            if probe is not None:
                probe.mark("sched_step")
            if window is not None:  # the update's completion: the window's endpoint time (a sync on MPS)
                window.update_done(current_step + 1)
            if batcher is not None:
                if timer is not None:
                    timer.begin("commit")
                read, sample_digest = batcher.commit()
                if probe is not None:
                    probe.mark("commit")
            current_step += 1
            step_rng = None
            if timer is not None:
                timer.begin("sums")
            for h, out in results.items():
                train_sums[h][0] += out.loss.item() * out.y.numel()
                train_sums[h][1] += out.y.numel()
            consistent = True
            if probe is not None:
                probe.mark("sums")

            if timer is not None:
                timer.begin("log")
            if batcher is None:
                trace = f" | files {batch_files_hash(consumed)}"
            else:  # the files admitted since the last update, and the provenance of the rows this update trained on
                trace = f" | read {batch_files_hash(read)} | sample {sample_digest}"
            log(f"step {current_step} | " + " | ".join(f"{h} {results[h].loss.item():.4f}" for h in HEADS if h in results)
                + trace)
            consumed = []
            if current_step % MPS_MEMORY_EVERY == 0:
                log_mps_memory()
            if probe is not None:
                probe.mark("log")
            if window is not None:
                window.step_end(current_step)

            if is_eval_step(current_step):
                evaluate() if window is None else window.timed("eval", current_step, evaluate)
            if is_save_step(current_step):
                save_checkpoint() if window is None else window.timed("save", current_step, save_checkpoint)
            if is_eval_step(current_step) or is_save_step(current_step):
                after_eval_save()
        if current_step < total_steps:  # --stop-after-step: a checkpoint of the stop step (none if it is on disk)
            if current_step != latest_step and not is_save_step(current_step):
                save_checkpoint()
            log_mps_memory(" (final)")
            phase_stop()
            log(f"stopped after update {current_step} (--stop-after-step {stop_after_step}, total_steps "
                f"{total_steps}): a start on this run dir resumes here")
            return
    except BaseException as e:
        log(f"STOP at step {current_step}: {type(e).__name__}: {str(e)[:500]}")
        if consistent:
            try:
                save_checkpoint()
            except Exception as save_error:
                log(f"checkpoint on stop failed: {type(save_error).__name__}: {save_error}")
        else:
            log("no checkpoint: the stop came inside an optimizer update")
        raise
    log_mps_memory(" (final)")
    phase_stop()
    log("Training completed.")


def parse_position_sizes(text: str) -> dict:
    values = [int(v) for v in text.split(",")]
    if len(values) != len(HEADS) or min(values) <= 0:
        raise argparse.ArgumentTypeError(f"--position-sizes needs {len(HEADS)} positive counts ({','.join(HEADS)})")
    return dict(zip(HEADS, values))


def parse_measure_window(text: str) -> tuple:
    values = [int(v) for v in text.split(",")]
    if len(values) != 2 or min(values) < 1:
        raise argparse.ArgumentTypeError("--measure-window needs W,M: warm-up and measured updates, both >= 1")
    return tuple(values)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value}: must be >= 1")
    return value


def run_header(args) -> str:
    """The CLI's first log line. The T-061 settings (--data-workers / --data-prefetch / --torch-threads) are appended,
    resolved, only when one of them is given: a run without them logs today's line."""
    tuned = (args.data_workers, args.data_prefetch, args.torch_threads) != (None, None, None)
    return (f"Training {args.model_name} for {args.max_steps} steps in {args.run_dir} (eval every {args.eval_every}, "
            f"save every {args.save_every}, embd_size {args.embd_size}, n_layers {args.n_layers}, batch mode "
            f"{args.batch_mode}, device {args.device})"
            + (f", stop after step {args.stop_after_step}" if args.stop_after_step is not None else "")
            + (", phase timing" if args.phase_timing else "")
            + (f", measure window {args.measure_window[0]},{args.measure_window[1]}" if args.measure_window else "")
            + (f", data workers {DATA_WORKERS if args.data_workers is None else args.data_workers}, data prefetch "
               f"{DATA_PREFETCH if args.data_prefetch is None else args.data_prefetch}, torch threads "
               f"{TORCH_THREADS if args.torch_threads is None else args.torch_threads}" if tuned else "")
            + (" [profiling on]" if args.profile else ""))


def parse_expect_prior_ce(text: str):
    values = [float(v) for v in text.split(",")]
    if len(values) != len(HEADS):
        raise argparse.ArgumentTypeError(f"--expect-prior-ce needs {len(HEADS)} values ({','.join(HEADS)})")
    return values


if __name__ == "__main__":
    valid_models = [
        "encoder_conv",
        # transformer_conv is archived (agents/archive/transformer_conv.py): it returns (logits, loss) tuples and
        # reads the old inputs (plan T-013)
    ]

    parser = argparse.ArgumentParser(description="Train a MahjongAI model.")
    parser.add_argument("model_name", choices=valid_models, help="Model to train.")
    parser.add_argument("max_steps", type=int, help="Total number of optimizer updates.")
    parser.add_argument("--run-dir", default=None,
                        help="train.log, metrics.jsonl, checkpoints/ (default runs/<model>_<timestamp>); "
                             "an existing run dir resumes from its latest checkpoint")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_STEPS)
    parser.add_argument("--expect-prior-ce", type=parse_expect_prior_ce, default=None, metavar="D,C,P",
                        help="stop unless the prior's val CE per head (during, discard, post) equals these within 1e-4")
    parser.add_argument("--expect-shanten", type=int, default=None, metavar="N",
                        help="stop unless the shanten-greedy discard hits on the val set equal N")
    parser.add_argument("--embd-size", type=int, default=EMBD_SIZE, help="model width (T-010)")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS, help="encoder blocks (T-010)")
    parser.add_argument("--row0-tol", type=float, default=None, metavar="T",
                        help=f"row-0 |val CE - uniform-legal CE| limit per head (default {UNIFORM_ROW0_TOL}); a wider "
                             "init model starts further from uniform (T-010 128x2: post -0.069)")
    parser.add_argument("--batch-mode", choices=BATCH_MODES, default="games",
                        help="games: 2-game batches, sampled per head (default); positions: per-head steps of the "
                             "position sizes (discard N rows; during / post 0, N or 2 N), rows carried over (T-055)")
    parser.add_argument("--position-sizes", type=parse_position_sizes, default=None, metavar="U,D,P",
                        help="positions mode: step sizes N of the during, discard, post heads; a step takes N "
                             "discard rows and 0, N or 2 N during / post rows (default "
                             f"{','.join(str(DEFAULT_SIZES[h]) for h in HEADS)})")
    parser.add_argument("--device", choices=DEVICES, default="cpu", help="training device (validation: the CPU)")
    parser.add_argument("--stop-after-step", type=int, default=None, metavar="K",
                        help="stop after update K with a checkpoint; max_steps stays the LR horizon (T-056)")
    parser.add_argument("--phase-timing", action="store_true",
                        help="positions mode: one line per step with the time of each phase (wait, admit, transfer, "
                             "forward, backward, update, overhead), from the phase probe's marks (a device sync each; "
                             "T-057)")
    parser.add_argument("--measure-window", type=parse_measure_window, default=None, metavar="W,M",
                        help="positions mode, a fresh run: time updates W+1..W+M (outer timer, rows, worker busy time), "
                             "then the one eval and save at W+M apart; needs --stop-after-step, --eval-every and "
                             "--save-every W+M (T-057)")
    parser.add_argument("--data-workers", type=positive_int, default=None, metavar="N",
                        help=f"training data worker processes (default {DATA_WORKERS}; T-061); no batch changes")
    parser.add_argument("--data-prefetch", type=positive_int, default=None, metavar="P",
                        help=f"files a training loader keeps submitted ahead (default {DATA_PREFETCH}; T-061)")
    parser.add_argument("--torch-threads", type=positive_int, default=None, metavar="T",
                        help=f"the trainer's torch threads, training and CPU eval (default {TORCH_THREADS}; T-061); a "
                             "run-identity key: a resume needs the same T")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and print the top-50 slowest calls on exit.",
    )
    parser.add_argument(
        "--profile-out",
        default=None,
        metavar="FILE",
        help="Save raw cProfile stats to FILE (e.g. profile.out) for later analysis with snakeviz.",
    )
    args = parser.parse_args()
    assert args.eval_every > 0 and args.save_every > 0
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        parser.error("--stop-after-step needs K > 0")
    if args.position_sizes is not None and args.batch_mode != "positions":
        parser.error("--position-sizes needs --batch-mode positions")
    if (args.phase_timing or args.measure_window) and args.batch_mode != "positions":
        parser.error("--phase-timing and --measure-window need --batch-mode positions")
    if args.measure_window and not (args.stop_after_step == args.eval_every == args.save_every
                                    == sum(args.measure_window)):
        last = sum(args.measure_window)
        parser.error(f"--measure-window W,M needs --stop-after-step {last} --eval-every {last} --save-every {last}")

    run_dir = args.run_dir = (args.run_dir or os.path.join(
        "runs", f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(run_dir, exist_ok=True)
    log = RunLog(os.path.join(run_dir, "train.log"))
    log(run_header(args))

    def _run():
        train(args.model_name, args.max_steps, run_dir, log, args.eval_every, args.save_every,
              args.expect_prior_ce, args.expect_shanten, args.embd_size, args.n_layers, args.row0_tol,
              args.batch_mode, args.position_sizes, args.device, args.stop_after_step, args.phase_timing,
              args.measure_window, args.data_workers, args.data_prefetch, args.torch_threads)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        try:
            _run()
        finally:
            pr.disable()
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(50)  # top-50 by cumulative time
            print(buf.getvalue())
            with open(log.path, "a") as f:
                f.write("\n=== cProfile report ===\n" + buf.getvalue())
            if args.profile_out:
                pr.dump_stats(args.profile_out)
                log(f"cProfile stats saved to {args.profile_out}")
    else:
        _run()
