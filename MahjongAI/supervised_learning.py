"""Supervised training of encoder_conv on Tenhou logs, scored against a no-state prior (plan T-005).

Run from the repo root as a module: python -m MahjongAI.supervised_learning encoder_conv 12000 --run-dir runs/T-005

Step semantics: row s of metrics.jsonl = the model after s optimizer updates; row 0 = before any update. A step that
is both an eval and a save step evals (row written) first, then saves. The final step is always both.

Run dir: train.log, metrics.jsonl (a header row, then one row per eval), checkpoints/step_<s>.pt. Starting again on
the same run dir resumes from its latest checkpoint: the run config and the val ids must match (total_steps may
change), rows of a later step are dropped, and a checkpoint step that is an eval step without a row is evaluated first.
"""
import argparse
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
from datetime import datetime

import torch
import torch.nn.functional as F

from MahjongAI.agents.encoder_conv import EncoderConvModel, TransformerTensorProcessor
from MahjongAI.utils.baselines import fit_bias_prior, score_bias, shanten_greedy, uniform_legal
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster, UnexpectedSkip

# Root of the MahjongAI package: data paths resolve the same from any working directory.
_PACKAGE_DIR = pathlib.Path(__file__).parent
DATA_ROOT = _PACKAGE_DIR / "data" / "processed"

HEADS = ("during", "discard", "post")
learning_rate = 1e-3
TRAIN_YEARS = (2021,)
DATA_SEED = 0  # training file order; the loader position is in the checkpoint, so a resume continues where it stopped
MODEL_SEED = 0  # torch.manual_seed before the model is built
TORCH_THREADS = 10
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
))
LOGGED_SKIPS = frozenset({"invalid:Own-turn action after own discard"})  # rare: each one is logged with its file
EVAL_EVERY_STEPS = 500
SAVE_EVERY_STEPS = 500
MAX_BATCHES_WITHOUT_UPDATE = 50  # consecutive empty / sample-less batches before the run aborts (data failure)
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")


class RunLog:
    """Timestamped lines to stdout and to the run's train.log (opened only when a run starts, never at import)."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, message: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
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


class ValSet:
    """The fixed validation games in RAM, with their per-head masks, labels and (discard) hands.

    ids: game file basenames, in evaluation order. Every per-game list (here and in metrics.jsonl) follows `ids`;
    a head with no sample in a game has n = 0 and stays in the list, so the lists of every row align.
    """

    def __init__(self, files, games, processor: TransformerTensorProcessor):
        self.ids = [os.path.basename(f) for f in files]
        assert len(set(self.ids)) == len(self.ids), "duplicate validation game ids"
        self.games = games  # [(halfturns, tokens)]
        self.tensors = [head_tensors(processor, h, t) for h, t in games]  # per game: {head: (mask, y, hand37)}
        _check_tensor_skips(processor, "the validation set", files)

    @classmethod
    def load(cls, log: RunLog) -> "ValSet":
        loader = DataLoader(str(DATA_ROOT / VAL_YEAR), 1, log, days=VAL_DAYS, seed=VAL_SEED,
                            expected_skips=EXPECTED_SKIPS, logged_skips=LOGGED_SKIPS)
        assert loader.file_list, f"no validation games for {VAL_YEAR} {VAL_DAYS}"
        log(f"Validation: first {VAL_GAMES} accepted games of {len(loader.file_list)} files "
            f"({VAL_YEAR} {VAL_DAYS[0]}..{VAL_DAYS[-1]}, seed {VAL_SEED})")
        games = [_next_game(loader) for _ in range(VAL_GAMES)]
        log(f"  {VAL_GAMES} games from {loader.current_index} files; skipped {dict(loader.skipped)}")
        return cls([f for f, _, _ in games], [(h, t) for _, h, t in games], TransformerTensorProcessor())

    def head_samples(self, head: str):
        """Per game (id, mask, y, hand37) of one head."""
        return [(gid, *t[head]) for gid, t in zip(self.ids, self.tensors)]

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        """{head: [(game_id, n, nll_sum, correct)]}, one game per forward call (bounds attention memory)."""
        model.eval()
        try:
            out = {h: [] for h in HEADS}
            for gid, (halfturns, tokens), tensors in zip(self.ids, self.games, self.tensors):
                results = model([halfturns], [tokens], sample=False)
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


def fit_prior(log: RunLog) -> dict:
    """{head: bias} of the no-state prior, fitted on the first PRIOR_GAMES accepted 2021 games (loader seed PRIOR_SEED)."""
    loader = DataLoader(str(DATA_ROOT / "2021"), 1, log, seed=PRIOR_SEED, expected_skips=EXPECTED_SKIPS,
                        logged_skips=LOGGED_SKIPS)
    log(f"Prior: fitting on the first {PRIOR_GAMES} accepted games of {len(loader.file_list)} 2021 files (seed {PRIOR_SEED})")
    processor = TransformerTensorProcessor()
    masks, labels = {h: [] for h in HEADS}, {h: [] for h in HEADS}
    for k in range(PRIOR_GAMES):  # parse one game at a time and keep only its masks / labels
        f, h, t = _next_game(loader)
        for head, (mask, y, _) in head_tensors(processor, h, t).items():
            masks[head].append(mask)
            labels[head].append(y)
        _check_tensor_skips(processor, "the prior's training games", [f])
        if (k + 1) % 500 == 0:
            log(f"  prior games {k + 1} / {PRIOR_GAMES} (files read {loader.current_index}, skipped {dict(loader.skipped)})")
    bias = {}
    for head in HEADS:
        m, y = torch.cat(masks[head]), torch.cat(labels[head])
        bias[head] = fit_bias_prior(m, y, l2=PRIOR_L2)
        nll, _ = score_bias(bias[head], m, y)
        assert torch.isfinite(bias[head]).all() and torch.isfinite(nll).all(), f"prior {head}: non-finite bias or NLL"
        unseen = (torch.bincount(y, minlength=m.shape[1]) == 0).sum().item()
        log(f"  prior {head}: n={len(y)}, train CE {nll.mean().item():.4f}, slots never taken {unseen}")
    return bias


def baseline_header(val: ValSet, bias: dict) -> dict:
    """Per head, per game: uniform-legal [id, n, nll_sum, p_correct_sum] and prior [id, n, nll_sum, correct]; and
    the discard shanten-greedy hits per game [id, n, correct] (ties broken by the prior's discard bias)."""
    out = {"uniform": {}, "prior": {}, "shanten_greedy": []}
    for head in HEADS:
        out["uniform"][head], out["prior"][head] = [], []
        for gid, mask, y, hand37 in val.head_samples(head):
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
                out["shanten_greedy"].append([gid, n, int(shanten_greedy(hand37, mask, y, bias[head]).sum())])
    return out


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
    """Hash of every training file path (relative to its loader root), in loader order."""
    h = hashlib.sha256()
    for dl in cluster.dataloaders:
        for f in dl.file_list:
            h.update(os.path.relpath(f, dl.path).encode() + b"\n")
    return h.hexdigest()


CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")


def find_latest_checkpoint(checkpoint_dir: str):
    """(step, path) of the highest-step checkpoint in the directory, or (0, None)."""
    found = [(int(m.group(1)), p) for p in glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
             if (m := CHECKPOINT_RE.search(p))]
    return max(found) if found else (0, None)


def train(model_name: str, total_steps: int, run_dir: str, log: RunLog, eval_every: int = EVAL_EVERY_STEPS,
          save_every: int = SAVE_EVERY_STEPS, expect_prior_ce=None, expect_shanten=None):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics = Metrics(os.path.join(run_dir, "metrics.jsonl"))
    torch.set_num_threads(TORCH_THREADS)

    if model_name == "encoder_conv":
        torch.manual_seed(MODEL_SEED)
        model = EncoderConvModel().to(device)
        batch_size = 2
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    warmup_steps = min(1000, total_steps // 10)  # a 300-step smoke must leave warmup; a 1,100-step run must not collapse

    def lr_lambda(step: int) -> float:  # linear warmup, then cosine decay to ~0 at total_steps
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / float(max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    dataloaders = [
        DataLoader(str(DATA_ROOT / str(yr)), batch_size, log, seed=DATA_SEED, expected_skips=EXPECTED_SKIPS,
                   logged_skips=LOGGED_SKIPS)
        for yr in TRAIN_YEARS
    ]
    cluster = DataLoaderCluster(dataloaders)
    val = ValSet.load(log)
    train_ids = {os.path.basename(f) for dl in dataloaders for f in dl.file_list}
    assert not train_ids & set(val.ids), f"train and validation games overlap: {sorted(train_ids & set(val.ids))[:5]}"

    config = {
        "model": model_name, "total_steps": total_steps, "batch_size": batch_size, "lr": learning_rate,
        "train_years": list(TRAIN_YEARS), "data_seed": DATA_SEED, "model_seed": MODEL_SEED,
        "train_files": sum(len(dl.file_list) for dl in dataloaders), "train_files_sha256": train_files_sha256(cluster),
        "val_year": VAL_YEAR, "val_days": list(VAL_DAYS), "val_games": VAL_GAMES, "val_seed": VAL_SEED,
        "prior_games": PRIOR_GAMES, "prior_seed": PRIOR_SEED, "prior_l2": PRIOR_L2,
        "expected_skips": sorted(EXPECTED_SKIPS), "torch_threads": TORCH_THREADS,
    }

    def identity(c: dict) -> dict:  # total_steps may change on a resume (a run extended past its budget)
        return {k: v for k, v in c.items() if k != "total_steps"}

    current_step, rows = 0, []
    train_sums = {h: [0.0, 0] for h in HEADS}  # train-mode batch loss x n, n; since the last eval row
    latest_step, latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
        assert identity(checkpoint["config"]) == identity(config), (
            f"{latest_checkpoint} is from another run config:\n{checkpoint['config']}\nvs\n{config}")
        assert checkpoint["val_ids"] == val.ids, f"{latest_checkpoint}: validation games differ"
        if checkpoint["config"]["total_steps"] != total_steps:
            log(f"total_steps {checkpoint['config']['total_steps']} -> {total_steps} (cosine horizon recomputed)")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # The optimizer state restores the LR of the saved schedule; recompute it on THIS run's horizon
        # (total_steps may have changed) so the first resumed update already uses it.
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        cluster.load_state_dict(checkpoint["loader_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        current_step = checkpoint["current_step"]
        assert current_step == latest_step, f"{latest_checkpoint} holds step {current_step}"
        bias = {h: checkpoint["prior_bias"][h] for h in HEADS}
        train_sums = {h: list(checkpoint["train_sums"][h]) for h in HEADS}
        header, _ = metrics.read()
        assert header is not None, f"{metrics.path} is missing: cannot resume a run without its header"
        assert identity(header["config"]) == identity(config) and header["val_ids"] == val.ids, (
            f"{metrics.path} header is from another run")
        rows = metrics.truncate_after(current_step)
        log(f"Loaded checkpoint {latest_checkpoint}: resuming at step {current_step}; prior loaded "
            f"(no refit); data loader at {checkpoint['loader_state']}; metrics rows kept {[r['step'] for r in rows]}")
    else:
        log("No checkpoint found. Starting from scratch.")
        bias = fit_prior(log)

    baselines = baseline_header(val, bias)
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
                       "prior_bias": {h: bias[h].tolist() for h in HEADS}, **baselines})

    def is_eval_step(s: int) -> bool:
        return s % eval_every == 0 or s == total_steps

    def is_save_step(s: int) -> bool:
        return s % save_every == 0 or s == total_steps

    def evaluate():
        """Row `current_step`: val per head and game, mean train CE since the last row. Row 0 is written only if
        it passes the uniform-legal check, so a restart after that stop checks it again."""
        per_game = val.evaluate(model)
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
                + f" (|gap| <= {UNIFORM_ROW0_TOL} required)")
            assert all(abs(v) <= UNIFORM_ROW0_TOL for v in gaps.values()), f"row 0 is not near uniform-legal: {gaps}"
        bad = {h: heads[h]["ce"] for h in HEADS if not math.isfinite(heads[h]["ce"])}
        assert not bad, f"non-finite val CE at step {current_step}: {bad}"
        metrics.append({"type": "eval", "step": current_step, "total_steps": total_steps,
                        "lr": optimizer.param_groups[0]["lr"], "heads": heads})
        for h in HEADS:  # reset only once the row is on disk: a stop before keeps the sums in the checkpoint
            train_sums[h] = [0.0, 0]

    def save_checkpoint():
        """Everything a resume needs, written atomically (*.tmp, then os.replace)."""
        path = os.path.join(checkpoint_dir, f"step_{current_step}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "current_step": current_step,
            "loader_state": cluster.state_dict(),
            "prior_bias": bias,
            "torch_rng_state": torch.get_rng_state(),
            "train_sums": train_sums,
            "config": config,
            "val_ids": val.ids,
        }, path + ".tmp")
        os.replace(path + ".tmp", path)
        log(f"Checkpoint saved at {path}; skipped games: {dict(cluster.skipped())}; "
            f"skipped samples: {dict(model.tensor_processor.skipped)}")

    # total_steps counts optimizer UPDATES: a batch without samples does not consume the budget, so the
    # cosine horizon and the number of updates agree. A persistent data failure aborts.
    batches_without_update = 0
    consistent = True  # False only between optimizer.step() and current_step += 1: no checkpoint from there
    try:
        if is_eval_step(current_step) and not any(r["step"] == current_step for r in rows):
            evaluate()  # row 0 of a fresh run, or the row a stop / kill lost after its checkpoint
        while current_step < total_steps:
            games, tokens = next(cluster)
            results = model(games, tokens, sample=True) if games else None  # no games: every one was skipped
            _check_tensor_skips(model.tensor_processor, f"the training batch after step {current_step}",
                                cluster.last_batch_files)
            if not results:
                batches_without_update += 1
                if batches_without_update >= MAX_BATCHES_WITHOUT_UPDATE:
                    raise RuntimeError(
                        f"{batches_without_update} consecutive batches gave no training sample at step {current_step} "
                        f"(skipped games: {dict(cluster.skipped())}); check the data"
                    )
                continue
            batches_without_update = 0

            total_loss = sum(out.loss for out in results.values())
            if not torch.isfinite(total_loss):  # before backward: the weights a stop saves are clean
                per_head = {h: f"{out.loss.item():.4f}" for h, out in results.items()}
                raise RuntimeError(f"NaN or Inf in total_loss at update {current_step + 1} (per-head: {per_head})")

            optimizer.zero_grad()
            total_loss.backward()
            # a finite loss can still give non-finite gradients: raise before optimizer.step, the weights stay clean
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            consistent = False
            optimizer.step()
            scheduler.step()
            current_step += 1
            consistent = True

            for h, out in results.items():
                train_sums[h][0] += out.loss.item() * out.y.numel()
                train_sums[h][1] += out.y.numel()
            log(f"step {current_step} | " + " | ".join(f"{h} {results[h].loss.item():.4f}" for h in HEADS if h in results))

            if is_eval_step(current_step):
                evaluate()
            if is_save_step(current_step):
                save_checkpoint()
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
    log("Training completed.")


def parse_expect_prior_ce(text: str):
    values = [float(v) for v in text.split(",")]
    if len(values) != len(HEADS):
        raise argparse.ArgumentTypeError(f"--expect-prior-ce needs {len(HEADS)} values ({','.join(HEADS)})")
    return values


if __name__ == "__main__":
    valid_models = [
        "encoder_conv",
        # transformer_conv.py is archived: it returns (logits, loss) tuples and reads the old inputs (plan T-013)
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

    run_dir = args.run_dir or os.path.join("runs", f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    log = RunLog(os.path.join(run_dir, "train.log"))
    log(f"Training {args.model_name} for {args.max_steps} steps in {run_dir} (eval every {args.eval_every}, "
        f"save every {args.save_every})" + (" [profiling on]" if args.profile else ""))

    def _run():
        train(args.model_name, args.max_steps, run_dir, log, args.eval_every, args.save_every,
              args.expect_prior_ce, args.expect_shanten)

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
