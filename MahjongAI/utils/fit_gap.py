"""Eval-mode fit gap of a checkpoint (plan T-068 R0): CE per head on training-year games and on the validation games.

  python -m MahjongAI.utils.fit_gap runs/<run>/checkpoints/step_<s>.pt
  python -m MahjongAI.utils.fit_gap --test

Both samples go through the trainer's own code: ValSet.load builds them (the validation builder: the first VAL_GAMES
accepted games, each built once, its labels stored) and ValSet.evaluate scores them (eval mode, one game per forward,
float64 CE summed per game), on the CPU with the run's torch_threads, as the trainer's eval does.
  train  the first VAL_GAMES accepted games of TRAIN_YEAR in the loader order of DATA_SEED (the training year and
         seed: epoch-0 games the model trained on)
  val    the run's validation set
CE per head = Σ_games NLL / Σ_games n (supervised_learning.pooled). gap = val CE - train CE (> 0: val is harder for the
model). Reported, not a gate: the two samples differ in year and in difficulty, so the gap has no pass line.

Self-check: the val CE of every head equals the run's logged val CE at the checkpoint's step (the metrics.jsonl next to
the checkpoints dir) within VAL_TOL, and the val ids equal the header's val_ids; else CheckError with the numbers. It
shows that the numbers here come from the trainer's eval path and the trainer's validation games.
"""
import argparse
import contextlib
import io
import os

import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.utils.phase_timing import CheckError
from MahjongAI.utils.run_report import read_metrics

HEADS = SL.HEADS
TRAIN_YEAR = str(SL.TRAIN_YEARS[0])
VAL_TOL = 1e-4  # nats: |val CE here - logged val CE| per head


def metrics_path_of(checkpoint_path: str) -> str:
    """The metrics.jsonl of a checkpoint's run: <run dir>/checkpoints/step_<s>.pt -> <run dir>/metrics.jsonl."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))), "metrics.jsonl")


def logged_row(rows: list, step: int) -> dict:
    """The eval row of `step`; CheckError when the run logged none."""
    found = [r for r in rows if r["step"] == step]
    if len(found) != 1:
        raise CheckError(f"{len(found)} eval rows at step {step} (logged steps {[r['step'] for r in rows][-5:]} ...)")
    return found[0]


def pooled_ce(per_game: dict) -> dict:
    """{head: CE} of ValSet.evaluate's per-game rows (id, n, nll_sum, correct)."""
    return {h: SL.pooled(per_game[h]) for h in HEADS}


def check_val(val_ce: dict, val_ids: list, header: dict, row: dict, tol: float = VAL_TOL) -> float:
    """The self-check: the scored val ids are the header's val_ids, and every head's val CE is the row's within tol.
    Returns the largest |difference|; raises CheckError with every head's numbers."""
    if val_ids != header["val_ids"]:
        raise CheckError(f"val ids differ from the run header's val_ids ({len(val_ids)} / {len(header['val_ids'])})")
    diffs = {h: val_ce[h] - row["heads"][h]["ce"] for h in HEADS}
    if not all(abs(d) <= tol for d in diffs.values()):
        raise CheckError(f"val CE differs from the logged row of step {row['step']} by more than {tol}: " + " | ".join(
            f"{h} here {val_ce[h]:.6f} logged {row['heads'][h]['ce']:.6f} diff {diffs[h]:+.2e}" for h in HEADS))
    return max(abs(d) for d in diffs.values())


def gap_lines(train_ce: dict, val_ce: dict) -> list:
    """The printed table: per head train CE, val CE, gap = val - train."""
    lines = [f"{'head':<8} {'train CE':>9} {'val CE':>9} {'gap (val - train)':>18}"]
    lines += [f"{h:<8} {train_ce[h]:>9.4f} {val_ce[h]:>9.4f} {val_ce[h] - train_ce[h]:>+18.4f}" for h in HEADS]
    return lines


def load_model(checkpoint: dict) -> EncoderConvModel:
    """The checkpoint's model on the CPU (its config's architecture; a pre-T-010 config reads as 64 x 1)."""
    config = SL.with_legacy_arch(checkpoint["config"])
    model = EncoderConvModel(n_layers=config["n_layers"], embd_size=config["embd_size"], verbose=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def fit_gap(checkpoint_path: str, log=print) -> dict:
    """Score the checkpoint on the val set (self-checked against the logged row first) and on the training-year
    sample; print the table. Returns {"train": {head: CE}, "val": {head: CE}, "step": s, "max_diff": x}."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = checkpoint["current_step"]
    header, rows = read_metrics(metrics_path_of(checkpoint_path))
    row = logged_row(rows, step)
    model = load_model(checkpoint)
    threads = torch.get_num_threads()
    torch.set_num_threads(checkpoint["config"].get("torch_threads", SL.TORCH_THREADS))
    try:
        val = SL.ValSet.load(log)
        val_ce = pooled_ce(val.evaluate(model))
        max_diff = check_val(val_ce, val.ids, header, row)
        log(f"self-check: val CE = the logged row of step {step} within {VAL_TOL} (max |diff| {max_diff:.1e})")
        del val
        train = SL.ValSet.load(log, year=TRAIN_YEAR, seed=SL.DATA_SEED)
        train_ce = pooled_ce(train.evaluate(model))
    finally:
        torch.set_num_threads(threads)
    log(f"{checkpoint_path} (step {step}): eval mode, {SL.VAL_GAMES} games each; train = {TRAIN_YEAR} seed "
        f"{SL.DATA_SEED}, val = {header['config']['val_year']} {header['config']['val_days'][0]}.."
        f"{header['config']['val_days'][-1]}")
    for line in gap_lines(train_ce, val_ce):
        log(line)
    return {"train": train_ce, "val": val_ce, "step": step, "max_diff": max_diff}


# ----------------------------------------------------------------------------------------------- self-test


def _synthetic():
    """A header, its row of step 50 and the per-game rows that give that row's CE."""
    ids = [f"g{i}.log" for i in range(4)]
    per_game = {h: [(gid, n, n * (0.3 + 0.1 * j + 0.01 * i), 0) for i, (gid, n) in enumerate(zip(ids, (3, 0, 7, 5)))]
                for j, h in enumerate(HEADS)}
    ce = pooled_ce(per_game)
    row = {"type": "eval", "step": 50, "heads": {h: {"ce": ce[h]} for h in HEADS}}
    return {"val_ids": ids}, [dict(row, step=0), row], per_game


def _expect_failure(fn, needle: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except CheckError as e:
        assert needle in str(e), f"raised, but without {needle!r}: {e}"
        print(f"  dirty case raised as expected: {needle}")
        return
    raise AssertionError(f"dirty case passed: expected a failure with {needle!r}")


def self_test():
    """The clean case passes; a val CE off by more than VAL_TOL, other val ids and a missing row each raise; the table
    holds gap = val - train."""
    header, rows, per_game = _synthetic()
    val_ce = pooled_ce(per_game)
    assert abs(val_ce["during"] - (3 * 0.30 + 7 * 0.32 + 5 * 0.33) / 15) < 1e-12  # Σ NLL / Σ n, not a mean of games
    row = logged_row(rows, 50)
    assert check_val(val_ce, header["val_ids"], header, row) == 0.0
    near = dict(val_ce, post=val_ce["post"] + VAL_TOL / 2)  # inside the tolerance
    assert abs(check_val(near, header["val_ids"], header, row) - VAL_TOL / 2) < 1e-12
    print("  clean case passed (exact and within VAL_TOL)")
    for head in HEADS:  # a forced mismatch on any one head raises
        off = dict(val_ce, **{head: val_ce[head] + 2 * VAL_TOL})
        _expect_failure(lambda: check_val(off, header["val_ids"], header, row), f"{head} here {off[head]:.6f}")
    _expect_failure(lambda: check_val(val_ce, header["val_ids"][::-1], header, row), "val ids differ")
    _expect_failure(lambda: logged_row(rows, 40), "0 eval rows at step 40")
    lines = gap_lines({"during": 0.30, "discard": 0.75, "post": 0.20}, {"during": 0.33, "discard": 0.78, "post": 0.19})
    assert lines[1].split()[-1] == "+0.0300" and lines[3].split()[-1] == "-0.0100", lines
    assert metrics_path_of("/r/x/checkpoints/step_5.pt") == "/r/x/metrics.jsonl"
    print("fit_gap self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="a run's checkpoints/step_<s>.pt")
    ap.add_argument("--test", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit(0)
    assert args.checkpoint, "give a checkpoint"
    fit_gap(args.checkpoint)
