"""Paired game bootstrap of two runs' val NLL at the same training progress (plan T-068 G2).

  python -m MahjongAI.utils.compare_runs <ref metrics.jsonl> <new metrics.jsonl> --games G [--ref-step S]
  python -m MahjongAI.utils.compare_runs --test

Row of each run: the first eval row with games_read >= G. Rows written before T-067 have no games_read: they must all
come before the rows that have it (a migrated run) and are skipped. --ref-step S takes the ref run's row of step S
instead (a ref run from before T-067, whose rows have no games_read).
Alignment: both rows score the header's val_ids in that order with the same sample count per game and head
(run_report._aligned; the new header's val_ids must be the ref header's); else AssertionError.
Per head: Δ = Σ_games (ref NLL - new NLL) / Σ_games n (> 0: the new run is better), and its 2.5 % and 97.5 % quantiles
over BOOTSTRAP game resamples (run_report.resamples, seed 0: one draw shared by every head).
"""
import argparse
import contextlib
import io
from typing import NamedTuple

import numpy as np

from MahjongAI.utils.phase_timing import CheckError
from MahjongAI.utils.run_report import BOOTSTRAP, BOOTSTRAP_SEED, HEADS, _aligned, paired_delta, read_metrics, resamples

Q_LOW, Q_HIGH = 0.025, 0.975
CE_TOL = 1e-9  # Δ vs the rows' own CE difference: the per-game lists are the ones the logged CE came from


class HeadDelta(NamedTuple):
    head: str
    ref_ce: float
    new_ce: float
    delta: float  # ref - new
    q_low: float
    q_high: float
    n: int


def select_row(rows: list, games: int = None, step: int = None) -> dict:
    """The row of `step`, or (step None) the first row with games_read >= games; CheckError when there is none, or
    when a row without games_read follows one with it."""
    assert (games is None) != (step is None), "give games or step"
    if step is not None:
        found = [r for r in rows if r["step"] == step]
        if len(found) != 1:
            raise CheckError(f"{len(found)} eval rows at step {step} (last step {rows[-1]['step'] if rows else None})")
        return found[0]
    counted = [i for i, r in enumerate(rows) if "games_read" in r]
    if counted and counted != list(range(counted[0], len(rows))):
        raise CheckError(f"rows without games_read after row {rows[counted[0]]['step']}, which has it")
    for i in counted:
        if rows[i]["games_read"] >= games:
            return rows[i]
    last = rows[counted[-1]]["games_read"] if counted else None
    raise CheckError(f"no row with games_read >= {games} (the last row with games_read has {last})")


def compare(ref_header: dict, ref_row: dict, new_header: dict, new_row: dict) -> list:
    """[HeadDelta] per head, the bootstrap drawn once over the val games."""
    ids = ref_header["val_ids"]
    assert ids and len(set(ids)) == len(ids), "ref header val_ids empty or not unique"
    assert new_header["val_ids"] == ids, "the two runs' header val_ids differ"
    idx = resamples(len(ids))
    out = []
    for head in HEADS:
        ref, new = ref_row["heads"][head], new_row["heads"][head]
        n, ref_nll, new_nll = _aligned(ref["games"], new["games"], f"{head} ref vs new", ids)
        delta, q_low = paired_delta(n, ref_nll - new_nll, idx, Q_LOW)
        _, q_high = paired_delta(n, ref_nll - new_nll, idx, Q_HIGH)
        assert abs(delta - (ref["ce"] - new["ce"])) <= CE_TOL, (
            f"{head}: Δ {delta} from the per-game rows, {ref['ce'] - new['ce']} from the rows' CE")
        out.append(HeadDelta(head, ref["ce"], new["ce"], delta, q_low, q_high, int(n.sum())))
    return out


def _where(row: dict) -> str:
    return f"step {row['step']} games_read {row.get('games_read', '-')}"


def report_lines(ref_path: str, ref_row: dict, new_path: str, new_row: dict, results: list, how: dict) -> list:
    lines = [f"ref {ref_path}: {_where(ref_row)} ({how['ref']})",
             f"new {new_path}: {_where(new_row)} ({how['new']})",
             f"Δ = ref - new (> 0: new is better), paired game bootstrap {BOOTSTRAP} resamples seed {BOOTSTRAP_SEED}",
             f"{'head':<8} {'ref CE':>8} {'new CE':>8} {'Δ':>8} {'q2.5':>8} {'q97.5':>8} {'n':>7}"]
    lines += [f"{r.head:<8} {r.ref_ce:>8.4f} {r.new_ce:>8.4f} {r.delta:>+8.4f} {r.q_low:>+8.4f} {r.q_high:>+8.4f} "
              f"{r.n:>7}" for r in results]
    return lines


def compare_runs(ref_path: str, new_path: str, games: int, ref_step: int = None, out=print) -> list:
    """Select, align, compare and print; returns [HeadDelta]."""
    (ref_header, ref_rows), (new_header, new_rows) = read_metrics(ref_path), read_metrics(new_path)
    ref_row = select_row(ref_rows, games=None if ref_step is not None else games, step=ref_step)
    new_row = select_row(new_rows, games=games)
    results = compare(ref_header, ref_row, new_header, new_row)
    first = f"first row with games_read >= {games}"
    for line in report_lines(ref_path, ref_row, new_path, new_row, results,
                             {"ref": f"--ref-step {ref_step}" if ref_step is not None else first, "new": first}):
        out(line)
    return results


# ----------------------------------------------------------------------------------------------- self-test


def _synthetic(gain: float, n_games: int = 40, seed: int = 3):
    """(header, rows) of a ref and a new run: rows at steps 0, 5, 10 with games_read 0, 12, 25; per game sizes differ
    by head, game 2 has no during sample. The new run's per-sample NLL is the ref's minus `gain` (+ noise), at every
    row."""
    rng = np.random.default_rng(seed)
    ids = [f"g{i:03d}.log" for i in range(n_games)]
    sizes = {h: rng.integers(s // 2, 2 * s, n_games) for h, s in zip(HEADS, (25, 400, 110))}
    sizes["during"][2] = 0
    ref_nll = {h: k * (0.5 + 0.3 * rng.random(n_games)) for h, k in sizes.items()}
    new_nll = {h: np.where(k > 0, ref_nll[h] - k * gain + rng.normal(0, 0.05, n_games) * np.sqrt(k), 0.0)
               for h, k in sizes.items()}

    def run(nll):
        rows = []
        for step, g in ((0, 0), (5, 12), (10, 25)):
            heads = {}
            for h in HEADS:
                games = [[i, int(k), float(v), 0] for i, k, v in zip(ids, sizes[h], nll[h])]
                heads[h] = {"ce": sum(x[2] for x in games) / sum(x[1] for x in games), "games": games}
            rows.append({"type": "eval", "step": step, "games_read": g, "heads": heads})
        return {"type": "header", "val_ids": list(ids)}, rows

    return run(ref_nll), run(new_nll)


def _expect_failure(fn, needle: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except AssertionError as e:  # CheckError included
        assert needle in str(e), f"raised, but without {needle!r}: {e}"
        print(f"  dirty case raised as expected: {needle}")
        return
    raise AssertionError(f"dirty case passed: expected a failure with {needle!r}")


def self_test():
    """Row selection (>= G, --ref-step, a pre-T-067 prefix), Δ = a ratio of sums with a bootstrap interval around it,
    and every misalignment raises."""
    (rh, rr), (nh, nr) = _synthetic(0.03)
    assert select_row(rr, games=12)["step"] == 5 and select_row(rr, games=13)["step"] == 10
    assert select_row(rr, games=0)["step"] == 0 and select_row(rr, step=10)["step"] == 10
    legacy = [{k: v for k, v in r.items() if k != "games_read"} for r in rr[:2]] + rr[2:]  # a migrated run
    assert select_row(legacy, games=1)["step"] == 10
    _expect_failure(lambda: select_row(rr, games=26), "no row with games_read >= 26")
    _expect_failure(lambda: select_row(rr, step=7), "0 eval rows at step 7")
    _expect_failure(lambda: select_row(rr[:1] + legacy[1:2] + rr[2:], games=1), "rows without games_read after")
    print("  row selection: first row >= G, --ref-step, a pre-T-067 prefix skipped")

    results = compare(rh, select_row(rr, games=12), nh, select_row(nr, games=12))
    for r in results:
        games_ref = select_row(rr, step=5)["heads"][r.head]["games"]
        games_new = select_row(nr, step=5)["heads"][r.head]["games"]
        total = sum(g[1] for g in games_ref)
        expect = sum(a[2] - b[2] for a, b in zip(games_ref, games_new)) / total
        assert abs(r.delta - expect) < 1e-12 and r.n == total, (r, expect)  # a ratio of sums, not a mean of games
        assert r.q_low < r.delta < r.q_high and r.q_low > 0, r  # gain 0.03 / sample: the interval excludes 0
    flipped = compare(nh, select_row(nr, games=12), rh, select_row(rr, games=12))  # ref and new swapped: Δ flips
    assert all(abs(a.delta + b.delta) < 1e-12 for a, b in zip(results, flipped))
    (rh0, rr0), (nh0, nr0) = _synthetic(0.0)
    assert all(r.q_low < 0 < r.q_high for r in compare(rh0, rr0[1], nh0, nr0[1])), "no gain: the interval holds 0"
    with contextlib.redirect_stdout(io.StringIO()):
        lines = report_lines("a", rr[1], "b", nr[1], results, {"ref": "x", "new": "y"})
    assert len(lines) == 4 + len(HEADS) and lines[-2].split()[0] == "discard"
    print("  numbers: Δ = ratio of sums, interval around it, sign flips with the order")

    def broken(change):
        (rh, rr), (nh, nr) = _synthetic(0.03)
        change(rh, rr[1], nh, nr[1])
        return lambda: compare(rh, rr[1], nh, nr[1])

    def reorder(rh, r, nh, n):
        n["heads"]["discard"]["games"] = n["heads"]["discard"]["games"][::-1]
    _expect_failure(broken(reorder), "game ids differ")

    def count(rh, r, nh, n):
        n["heads"]["post"]["games"][4][1] += 1
    _expect_failure(broken(count), "sample counts differ")

    def other_ids(rh, r, nh, n):
        nh["val_ids"] = nh["val_ids"][1:] + nh["val_ids"][:1]
    _expect_failure(broken(other_ids), "header val_ids differ")

    def short(rh, r, nh, n):
        n["heads"]["during"]["games"] = n["heads"]["during"]["games"][:-1]
    _expect_failure(broken(short), "game ids differ")

    def stale_ce(rh, r, nh, n):
        n["heads"]["discard"]["ce"] += 1e-6
    _expect_failure(broken(stale_ce), "from the rows' CE")
    print("compare_runs self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", help="the reference run's metrics.jsonl")
    ap.add_argument("new", nargs="?", help="the new run's metrics.jsonl")
    ap.add_argument("--games", type=int, help="compare each run's first row with games_read >= G")
    ap.add_argument("--ref-step", type=int, default=None, metavar="S",
                    help="take the ref run's row of step S instead (a run from before T-067: no games_read)")
    ap.add_argument("--test", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit(0)
    assert args.ref and args.new and args.games is not None, "give two metrics.jsonl and --games G"
    compare_runs(args.ref, args.new, args.games, args.ref_step)
