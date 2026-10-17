"""Kill gates and the result figure of a training run's metrics.jsonl (plan T-005).

  python -m MahjongAI.utils.run_report runs/T-005/metrics.jsonl --out <figure.png>
  python -m MahjongAI.utils.run_report --test

Gates, at the final row (step == total_steps), on the same validation games in the same order (ids checked):
  G1  every head beats the no-state prior: Δ = Σ_games (prior NLL - model NLL) / Σ_games n >= MARGIN[head] AND the
      2.5 % quantile of Δ over BOOTSTRAP game resamples > 0. Every head is printed even after a failure.
  G2  discard imitation beats shanten-greedy (ties by the prior): Δtop1 = Σ_games (model hits - rule hits) / Σ_games n,
      2.5 % bootstrap quantile > 0. Only checked once G1 passes (the gates are ordered).
The bootstrap draws BOOTSTRAP resamples of the games with seed 0; the same resample serves both terms of a Δ
(paired), every head and both gates. check_gates prints one line per gate on success and raises with the numbers.
"""
import argparse
import contextlib
import io
import json
import os
from dataclasses import dataclass

import numpy as np

HEADS = ("during", "discard", "post")
MARGIN = {"during": 0.05, "discard": 0.03, "post": 0.03}  # nats, ≈ 4 × the prior's game-clustered SE
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
QUANTILE = 0.025


@dataclass
class GateResult:
    gate: str  # "G1 during", ..., "G2 discard"
    delta: float
    q025: float
    margin: float
    n: int
    passed: bool

    def line(self) -> str:
        need = f" (need >= {self.margin:.2f})" if self.gate.startswith("G1") else ""
        return (f"{self.gate}: {'PASS' if self.passed else 'FAIL'} | Δ {self.delta:+.4f}{need} | "
                f"bootstrap 2.5% {self.q025:+.4f} (need > 0) | n {self.n}")


def read_metrics(path: str):
    """(header, eval rows) of a metrics.jsonl."""
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines and lines[0]["type"] == "header", f"{path}: no header row"
    return lines[0], lines[1:]


def final_row(header: dict, rows: list) -> dict:
    assert rows, "no eval rows"
    steps = [r["step"] for r in rows]
    assert steps == sorted(set(steps)), f"eval steps not strictly increasing: {steps}"
    last = rows[-1]
    assert last["step"] == last["total_steps"], f"the last row is step {last['step']}, the run budget is {last['total_steps']}"
    return last


def _aligned(a: list, b: list, what: str, ids: list):
    """Per-game arrays (n, value_a, value_b) of two per-game lists [id, n, value, ...]; both must list exactly the
    header's val_ids in that order (so every head and gate resamples the same games)."""
    for name, rows in (("reference", a), ("model", b)):
        got = [r[0] for r in rows]
        if got != ids:
            bad = next((i for i, (x, y) in enumerate(zip(got, ids)) if x != y), None)
            raise AssertionError(f"{what}: {name} game ids differ from the header's val_ids "
                                 f"(lengths {len(got)} / {len(ids)}, first mismatch at {bad})")
    ids_a = ids
    n_a, n_b = np.array([r[1] for r in a]), np.array([r[1] for r in b])
    if not np.array_equal(n_a, n_b):
        i = int(np.flatnonzero(n_a != n_b)[0])
        raise AssertionError(f"{what}: sample counts differ at game {ids_a[i]}: {n_a[i]} vs {n_b[i]}")
    return n_a.astype(float), np.array([r[2] for r in a], float), np.array([r[2] for r in b], float)


def resamples(n_games: int) -> np.ndarray:
    """(BOOTSTRAP, n_games) game indices: one fixed draw shared by every Δ."""
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_games, size=(BOOTSTRAP, n_games))


def paired_delta(n: np.ndarray, gain: np.ndarray, idx: np.ndarray):
    """Δ = Σ gain / Σ n, and the QUANTILE of Δ over the resamples `idx` (a resample with Σ n = 0 counts as Δ = 0)."""
    total = n.sum()
    assert total > 0, "no samples"
    nb, gb = n[idx].sum(axis=1), gain[idx].sum(axis=1)
    db = np.divide(gb, nb, out=np.zeros_like(gb), where=nb > 0)
    return gain.sum() / total, float(np.quantile(db, QUANTILE))


def gate_results(header: dict, rows: list) -> list:
    """[GateResult] of G1 (every head), then G2 (only when every G1 head passed)."""
    last = final_row(header, rows)
    ids = header["val_ids"]
    assert ids and len(set(ids)) == len(ids), "header val_ids empty or not unique"
    idx = resamples(len(ids))
    out = []
    for head in HEADS:
        model_games = [[r[0], r[1], r[2]] for r in last["heads"][head]["games"]]
        n, prior_nll, model_nll = _aligned(header["prior"][head], model_games, f"G1 {head}", ids)
        delta, q = paired_delta(n, prior_nll - model_nll, idx)
        out.append(GateResult(f"G1 {head}", delta, q, MARGIN[head], int(n.sum()), delta >= MARGIN[head] and q > 0))
    if all(r.passed for r in out):
        model_hits = [[r[0], r[1], r[3]] for r in last["heads"]["discard"]["games"]]
        n, rule_hits, model = _aligned(header["shanten_greedy"], model_hits, "G2 discard", ids)
        delta, q = paired_delta(n, model - rule_hits, idx)
        out.append(GateResult("G2 discard", delta, q, 0.0, int(n.sum()), q > 0))
    return out


def check_gates(header: dict, rows: list) -> list:
    """Print one line per gate result; raise AssertionError with every line if any gate failed."""
    results = gate_results(header, rows)
    for r in results:
        print(r.line())
    failed = [r for r in results if not r.passed]
    if failed:
        raise AssertionError("gate failed at step {}:\n{}".format(rows[-1]["step"], "\n".join(r.line() for r in results)))
    return results


def _baseline_lines(header: dict, head: str) -> dict:
    """Pooled uniform-legal and prior CE / top-1 of a head (and shanten-greedy top-1 for discard)."""
    def pooled(games, col):
        n = sum(g[1] for g in games)
        return sum(g[col] for g in games) / n

    out = {
        "uniform": (pooled(header["uniform"][head], 2), pooled(header["uniform"][head], 3)),
        "prior": (pooled(header["prior"][head], 2), pooled(header["prior"][head], 3)),
    }
    if head == "discard":
        out["shanten"] = (None, pooled(header["shanten_greedy"], 2))
    return out


def fig_val_vs_prior(header: dict, rows: list, out):
    """2 x 3 panels (CE / top-1 x during / discard / post): val and mean train CE over steps, uniform-legal and the
    no-state prior as horizontal lines, shanten-greedy on discard top-1; final numbers annotated; the title states the
    gate answer. `out`: a path or a binary file object (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        results = gate_results(header, rows)
        verdict = " · ".join(f"{r.gate} {'PASS' if r.passed else 'FAIL'} (Δ {r.delta:+.3f})" for r in results)
        if not all(r.passed for r in results[:len(HEADS)]):
            answer = "G1 fails: state benefit NOT shown on every head, no scaling"
        elif not results[-1].passed:
            answer = ("G1 passes (state benefit on every head), G2 fails (discard imitation does not beat the rule): "
                      "no scaling yet")
        else:
            answer = "G1 and G2 pass: go (T-011 / T-012 speed, then T-010)"
    except AssertionError as e:
        verdict, answer = f"gates not computable: {str(e)[:120]}", "no answer"
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for j, head in enumerate(HEADS):
        base = _baseline_lines(header, head)
        val_ce = [r["heads"][head]["ce"] for r in rows]
        val_top1 = [r["heads"][head]["top1"] for r in rows]
        train = [(r["step"], r["heads"][head]["train_ce"]) for r in rows if r["heads"][head]["train_ce"] is not None]
        ax = axes[0, j]
        ax.plot(steps, val_ce, "o-", color="C0", ms=3, label="model, val (eval mode)")
        if train:
            ax.plot(*zip(*train), "-", color="C0", alpha=0.4, label="model, train (train mode, dropout 0.2)")
        ax.axhline(base["uniform"][0], color="grey", ls=":", label=f"uniform-legal {base['uniform'][0]:.3f}")
        ax.axhline(base["prior"][0], color="C3", ls="--", label=f"no-state prior {base['prior'][0]:.3f}")
        ax.annotate(f"{val_ce[-1]:.3f}", (steps[-1], val_ce[-1]), textcoords="offset points", xytext=(-10, 8),
                    ha="right", color="C0", weight="bold")
        ax.set_title(f"{head}: val cross-entropy (lower is better)")
        ax.set_xlabel("optimizer updates")
        ax.set_ylabel("CE (nats / sample)")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(steps, val_top1, "o-", color="C0", ms=3, label="model, val")
        ax.axhline(base["uniform"][1], color="grey", ls=":", label=f"uniform-legal {base['uniform'][1]:.3f}")
        ax.axhline(base["prior"][1], color="C3", ls="--", label=f"no-state prior {base['prior'][1]:.3f}")
        if "shanten" in base:
            ax.axhline(base["shanten"][1], color="C2", ls="-.", label=f"shanten-greedy {base['shanten'][1]:.3f}")
        ax.annotate(f"{val_top1[-1]:.3f}", (steps[-1], val_top1[-1]), textcoords="offset points", xytext=(-10, -14),
                    ha="right", color="C0", weight="bold")
        ax.set_title(f"{head}: val top-1 = the human action (higher is better)", fontsize=10)
        ax.set_xlabel("optimizer updates")
        ax.set_ylabel("top-1")
        ax.legend(fontsize=8)
    fig.suptitle(f"Does encoder_conv learn from the state? {answer}\n{verdict}\n"
                 f"{len(header['val_ids'])} val games; train CE is in train mode (dropout), so it sits above val early",
                 fontsize=11)
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- self-test


def _synthetic(model_gain: dict, discard_hit_gain: float, n_games: int = 60, seed: int = 1):
    """A header and two eval rows. Game sizes differ; game 3 has no during sample. The model's per-sample NLL is
    the prior's minus model_gain[head] (+ noise); its discard hits are the rule's + discard_hit_gain per sample."""
    rng = np.random.default_rng(seed)
    ids = [f"g{i:03d}.log" for i in range(n_games)]
    header = {"type": "header", "config": {"total_steps": 100}, "val_ids": ids,
              "uniform": {}, "prior": {}, "shanten_greedy": []}
    games = {}
    for head, size in zip(HEADS, (25, 400, 110)):
        n = rng.integers(size // 2, 2 * size, n_games)
        if head == "during":
            n[3] = 0
        prior = n * (0.5 + 0.1 * rng.random(n_games))
        model = prior - n * model_gain[head] + rng.normal(0, 0.02, n_games) * np.sqrt(n)
        model[n == 0] = 0.0
        header["uniform"][head] = [[i, int(k), float(k) * 0.9, float(k) * 0.3] for i, k in zip(ids, n)]
        header["prior"][head] = [[i, int(k), float(p), int(k * 0.5)] for i, k, p in zip(ids, n, prior)]
        hits = np.floor(n * 0.4).astype(int)
        if head == "discard":
            header["shanten_greedy"] = [[i, int(k), int(h)] for i, k, h in zip(ids, n, hits)]
        model_hits = np.clip(hits + np.round(n * discard_hit_gain).astype(int), 0, n)
        games[head] = [[i, int(k), float(m), int(h)] for i, k, m, h in zip(ids, n, model, model_hits)]

    def row(step, g):
        heads = {}
        for head in HEADS:
            n = sum(x[1] for x in g[head])
            heads[head] = {"ce": sum(x[2] for x in g[head]) / n, "top1": sum(x[3] for x in g[head]) / n, "n": n,
                           "games": g[head], "train_ce": None if step == 0 else 0.6, "train_n": 0 if step == 0 else n}
        return {"type": "eval", "step": step, "total_steps": 100, "lr": 0.0, "heads": heads}

    return header, [row(0, header["prior"] | {}), row(100, games)]


def _check_numbers():
    """Δ is a ratio of sums (not a mean of game means) and the bootstrap quantile matches a plain loop."""
    ids = ["a", "b", "c", "d"]
    n = [1, 9, 0, 10]
    prior, model = [2.0, 9.0, 0.0, 10.0], [1.0, 9.0, 0.0, 9.0]  # gains 1, 0, 0, 1 -> Δ = 2 / 20 = 0.1
    header = {"val_ids": ids, "prior": {}, "shanten_greedy": [[i, k, 0] for i, k in zip(ids, n)]}
    heads = {}
    for head in HEADS:
        header["prior"][head] = [[i, k, p, 0] for i, k, p in zip(ids, n, prior)]
        heads[head] = {"games": [[i, k, m, 0] for i, k, m in zip(ids, n, model)]}
    rows = [{"type": "eval", "step": 5, "total_steps": 5, "heads": heads}]
    for r in gate_results(header, rows)[:len(HEADS)]:
        assert abs(r.delta - 0.1) < 1e-12 and r.n == 20, r  # a mean of game means would give (1 + 0 + 0.1) / 3

    rng = np.random.default_rng(7)  # heterogeneous sizes and gains, zero-sample games included
    k = rng.integers(0, 50, 40).astype(float)
    k[:3] = 0
    gain = rng.normal(0.02, 0.3, 40) * k
    idx = resamples(40)
    delta, q = paired_delta(k, gain, idx)
    loop = []
    for row in idx:
        nb = sum(k[j] for j in row)
        loop.append(sum(gain[j] for j in row) / nb if nb else 0.0)
    assert abs(delta - gain.sum() / k.sum()) < 1e-12 and abs(q - float(np.quantile(loop, QUANTILE))) < 1e-12, (delta, q)
    print("  numbers: Δ = ratio of sums, bootstrap quantile = plain loop")


def _expect_failure(header, rows, needle: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_gates(header, rows)
    except AssertionError as e:
        assert needle in str(e), f"raised, but without {needle!r}: {e}"
        print(f"  dirty case raised as expected: {needle}")
        return
    raise AssertionError(f"dirty case passed: expected a failure with {needle!r}")


def self_test():
    """Clean synthetic rows pass; each gate's dirty case raises; an id mismatch raises; the figure renders."""
    clean = {"during": 0.10, "discard": 0.06, "post": 0.06}
    header, rows = _synthetic(clean, 0.05)
    results = check_gates(header, rows)
    assert [r.gate for r in results] == ["G1 during", "G1 discard", "G1 post", "G2 discard"] and all(r.passed for r in results)
    fig_val_vs_prior(header, rows, io.BytesIO())

    for head in HEADS:  # G1 margin: a gain below the margin fails that head only
        _expect_failure(*_synthetic(clean | {head: MARGIN[head] / 2}, 0.05), f"G1 {head}: FAIL")
    # G1 bootstrap: Δ above the margin on average but carried by one game (every other game is worse than the
    # prior) -> the 2.5 % quantile is below 0
    header, rows = _synthetic(clean | {"post": -0.01}, 0.05)
    games = rows[-1]["heads"]["post"]["games"]
    games[0][2] -= MARGIN["post"] * 4 * sum(g[1] for g in games)
    results = gate_results(header, rows)
    g1post = next(r for r in results if r.gate == "G1 post")
    assert g1post.delta >= MARGIN["post"] and g1post.q025 <= 0 and not g1post.passed, g1post
    _expect_failure(header, rows, "G1 post: FAIL")
    _expect_failure(*_synthetic(clean, -0.02), "G2 discard: FAIL")  # G2: the model agrees less than the rule

    # G2 confidence: Δtop1 > 0, carried by one game, every other game below the rule -> the 2.5 % quantile <= 0
    header, rows = _synthetic(clean, -0.004)
    games, rule = rows[-1]["heads"]["discard"]["games"], header["shanten_greedy"]
    rule[0][2], games[0][3] = 0, games[0][1]
    g2 = gate_results(header, rows)[-1]
    assert g2.gate == "G2 discard" and g2.delta > 0 and g2.q025 <= 0 and not g2.passed, g2
    _expect_failure(header, rows, "G2 discard: FAIL")

    _check_numbers()

    header, rows = _synthetic(clean, 0.05)  # id mismatch: the same games in another order
    rows[-1]["heads"]["discard"]["games"] = rows[-1]["heads"]["discard"]["games"][::-1]
    _expect_failure(header, rows, "game ids differ")
    header, rows = _synthetic(clean, 0.05)  # reference and model reordered alike: still not the header's order
    header["prior"]["post"] = header["prior"]["post"][::-1]
    rows[-1]["heads"]["post"]["games"] = rows[-1]["heads"]["post"]["games"][::-1]
    _expect_failure(header, rows, "game ids differ")
    header, rows = _synthetic(clean, 0.05)  # header val_ids that are not the scored games
    header["val_ids"] = [f"other{i}" for i in range(len(header["val_ids"]))]
    _expect_failure(header, rows, "game ids differ")
    header, rows = _synthetic(clean, 0.05)  # a final row short of the run budget
    rows[-1]["step"] = 90
    _expect_failure(header, rows, "the run budget")
    print("run_report self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="?", help="a run's metrics.jsonl")
    ap.add_argument("--out", help="figure path (PNG)")
    ap.add_argument("--test", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit(0)
    assert args.metrics and args.out, "give a metrics.jsonl and --out"
    header, rows = read_metrics(args.metrics)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig_val_vs_prior(header, rows, args.out)  # first: the figure is the result, also when a gate fails
    print(f"figure: {args.out}")
    check_gates(header, rows)
