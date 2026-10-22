"""Self-play match driver (the chess match.py shape): N hanchan between four player specs.

    python -m MahjongAI.engine.match --seats model:CKPT greedy uniform greedy --games 8 --seed 0 --out-dir runs/

Game i: wall seed = --seed + i, every player reseeded with it; seat s is played by seats[(s - i) % 4] (seat
rotation, so every spec sits in every seat equally often over 4k games). Each game saves `<out-dir>/<ts>_<seed>/`
(game.log + decisions.json + meta.json, engine/game.py). `--workers` > 1 spreads the games over processes. Every
seat gets its own player object per game (its own sampling stream, as the replay check rebuilds them); the
checkpoint weights load once per process (players.load_model is memoised). `<out-dir>/summary.json`: per player
identity (the label: `model:<file>@<checkpoint sha256[:12]>`, `greedy`, `uniform`; two checkpoints with one file
name never merge) the mean placement, mean final score, the win and deal-in rates per round, plus every game's
per-seat specification; `<out-dir>/manifest.txt` lists the logs.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import time
from typing import List, Optional, Sequence

from MahjongAI.engine.game import Game
from MahjongAI.engine.players import make_player


def seats_for_game(specs: Sequence[str], i: int) -> List[str]:
    """specs[(s - i) % 4] plays seat s in game i."""
    return [specs[(s - i) % 4] for s in range(4)]


def game_dir(out_dir: str, stamp: str, seed: int) -> str:
    return os.path.join(out_dir, f"{stamp}_{seed:04d}")


def play_one(job: dict) -> dict:
    """One hanchan in this process; returns the summary row (picklable)."""
    specs, seed, temperature = job["specs"], job["seed"], job["temperature"]
    players = [make_player(spec, temperature, seed) for spec in specs]
    labels = [p.label for p in players]
    record = Game(players, labels, seed, temperature).play()
    out = game_dir(job["out_dir"], job["stamp"], seed)
    record.save(out)
    wins, deal_ins = [0] * 4, [0] * 4
    for o in record.outcomes:
        for w in o.winners:
            wins[w] += 1
        if o.kind == "agari" and o.from_who not in o.winners:
            deal_ins[o.from_who] += 1
    return {"seed": seed, "dir": out, "labels": labels, "seats": record.meta["seats"],
            "final_scores": record.meta["final_scores"],
            "placements": record.meta["placements"], "rounds": len(record.outcomes), "wins": wins,
            "deal_ins": deal_ins, "n_decisions": record.meta["n_decisions"], "duration_s": record.meta["duration_s"]}


def summarize(rows: Sequence[dict]) -> dict:
    per = collections.defaultdict(lambda: {"games": 0, "placement_sum": 0, "score_sum": 0, "wins": 0, "deal_ins": 0,
                                           "rounds": 0})
    for r in rows:
        for s, label in enumerate(r["labels"]):
            d = per[label]
            d["games"] += 1
            d["placement_sum"] += r["placements"][s]
            d["score_sum"] += r["final_scores"][s] * 100
            d["wins"] += r["wins"][s]
            d["deal_ins"] += r["deal_ins"][s]
            d["rounds"] += r["rounds"]
    labels = {}
    for label, d in per.items():
        labels[label] = {"games": d["games"], "mean_placement": d["placement_sum"] / d["games"],
                         "mean_score": d["score_sum"] / d["games"], "win_rate": d["wins"] / d["rounds"],
                         "deal_in_rate": d["deal_ins"] / d["rounds"]}
    return {"games": len(rows), "rounds": sum(r["rounds"] for r in rows),
            "mean_duration_s": sum(r["duration_s"] for r in rows) / max(1, len(rows)),
            "mean_decisions": sum(r["n_decisions"] for r in rows) / max(1, len(rows)), "labels": labels}


def run_match(specs: Sequence[str], games: int, seed: int, temperature: float, out_dir: str, workers: int = 1,
              stamp: Optional[str] = None) -> dict:
    assert len(specs) == 4, specs
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [{"specs": seats_for_game(specs, i), "seed": seed + i, "temperature": temperature, "out_dir": out_dir,
             "stamp": stamp} for i in range(games)]
    t0 = time.perf_counter()
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(play_one, jobs, chunksize=1)
    else:
        rows = [play_one(j) for j in jobs]
    rows.sort(key=lambda r: r["seed"])
    summary = summarize(rows)
    summary.update({"specs": list(specs), "seed": seed, "temperature": temperature, "workers": workers,
                    "wall_s": round(time.perf_counter() - t0, 1), "games_list": rows})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("".join(os.path.join(r["dir"], "game.log") + "\n" for r in rows))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seats", nargs=4, required=True, metavar="SPEC",
                    help="four player specs: greedy | uniform | model:<checkpoint> (or a checkpoint path)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    summary = run_match(args.seats, args.games, args.seed, args.temperature, args.out_dir, args.workers)
    for label, d in summary["labels"].items():
        print(f"{label:40s} games {d['games']:4d} placement {d['mean_placement']:.3f} score {d['mean_score']:8.0f} "
              f"win {d['win_rate']:.3f} deal-in {d['deal_in_rate']:.3f}")
    print(f"{summary['games']} games, {summary['rounds']} rounds, {summary['mean_duration_s']:.1f} s per game, "
          f"wall {summary['wall_s']} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
