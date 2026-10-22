"""G2 display-state checks of an exported game (T-052). Every check raises ViewerCheckError (with `.check` naming the
check and the offending numbers in the message); `check_log` prints one line on success.

  counts      (a) each seat's concealed count = 13 - 3 * melds (+1 for the actor at a during / discard point)
  ownership   (b) every 136-id in a hand, a meld, a pond (called tiles excluded) or a public indicator is unique,
                  and their number = 52 + draws so far + indicators flipped; draws left = 70 - draws so far
  snap        (c) the display hands, indicators and actor equal Snap.ids / Snap.doras / Snap.actor of
                  gates.xml_rounds at every decision point after the same BYE cut
  actual      (d) a discard point's actual action = DiscardTurn.discarded_tile; a during / post executed action = the
                  halfturn's executed decision index; the exact status of a non-executed decision from the response
                  priority (observed_pass vs unknown, restated here from ron > pon / kan > chi); no action on a pass
  ponds       (e) the four ponds equal an independent replay of the round's XML events at that decision point:
                  discard order per seat, tsumogiri = the tile just drawn, riichi = the REACH step-1 discard,
                  called_by = the seat of the N event that took it, nothing else marked

G1 inference parity (`check_parity`): for every row the training forward produces on a game, `infer.build_sample`
+ `infer.predict` on the same (round, halfturn, seat) gives the same mask (exact), finite legal logits on both paths
and the same masked logits (max |delta| < tol), and the row count per head equals the forward's. The gate runs both
paths in float64 (`double_model`: the same code, the inputs cast) at PARITY_TOL = 1e-8; the float32 production path
is reported as an informational line. Viewer-only rows (`viewer_only_rows`) get the expected mask and finite logits.

    python -m MahjongAI.viewer.viewer_lib --test                      # every corruption raises in the named check
    python -m MahjongAI.viewer.viewer_lib [--ckpt pt] <log> [<log> ...]  # export + check each log (G2)
    python -m MahjongAI.viewer.viewer_lib --parity --ckpt pt <log> ...   # G1 per log, then the totals
"""
import argparse
import copy
import math
import sys

import torch

from MahjongAI.agents.encoder_conv import TransformerTensorProcessor
from MahjongAI.draw import Naki
from MahjongAI.process_xml import _get_rounds, process
from MahjongAI.turn import DiscardTurn, DuringTurn, HalfTurn, PostTurn
from MahjongAI.utils import gates
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_NAKI_IDX, REMAINING_TSUMO
from MahjongAI.viewer import export, infer

DEALT = 52  # 4 x 13
CHECK_ORDER = ("counts", "ownership", "snap", "actual", "ponds")


class ViewerCheckError(AssertionError):
    def __init__(self, where: str, msg: str, check: str):
        super().__init__(f"{where}: [{check}] {msg}")
        self.check = check


def _fail(where: str, msg: str, check: str):
    raise ViewerCheckError(where, msg, check)


# ---- the per-step checks ----
class StepContext:
    """One decision point with everything its checks compare against."""

    def __init__(self, step: dict, snap, turn: HalfTurn, ponds: list, where: str):
        self.step, self.snap, self.turn, self.ponds, self.where = step, snap, turn, ponds, where


def check_counts(ctx: StepContext) -> None:
    """(a): concealed count per seat; the drawn tile is in its hand."""
    step, where = ctx.step, ctx.where
    for p in range(4):
        expected = 13 - 3 * len(step["melds"][p]) + (1 if step["kind"] != "post" and step["actor"] == p else 0)
        if len(step["hands"][p]) != expected:
            _fail(where, f"seat {p} holds {len(step['hands'][p])} tiles, expected {expected} "
                         f"({len(step['melds'][p])} melds, kind {step['kind']}, actor {step['actor']})", "counts")
    if step["drawn"] is not None and step["drawn"]["tile"] not in step["hands"][step["drawn"]["seat"]]:
        _fail(where, f"drawn tile {step['drawn']} is not in its hand", "counts")


def check_ownership(ctx: StepContext) -> None:
    """(b): one owner per tile; tile count = dealt + draws + indicators; draws left."""
    step, where = ctx.step, ctx.where
    seen = []
    for p in range(4):
        seen += step["hands"][p]
        for m in step["melds"][p]:
            seen += m["tiles"]
        seen += [e["tile"] for e in step["ponds"][p] if e["called_by"] is None]
    seen += step["doras"]
    dup = sorted({t for t in seen if seen.count(t) > 1})
    if dup:
        _fail(where, f"tiles with two owners: {dup}", "ownership")
    expected = DEALT + step["draws_so_far"] + len(step["doras"])
    if len(seen) != expected:
        _fail(where, f"{len(seen)} tiles on the table, expected {expected} "
                     f"(52 + {step['draws_so_far']} draws + {len(step['doras'])} indicators)", "ownership")
    if step["draws_left"] != REMAINING_TSUMO - step["draws_so_far"]:
        _fail(where, f"draws_left {step['draws_left']} != 70 - {step['draws_so_far']}", "ownership")


def check_vs_snap(ctx: StepContext) -> None:
    """(c): the display hands, indicators and actor equal the gates replay's."""
    step, snap, where = ctx.step, ctx.snap, ctx.where
    if step["hands"] != snap.ids:
        _fail(where, f"hands {step['hands']} != Snap.ids {snap.ids}", "snap")
    if step["doras"] != snap.doras:
        _fail(where, f"indicators {step['doras']} != Snap.doras {snap.doras}", "snap")
    if step["actor"] != snap.actor:
        _fail(where, f"actor {step['actor']} != Snap.actor {snap.actor}", "snap")


def _priority(d) -> int:
    """ron 3 > pon / daiminkan 2 > chi 1 (the response priority of the rules)."""
    return 1 if d.naki.is_chi() else 2


def expected_actual(turn: HalfTurn, seat: int) -> tuple:
    """(status, executed index or None) of `seat` at `turn`, restated from the XML evidence: a discard point is
    always executed; during / post: executed when a decision of the seat is marked executed; otherwise the seat
    passed observably unless another responder executed an action of HIGHER priority than the lowest-priority
    action this seat was offered (then the seat may have lost the race: unknown)."""
    kind = infer.HEAD_OF[turn.type_]
    if kind == "discard":
        return "executed", export.tile_slot(turn.discarded_tile)
    if kind == "during":
        idx = infer.during_executed(turn)
        return ("executed", idx) if idx else ("observed_pass", None)
    idx = infer.post_executed(turn, seat)
    if idx:
        return "executed", idx
    offered = [3] * len(turn.decisions[seat][DECISION_AGARI_IDX])
    offered += [_priority(d) for d in turn.decisions[seat][DECISION_NAKI_IDX]]
    assert offered, f"seat {seat} has no decision at {turn!r}"
    done = 0
    for q in range(4):
        if any(d.executed for d in turn.decisions[q][DECISION_AGARI_IDX]):
            done = max(done, 3)
        done = max([done] + [_priority(d) for d in turn.decisions[q][DECISION_NAKI_IDX] if d.executed])
    return ("observed_pass" if done <= min(offered) else "unknown"), None


def _check_decision(d: dict, turn: HalfTurn, kind: str, where: str) -> None:
    """(d) for one deciding seat."""
    actual, seat = d["actual"], d["seat"]
    if d["head"] != kind:
        _fail(where, f"seat {seat} head {d['head']} != {kind}", "actual")
    status, idx = expected_actual(turn, seat)
    if actual["status"] != status:
        _fail(where, f"seat {seat} actual status {actual['status']!r} != {status!r} from the response priority", "actual")
    if status == "executed":
        if actual["action"] is None or actual["action"]["idx"] != idx:
            _fail(where, f"seat {seat} actual {actual} != executed {idx}", "actual")
    elif actual["action"] is not None:
        _fail(where, f"seat {seat} actual {actual}: a {status} carries no action", "actual")
    legal = {a["idx"] for a in d["legal"]}
    if actual["action"] is not None and actual["action"]["idx"] not in legal:
        _fail(where, f"seat {seat} executed {actual['action']} outside the legal set {sorted(legal)}", "actual")


def check_actual(ctx: StepContext) -> None:
    """(d): the logged action per deciding seat."""
    step, turn, where = ctx.step, ctx.turn, ctx.where
    kind = infer.HEAD_OF[turn.type_]
    if kind != step["kind"]:
        _fail(where, f"step kind {step['kind']} vs halfturn {turn!r}", "actual")
    seats = [d["seat"] for d in step["decisions"]]
    if seats != infer.deciding_seats(turn):
        _fail(where, f"deciding seats {seats} != {infer.deciding_seats(turn)}", "actual")
    for d in step["decisions"]:
        _check_decision(d, turn, kind, where)


def pond_stream(events: list) -> list:
    """The four ponds at every decision point of one round, replayed from the raw XML events alone: a T/U/V/W
    yields the during point, a D/E/F/G the discard point (before the tile moves) and the post point (after), a kakan
    N the chankan post point. tsumogiri = the tile of the immediately preceding draw by the same seat; riichi = the
    first discard after REACH step 1; called_by = the N caller on the last discard of the seat it came from."""
    ponds = [[] for _ in range(4)]
    pending_draw, riichi_seat, out = None, None, []

    def snap():
        return [[dict(e) for e in pond] for pond in ponds]

    assert events[0]["event"] == "INIT"
    for event in events[1:]:
        tag, attr = event["event"], event["attr"]
        if tag in ("AGARI", "RYUUKYOKU"):
            break
        if tag == "DORA":
            continue
        if tag == "REACH":
            if int(attr["step"]) == 1:
                riichi_seat = int(attr["who"])
            continue
        if tag == "N":
            who, naki = int(attr["who"]), Naki(int(attr["m"]))
            pending_draw = None
            if naki.is_kakan():  # takes no discard (its code keeps the pon's from_who bits); the chankan post point
                out.append(snap())
            elif naki.from_who():  # chi / pon / daiminkan: the last discard of the seat it came from
                ponds[(who + naki.from_who()) % 4][-1]["called_by"] = who
            continue
        if tag[0] in "TUVW":
            pending_draw = ("TUVW".index(tag[0]), int(tag[1:]))
            out.append(snap())
            continue
        assert tag[0] in "DEFG", tag
        seat, tile = "DEFG".index(tag[0]), int(tag[1:])
        out.append(snap())
        ponds[seat].append({"tile": tile, "tsumogiri": pending_draw == (seat, tile), "riichi": riichi_seat == seat,
                            "called_by": None})
        pending_draw = None
        if riichi_seat == seat:
            riichi_seat = None
        out.append(snap())
    return out


def check_ponds(ctx: StepContext) -> None:
    """(e): the ponds equal the XML replay's at this point (order, tsumogiri, riichi, called_by)."""
    step, where = ctx.step, ctx.where
    for p in range(4):
        if step["ponds"][p] != ctx.ponds[p]:
            diff = next((i for i, (a, b) in enumerate(zip(step["ponds"][p], ctx.ponds[p])) if a != b),
                        min(len(step["ponds"][p]), len(ctx.ponds[p])))
            shown = (step["ponds"][p][diff] if diff < len(step["ponds"][p]) else None,
                     ctx.ponds[p][diff] if diff < len(ctx.ponds[p]) else None)
            _fail(where, f"pond of seat {p} differs from the XML replay at discard {diff}: {shown[0]} != {shown[1]} "
                         f"({len(step['ponds'][p])} vs {len(ctx.ponds[p])} discards)", "ponds")


CHECKS = {"counts": check_counts, "ownership": check_ownership, "snap": check_vs_snap, "actual": check_actual,
          "ponds": check_ponds}
assert tuple(CHECKS) == CHECK_ORDER


def step_contexts(game: dict, xr: list, all_halfturns: list, rounds_events: list):
    """Yield a StepContext per decision point (the structural checks: round / step / snap counts)."""
    n_rounds = len(game["rounds"])
    if len(xr) < n_rounds or len(all_halfturns) != n_rounds or len(rounds_events) != n_rounds:
        _fail(game["source"], f"{n_rounds} rounds vs {len(xr)} replayed, {len(all_halfturns)} parsed, "
                              f"{len(rounds_events)} event groups", "structure")
    for ri, (rnd, r, halfturns, events) in enumerate(zip(game["rounds"], xr, all_halfturns, rounds_events)):
        snaps = {k: list(r[k]) for k in ("during", "discard", "post")}
        ponds = pond_stream(events)
        if len(rnd["steps"]) != len(halfturns) or len(ponds) != len(halfturns):
            _fail(f"{game['source']} round {ri}", f"{len(rnd['steps'])} steps vs {len(halfturns)} halfturns, "
                                                  f"{len(ponds)} XML decision points", "structure")
        for si, (step, turn, pond) in enumerate(zip(rnd["steps"], halfturns, ponds)):
            where = f"{game['source']} round {ri} step {si} ({step['kind']} by {step['actor']})"
            if not snaps[step["kind"]]:
                _fail(where, "no gates snap left of this kind", "structure")
            yield StepContext(step, snaps[step["kind"]].pop(0), turn, pond, where)
        left = {k: len(v) for k, v in snaps.items() if v}
        if left:
            _fail(f"{game['source']} round {ri}", f"gates snaps without a step: {left}", "structure")


def check_game(game: dict, xr: list, all_halfturns: list, rounds_events: list, only: str = None) -> int:
    """Every check (or `only` one, by name) on every step; returns the number of steps checked."""
    n = 0
    for ctx in step_contexts(game, xr, all_halfturns, rounds_events):
        for name in ((only,) if only else CHECK_ORDER):
            CHECKS[name](ctx)
        n += 1
    return n


def check_log(path: str, model=None, game: dict = None) -> dict:
    """Export `path` (unless `game` is given) and run every check; returns the game. A self-play game.log is parsed
    with the admission filters off, as the exporter parses it."""
    if game is None:
        game = export.export_game(path, model)
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", "replace")
    xr = gates.xml_rounds(gates.kept_xml(raw)[0])
    all_halfturns, _ = process(path, admission=export.selfplay_meta(path) is None)
    _, rounds_events = _get_rounds(path)
    n = check_game(game, xr, all_halfturns, rounds_events)
    print(f"G2 ok: {game['source']}: {len(game['rounds'])} rounds, {n} steps, complete={game['complete']}")
    return game


# ---- G1: inference parity ----
_processor = TransformerTensorProcessor()
PARITY_TOL = 1e-8  # float64, both paths: exact-input parity


def double_model(model):
    """A float64 copy of `model` whose state_net casts its (float32-built) inputs to float64: the training forward
    and the viewer path then run the same code in float64, with nothing else changed."""
    m = copy.deepcopy(model).double()
    m.state_net.register_forward_pre_hook(lambda mod, args: (tuple(t.double() for t in args[0]),))
    return m.eval()


def training_rows(all_halfturns: list) -> dict:
    """{head: [(round, halfturn index, seat)]} in the order the training builders emit them (their filters restated:
    a forced riichi discard, a during point with no option, a post point with no option, a pre-empted responder)."""
    rows = {"during": [], "discard": [], "post": []}
    for ri, halfturns in enumerate(all_halfturns):
        for hi, t in enumerate(halfturns):
            if isinstance(t, DuringTurn):
                if sum(map(len, t.decisions)):
                    rows["during"].append((ri, hi, t.player))
            elif isinstance(t, DiscardTurn):
                if not infer.forced_riichi_discard(t):
                    rows["discard"].append((ri, hi, t.player))
            elif isinstance(t, PostTurn):
                if sum(len(x) for seat in t.decisions for x in seat):
                    rows["post"] += [(ri, hi, p) for p in _processor._post_turn_responders(t)]
    return rows


def viewer_only_rows(all_halfturns: list) -> dict:
    """{kind: [(round, halfturn index, seat)]} of the decision points the viewer shows and the training drops."""
    out = {"forced_riichi_discard": [], "no_option_during": [], "preempted_responder": []}
    for ri, halfturns in enumerate(all_halfturns):
        for hi, t in enumerate(halfturns):
            if isinstance(t, DiscardTurn) and infer.forced_riichi_discard(t):
                out["forced_riichi_discard"].append((ri, hi, t.player))
            elif isinstance(t, DuringTurn) and not sum(map(len, t.decisions)):
                out["no_option_during"].append((ri, hi, t.player))
            elif isinstance(t, PostTurn):
                kept = set(_processor._post_turn_responders(t))
                out["preempted_responder"] += [(ri, hi, p) for p in infer.deciding_seats(t) if p not in kept]
    return out


def check_parity(model, all_halfturns: list, all_tokens: list, tol: float = PARITY_TOL) -> dict:
    """{head: {"n": rows, "max_abs_delta": x}} of one game; raises ViewerCheckError on a mask, row-count, non-finite
    logit or logit mismatch between the training forward and the viewer's per-sample path (both on `model`)."""
    rows = training_rows(all_halfturns)
    batches = dict(zip(("during", "discard", "post"), _processor.prepare_batches([all_halfturns], [all_tokens])))
    model.eval()
    with torch.no_grad():
        results = model([all_halfturns], [all_tokens], sample=False)
    out = {}
    for head, keys in rows.items():
        masks, logits = batches[head][2], results[head].logits.cpu() if head in results else None
        if len(keys) != masks.shape[0] or (logits is not None and logits.shape[0] != len(keys)):
            _fail(head, f"{len(keys)} restated rows vs {masks.shape[0]} built, "
                        f"{None if logits is None else logits.shape[0]} forward", "parity")
        worst = 0.0
        for k, (ri, hi, seat) in enumerate(keys):
            sample = infer.build_sample(all_halfturns[ri][hi], all_tokens[ri], seat)
            if sample is None or not torch.equal(sample.mask[0], masks[k]):
                _fail(f"{head} row {k} {(ri, hi, seat)}", "mask differs from the training mask", "parity")
            _, viewer_logits, _ = infer.predict(model, sample)
            legal = masks[k] > 0
            v, r = viewer_logits[legal], logits[k][legal]
            if not torch.isfinite(v).all() or not torch.isfinite(r).all():
                _fail(f"{head} row {k} {(ri, hi, seat)}",
                      f"non-finite legal logits: viewer {v.tolist()}, training {r.tolist()}", "parity")
            delta = (v.double() - r.double()).abs().max().item()
            worst = max(worst, delta)
            if not math.isfinite(delta) or delta >= tol:
                _fail(f"{head} row {k} {(ri, hi, seat)}", f"max |delta logits| {delta:.3e} >= {tol:.0e}", "parity")
        out[head] = {"n": len(keys), "max_abs_delta": worst}
    return out


def check_viewer_only(model, all_halfturns: list, all_tokens: list) -> dict:
    """{kind: n}: every viewer-only row has the expected mask and finite logits."""
    counts = {}
    for kind, keys in viewer_only_rows(all_halfturns).items():
        for ri, hi, seat in keys:
            t = all_halfturns[ri][hi]
            sample = infer.build_sample(t, all_tokens[ri], seat)
            if sample is None:
                _fail(f"{kind} {(ri, hi, seat)}", "no sample", "viewer_only")
            mask = sample.mask[0]
            if kind == "forced_riichi_discard":
                expected = torch.zeros_like(mask)
                expected[t.stateObj.drawn_tile[-1]] = 1.0
                if not torch.equal(mask, expected) or export.tile_slot(t.discarded_tile) != t.stateObj.drawn_tile[-1]:
                    _fail(f"{kind} {(ri, hi, seat)}", f"mask {mask.nonzero().flatten().tolist()} is not the drawn tile",
                          "viewer_only")
            elif kind == "no_option_during":
                if mask.nonzero().flatten().tolist() != [0]:
                    _fail(f"{kind} {(ri, hi, seat)}", f"mask {mask.nonzero().flatten().tolist()} is not pass only",
                          "viewer_only")
            else:
                offered = {1} if t.decisions[seat][0] else set()
                offered |= {d.naki.get_post_turn_filter_idx() for d in t.decisions[seat][2]}
                if set(mask.nonzero().flatten().tolist()) != offered | {0}:
                    _fail(f"{kind} {(ri, hi, seat)}", f"mask {mask.nonzero().flatten().tolist()} != pass + {offered}",
                          "viewer_only")
            _, logits, probs = infer.predict(model, sample)
            if not torch.isfinite(logits[mask > 0]).all() or not torch.isfinite(probs).all():
                _fail(f"{kind} {(ri, hi, seat)}", "non-finite logits", "viewer_only")
        counts[kind] = len(keys)
    return counts


def parity_report(model, paths: list, tol: float = PARITY_TOL) -> dict:
    """G1 over several logs: per head the total rows and the worst |delta| in float64 (the gate, `tol`) and in the
    float32 production path (informational, never fails on the delta), plus the viewer-only counts."""
    m64 = double_model(model)
    total = {"float64": {h: {"n": 0, "max_abs_delta": 0.0} for h in ("during", "discard", "post")},
             "float32": {h: {"n": 0, "max_abs_delta": 0.0} for h in ("during", "discard", "post")}}
    extra = {}
    for path in paths:
        all_halfturns, all_tokens = process(path)
        per = {"float64": check_parity(m64, all_halfturns, all_tokens, tol),
               "float32": check_parity(model, all_halfturns, all_tokens, math.inf)}
        for dtype, heads in per.items():
            for h, r in heads.items():
                total[dtype][h]["n"] += r["n"]
                total[dtype][h]["max_abs_delta"] = max(total[dtype][h]["max_abs_delta"], r["max_abs_delta"])
        for k, n in check_viewer_only(model, all_halfturns, all_tokens).items():
            extra[k] = extra.get(k, 0) + n
        name = path.split("/")[-1]
        for dtype, heads in per.items():
            print(f"G1 {dtype} {name}: "
                  + ", ".join(f"{h} n={r['n']} max|d|={r['max_abs_delta']:.2e}" for h, r in heads.items()))
    return {"heads": total["float64"], "float32": total["float32"], "viewer_only": extra, "games": len(paths)}


# ---- self-test ----
def _first_post(g):
    return next(s for s in g["rounds"][0]["steps"] if s["kind"] == "post")


def _first_discard(g):
    return next(s for s in g["rounds"][0]["steps"] if s["kind"] == "discard")


def _first_meld_step(g):
    return next(s for r in g["rounds"] for s in r["steps"] if any(s["melds"]))


def _step_with(g, pred):
    return next(s for r in g["rounds"] for s in r["steps"] if pred(s))


def _decision_with(g, pred):
    return next(d for r in g["rounds"] for s in r["steps"] for d in s["decisions"] if pred(s, d))


def _c_drop_tile(g):  # counts
    _first_post(g)["hands"][0].pop()


def _c_extra_tile(g):  # counts: a 14th tile at a post point
    s = _first_post(g)
    s["hands"][1].append(s["hands"][1][0])


def _c_drawn_not_in_hand(g):  # counts: the drawn tile names a tile the hand does not hold
    s = _step_with(g, lambda s: s["drawn"] is not None)
    s["drawn"] = dict(s["drawn"], tile=next(t for t in range(136) if t not in s["hands"][s["drawn"]["seat"]]))


def _c_duplicate_owner(g):  # ownership: a pond tile also in a hand (counts stay right by dropping one)
    s = _first_post(g)
    s["hands"][s["actor"]][0] = s["ponds"][s["actor"]][-1]["tile"]


def _c_called_tile_in_pond(g):  # ownership: a called discard counted in the pond again
    s = _first_meld_step(g)
    e = next(e for pond in s["ponds"] for e in pond if e["called_by"] is not None)
    e["called_by"] = None


def _c_draws_left_off(g):  # ownership
    _first_post(g)["draws_left"] += 1


def _c_hidden_dora_shown(g):  # ownership: an extra indicator (a tile with two owners)
    s = _first_post(g)
    s["doras"].append(s["hands"][0][0])


def _c_swap_hands(g):  # snap: same counts, other tiles
    s = _first_post(g)
    s["hands"][0], s["hands"][1] = s["hands"][1], s["hands"][0]


def _c_wrong_actor(g):  # snap
    s = _first_post(g)
    s["actor"] = (s["actor"] + 1) % 4


def _c_wrong_discard(g):  # actual
    a = _first_discard(g)["decisions"][0]["actual"]["action"]
    a["idx"] = (a["idx"] + 1) % 37


def _c_pass_with_action(g):  # actual
    s = _step_with(g, lambda s: s["kind"] == "during" and s["decisions"][0]["actual"]["status"] == "observed_pass")
    s["decisions"][0]["actual"]["action"] = {"idx": 0, "name": "pass"}


def _c_executed_as_pass(g):  # actual
    d = _decision_with(g, lambda s, d: s["kind"] == "post" and d["actual"]["status"] == "executed")
    d["actual"] = {"status": "observed_pass", "action": None}


def _c_pass_as_unknown(g):  # actual: an observed pass of a post responder relabelled unknown
    d = _decision_with(g, lambda s, d: s["kind"] == "post" and d["actual"]["status"] == "observed_pass")
    d["actual"]["status"] = "unknown"


def _c_unknown_as_pass(g):  # actual: a pre-empted responder relabelled an observed pass
    d = _decision_with(g, lambda s, d: d["actual"]["status"] == "unknown")
    d["actual"]["status"] = "observed_pass"


def _c_missing_seat(g):  # actual
    s = _step_with(g, lambda s: len(s["decisions"]) > 1)
    s["decisions"].pop()


def _c_pond_order(g):  # ponds: two discards of one seat swapped (same tiles, same owners)
    s = _step_with(g, lambda s: any(len(p) >= 2 for p in s["ponds"]))
    pond = next(p for p in s["ponds"] if len(p) >= 2)
    pond[0], pond[1] = pond[1], pond[0]


def _c_riichi_flag(g):  # ponds: a riichi declaration discard unmarked
    s = _step_with(g, lambda s: any(e["riichi"] for p in s["ponds"] for e in p))
    next(e for p in s["ponds"] for e in p if e["riichi"])["riichi"] = False


def _c_tsumogiri_flag(g):  # ponds: a hand discard marked tsumogiri
    s = _step_with(g, lambda s: any(not e["tsumogiri"] for p in s["ponds"] for e in p))
    next(e for p in s["ponds"] for e in p if not e["tsumogiri"])["tsumogiri"] = True


def _c_called_seat(g):  # ponds: a called discard credited to another seat (still marked called: ownership holds)
    s = _first_meld_step(g)
    e = next(e for pond in s["ponds"] for e in pond if e["called_by"] is not None)
    e["called_by"] = (e["called_by"] + 1) % 4


def _corruptions() -> dict:
    """name -> (edit(game), the check that must catch it); edits on sample_haifu's export."""
    table = (
        (_c_drop_tile, "counts"), (_c_extra_tile, "counts"), (_c_drawn_not_in_hand, "counts"),
        (_c_duplicate_owner, "ownership"), (_c_called_tile_in_pond, "ownership"), (_c_draws_left_off, "ownership"),
        (_c_hidden_dora_shown, "ownership"),
        (_c_swap_hands, "snap"), (_c_wrong_actor, "snap"),
        (_c_wrong_discard, "actual"), (_c_pass_with_action, "actual"), (_c_executed_as_pass, "actual"),
        (_c_pass_as_unknown, "actual"), (_c_unknown_as_pass, "actual"), (_c_missing_seat, "actual"),
        (_c_pond_order, "ponds"), (_c_riichi_flag, "ponds"), (_c_tsumogiri_flag, "ponds"), (_c_called_seat, "ponds"),
    )
    return {f.__name__[3:]: (f, check) for f, check in table}


def self_test() -> None:
    """The clean export passes; each corruption raises in the check it names, both when that check runs alone and
    as the first failure of the full pipeline; every check catches at least one corruption."""
    path = gates.SAMPLE_LOG
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    xr = gates.xml_rounds(raw)
    all_halfturns, _ = process(path)
    _, rounds_events = _get_rounds(path)
    clean = export.export_game(path, None)
    n = check_game(clean, xr, all_halfturns, rounds_events)
    assert n == sum(len(h) for h in all_halfturns), n
    print(f"clean: {n} steps pass")
    covered = set()
    for name, (corrupt, check) in _corruptions().items():
        game = copy.deepcopy(clean)
        corrupt(game)
        for only in (check, None):
            try:
                check_game(game, xr, all_halfturns, rounds_events, only=only)
            except ViewerCheckError as e:
                if e.check != check:
                    raise AssertionError(f"corruption {name}: check {e.check} raised, expected {check}: {e}") from e
                if only is None:
                    print(f"  {name}: caught by {check} ({str(e)[:90]})")
            else:
                raise AssertionError(f"corruption {name} passed {'check ' + only if only else 'every check'}")
        covered.add(check)
    missing = set(CHECK_ORDER) - covered
    assert not missing, f"no corruption exercises {sorted(missing)}"
    print(f"self-test ok: {len(_corruptions())} corruptions, each caught by its own check; checks {CHECK_ORDER}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--ckpt")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--parity", action="store_true", help="G1 instead of G2 (needs --ckpt)")
    ap.add_argument("--tol", type=float, default=PARITY_TOL, help="G1 float64 logit tolerance")
    args = ap.parse_args(argv)
    if args.test:
        self_test()
        return 0
    model = infer.load_model(args.ckpt) if args.ckpt else None
    if args.parity:
        assert model is not None, "--parity needs --ckpt"
        report = parity_report(model, args.logs, args.tol)
        worst = max(r["max_abs_delta"] for r in report["heads"].values())
        worst32 = max(r["max_abs_delta"] for r in report["float32"].values())
        print(f"G1 float64 totals over {report['games']} games: " + ", ".join(
            f"{h} rows={r['n']} max|delta|={r['max_abs_delta']:.3e}" for h, r in report["heads"].items()))
        print("G1 float32 (production path, informational): " + ", ".join(
            f"{h} rows={r['n']} max|delta|={r['max_abs_delta']:.3e}" for h, r in report["float32"].items())
            + f"; max {worst32:.3e}")
        print(f"viewer-only rows: {report['viewer_only']}")
        print(f"G1 {'PASS' if worst < args.tol else 'FAIL'}: float64 max |delta logits| {worst:.3e} vs {args.tol:.0e}"
              f" (masks exact, row counts equal on every game)")
        return 0 if worst < args.tol else 1
    for path in args.logs:
        check_log(path, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
