"""Tenhou log + checkpoint -> trajectory JSON (T-052, plan section A).

One pass over the XML events of the rounds `process()` keeps (the same BYE cut and the same rejections) builds the
DISPLAY state: all four hands as 136-ids, ponds in discard order, melds, scores, sticks, indicators. The parser's
halfturns give the MODEL state; the two streams are zipped decision point by decision point (the snapshot boundary
table of the plan) and every deciding seat gets the model's masked distribution next to what happened.

A self-play game (T-053: `game.log` beside the engine's `meta.json` + `decisions.json`) takes the sidecar instead:
every decision shows the distribution of the seat's OWN player at play time (a model's softmax(logits / T), greedy's
one-hot, uniform's flat), its choice, whether it was sampled and what the round did with it; the names are the
`meta.json` seat labels. While a game plays, `export_live_round` exports the engine's own events with the same
`export_round`, up to the last decision point whose outcome is final.

    python -m MahjongAI.viewer.export <log> --ckpt <pt> --out game.json
"""
import argparse
import json
import os
import urllib.parse
from typing import Optional

from MahjongAI.agents.encoder_conv import TransformerTensorProcessor
from MahjongAI.draw import RED_CHI_VARIANT, Naki
from MahjongAI.engine.game import GO_TYPE
from MahjongAI.gameinfo import GameInfo
from MahjongAI.process_xml import InvalidGameException, _get_rounds, process, process_events
from MahjongAI.turn import DiscardTurn, DuringTurn, HalfTurn, PostTurn
from MahjongAI.utils.constants import REMAINING_TSUMO, TILE2IDX
from MahjongAI.viewer import infer

SUITS = "mps"
HONORS = ["東", "南", "西", "北", "白", "發", "中"]
# the 37-slot names: 0-26 suits, 27-33 honors, 34-36 red fives
TILE_NAMES = [f"{n + 1}{s}" for s in SUITS for n in range(9)] + HONORS + [f"5{s}r" for s in SUITS]
WINDS = HONORS[:4]
LIMIT_NAMES = {0: "", 1: "満貫", 2: "跳満", 3: "倍満", 4: "三倍満", 5: "役満"}
# Tenhou's <AGARI yaku="id,han,..."> ids
YAKU_NAMES = [
    "門前清自摸和", "立直", "一発", "槍槓", "嶺上開花", "海底摸月", "河底撈魚", "平和", "断幺九", "一盃口",
    "自風 東", "自風 南", "自風 西", "自風 北", "場風 東", "場風 南", "場風 西", "場風 北",
    "役牌 白", "役牌 發", "役牌 中", "両立直", "七対子", "混全帯幺九", "一気通貫", "三色同順", "三色同刻",
    "三槓子", "対々和", "三暗刻", "小三元", "混老頭", "二盃口", "純全帯幺九", "混一色", "清一色", "人和",
    "天和", "地和", "大三元", "四暗刻", "四暗刻単騎", "字一色", "緑一色", "清老頭", "九蓮宝燈", "純正九蓮宝燈",
    "国士無双", "国士無双１３面", "大四喜", "小四喜", "四槓子", "ドラ", "裏ドラ", "赤ドラ",
]
RED_CHI_INVERSE = {v: k for k, v in RED_CHI_VARIANT.items()}
_processor = TransformerTensorProcessor()


def tile_slot(tile: int) -> int:
    """The 37-slot index of a 136-id (the red slot for a red five)."""
    return TILE2IDX[tile][-1]


def tile_name(tile: int) -> str:
    return TILE_NAMES[tile_slot(tile)]


def _run_name(color: int, number: int) -> str:
    return f"{number + 1}-{number + 3}{SUITS[color]}"


def during_action_name(idx: int) -> str:
    if idx == 0:
        return "pass"
    if idx == 1:
        return "tsumo"
    if idx == 2:
        return "riichi"
    if 3 <= idx <= 36:
        return f"ankan {TILE_NAMES[idx - 3]}"
    if 37 <= idx <= 70:
        return f"kakan {TILE_NAMES[idx - 37]}"
    assert idx == 71, idx
    return "kyuushu"


def post_action_name(idx: int) -> str:
    """The readable name of a post-head index (the layout of draw.Naki.get_post_turn_filter_idx)."""
    if idx == 0:
        return "pass"
    if idx == 1:
        return "ron"
    if 2 <= idx <= 64:
        v = idx - 2
        color, number, which = v // 21, (v % 21) // 3, v % 3
        return f"chi {_run_name(color, number)} ({number + which + 1}{SUITS[color]})"
    if 65 <= idx <= 82:
        v = idx - 65
        color, (number, which) = v // 6, RED_CHI_INVERSE[v % 6]
        return f"chi {_run_name(color, number)} red ({number + which + 1}{SUITS[color]})"
    if 83 <= idx <= 119:
        v = idx - 83
        if v < 27:
            return f"pon {TILE_NAMES[v]}"
        if v < 30:
            return f"pon 5{SUITS[v - 27]} red"
        return f"pon {HONORS[v - 30]}"
    assert 120 <= idx <= 153, idx
    return f"kan {TILE_NAMES[idx - 120]}"


ACTION_NAME = {"during": during_action_name, "discard": lambda i: TILE_NAMES[i], "post": post_action_name}


def action(head: str, idx: int) -> dict:
    return {"idx": int(idx), "name": ACTION_NAME[head](int(idx))}


def kakan_parts(naki: Naki) -> tuple:
    """(the pon's three 136-ids, its called tile, the added tile) of a kakan LOG code. Tenhou keeps the pon's bits
    (`Naki.kakan_of`): the unused-copy bits now name the added copy, the called position stays the pon's."""
    assert naki.is_kakan() and not naki.offered, naki
    color, number, which, _, exposed, _ = naki.pattern_kakan()
    added = exposed[0]
    pon = [t for t in range((9 * color + number) * 4, (9 * color + number) * 4 + 4) if t != added]
    called = pon.pop(which)  # the same pop as pattern_pon on the same three copies
    return pon, called, added


def _meld_dict(naki: Naki, who: int) -> dict:
    """The display record of a log meld code: kind, ALL its 136-ids sorted (four for a kakan), the called tile, the
    seat it came from, and a kakan's added tile."""
    from_seat = (who + naki.from_who()) % 4 if naki.from_who() else None
    if naki.is_kakan():
        pon, called, added = kakan_parts(naki)
        return {"kind": "kakan", "tiles": sorted(pon + [called, added]), "called": called, "from": from_seat,
                "added": added}
    exposed, acquired = naki.get_exposed()
    if naki.is_chi():
        kind = "chi"
    elif naki.is_pon():
        kind = "pon"
    elif naki.is_minkan():
        kind = "minkan"
    else:
        kind = "ankan"
    tiles = sorted(exposed + ([acquired] if acquired is not None else []))
    return {"kind": kind, "tiles": tiles, "called": acquired, "from": from_seat, "added": None}


class TableState:
    """The display state of one round: what every seat can see, plus the four concealed hands (replay: all open).

    Owns the invariants the parser also keeps: a tile is in exactly one of hand / meld / pond (a called discard
    stays in the pond, marked), the open-kan indicator deferral (T-035), the riichi declaration discard."""

    def __init__(self, init_attr: dict):
        seed = [int(x) for x in init_attr["seed"].split(",")]
        self.round_index, self.honba, self.kyotaku = seed[0], seed[1], seed[2]
        self.oya = int(init_attr["oya"])
        self.scores = [int(x) for x in init_attr["ten"].split(",")]
        self.hands = [sorted(int(t) for t in init_attr[f"hai{p}"].split(",")) for p in range(4)]
        self.ponds = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.reaches = [0] * 4
        self.doras = [seed[5]]
        self.hidden_dora = None
        self.draws_so_far = 0
        self.drawn = None  # (seat, tile) of the latest draw; None after a call or a discard
        self.riichi_declaring = None  # the seat whose next discard is its riichi declaration
        self.open_kan_owner = None  # a daiminkan / kakan seat until its rinshan draw
        self.open_kan_drawer = None  # that seat from its rinshan draw until its discard

    # ---- events ----
    def draw(self, seat: int, tile: int) -> None:
        self.open_kan_drawer, self.open_kan_owner = self.open_kan_owner, None
        self.hands[seat].append(tile)
        self.hands[seat].sort()
        self.draws_so_far += 1
        self.drawn = (seat, tile)

    def dora(self, tile: int) -> None:
        if self.open_kan_drawer is not None:  # `N T DORA D`: flipped only after the discard
            assert self.hidden_dora is None, f"two hidden kan indicators: {self.hidden_dora}, {tile}"
            self.hidden_dora = tile
        else:
            self.doras.append(tile)

    def reach(self, seat: int, step: int) -> None:
        if step == 1:
            self.reaches[seat] = 1
            self.riichi_declaring = seat
        else:
            self.kyotaku += 1
            self.scores[seat] -= 10

    def discard(self, seat: int, tile: int) -> None:
        """Move `tile` from the hand to the pond; then flip a deferred open-kan indicator."""
        self.hands[seat].remove(tile)
        tsumogiri = self.drawn is not None and self.drawn == (seat, tile)
        self.ponds[seat].append({"tile": tile, "tsumogiri": tsumogiri, "riichi": self.riichi_declaring == seat,
                                 "called_by": None})
        self.riichi_declaring = None
        self.drawn = None
        if self.hidden_dora is not None:
            self.doras.append(self.hidden_dora)
            self.hidden_dora = None
        self.open_kan_drawer = None

    def call(self, seat: int, naki: Naki) -> None:
        """Apply a log meld: the exposed tiles leave the hand; a called discard is marked; a kakan upgrades its pon."""
        self.open_kan_drawer = None
        self.open_kan_owner = seat if naki.is_minkan() or naki.is_kakan() else None
        self.drawn = None
        exposed, acquired = naki.get_exposed()
        for t in exposed:
            self.hands[seat].remove(t)
        meld = _meld_dict(naki, seat)
        if naki.is_kakan():
            kind = exposed[0] // 4
            i = next(i for i, m in enumerate(self.melds[seat]) if m["kind"] == "pon" and m["tiles"][0] // 4 == kind)
            pon = self.melds[seat][i]
            assert sorted(pon["tiles"] + exposed) == meld["tiles"] and (pon["called"], pon["from"]) == (
                meld["called"], meld["from"]), f"kakan {meld} does not extend the pon {pon}"
            self.melds[seat][i] = meld
            return
        if acquired is not None:
            last = self.ponds[meld["from"]][-1]
            assert last["tile"] == acquired and last["called_by"] is None, (
                f"{meld} by {seat} does not call the last discard of {meld['from']}: {last}")
            last["called_by"] = seat
        self.melds[seat].append(meld)

    # ---- snapshot ----
    def snapshot(self, kind: str, actor: int) -> dict:
        drawn = self.drawn if kind != "post" and self.drawn is not None and self.drawn[0] == actor else None
        return {
            "kind": kind,
            "actor": actor,
            "hands": [list(h) for h in self.hands],
            "drawn": {"seat": drawn[0], "tile": drawn[1]} if drawn else None,
            "ponds": [[dict(e) for e in pond] for pond in self.ponds],
            "melds": [[dict(m, tiles=list(m["tiles"])) for m in seat] for seat in self.melds],
            "scores": [s * 100 for s in self.scores],
            "reaches": list(self.reaches),
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "doras": list(self.doras),
            "draws_left": REMAINING_TSUMO - self.draws_so_far,
            "draws_so_far": self.draws_so_far,
            "round": self.round_index,
            "oya": self.oya,
        }


def _actual(turn: HalfTurn, seat: int) -> dict:
    """What the log shows `seat` did at `turn`: executed (with the action), observed_pass, or unknown [C6]."""
    head = infer.HEAD_OF[turn.type_]
    if head == "discard":
        return {"status": "executed", "action": action(head, tile_slot(turn.discarded_tile))}
    if head == "during":
        idx = infer.during_executed(turn)
        if idx:
            return {"status": "executed", "action": action(head, idx)}
        return {"status": "observed_pass", "action": None}
    idx = infer.post_executed(turn, seat)
    if idx:
        return {"status": "executed", "action": action(head, idx)}
    if seat in _processor._post_turn_responders(turn):
        return {"status": "observed_pass", "action": None}
    return {"status": "unknown", "action": None}


class Unsettled(Exception):
    """A decision point whose outcome is not final yet (live mode) or a self-play decision without its record."""


def _played(record: dict, head: str, legal: list) -> dict:
    """The decision as the seat's own player made it, from its sidecar record [C11]: the play-time distribution
    (`candidates`), the top action when it is unique, the choice, sampled or not, and what the round did with it."""
    assert record["kind"] == head and record["offer"] == legal, (
        f"sidecar record k={record['k']} ({record['kind']}, offer {record['offer']}) vs the parse ({head}, {legal})")
    probs = {k: float(p) for k, p in record["candidates"].items()}
    assert sorted(int(k) for k in probs) == legal, f"sidecar record k={record['k']}: candidates {sorted(probs)}"
    best = max(probs.values())
    tops = [int(k) for k, p in probs.items() if p == best]
    return {"probs": probs, "model_top": action(head, tops[0]) if len(tops) == 1 else None,
            "player": {"label": record["player"], "chosen": action(head, record["chosen"]),
                       "sampled": bool(record["sampled"]), "resolved": record["resolved"]["outcome"]}}


def _decision(model, turn: HalfTurn, tokens: list, seat: int, records: Optional[dict] = None,
              index: Optional[int] = None) -> dict:
    """records: the round's sidecar ((halfturn index, seat) -> record) of a self-play game, `index` the halfturn's
    index: the decision then shows the seat's own player, never `model`; only a forced move may lack a record."""
    sample = infer.build_sample(turn, tokens, seat)
    assert sample is not None, f"seat {seat} has no decision at {turn!r}"
    legal = [int(i) for i in sample.mask[0].nonzero().flatten().tolist()]
    out = {"seat": seat, "head": sample.head, "legal": [action(sample.head, i) for i in legal],
           "logits": {}, "probs": {}, "model_top": None, "actual": _actual(turn, seat)}
    if records is not None:
        record = records.get((index, seat))
        if record is not None and record["resolved"] is not None:
            out.update(_played(record, sample.head, legal))
            return out
        if record is not None or len(legal) > 1:
            raise Unsettled(f"seat {seat} at halfturn {index}: sidecar record {'unresolved' if record else 'missing'}")
    if len(legal) == 1:  # a forced move: no model call
        out["logits"], out["probs"] = {str(legal[0]): 0.0}, {str(legal[0]): 1.0}
        out["model_top"] = action(sample.head, legal[0])
    elif model is not None:
        _, logits, probs = infer.predict(model, sample)
        out["logits"] = {str(i): float(logits[i]) for i in legal}
        out["probs"] = {str(i): float(probs[i]) for i in legal}
        out["model_top"] = action(sample.head, int(probs.argmax()))
    return out


def _result(event: dict) -> dict:
    """The round summary of an AGARI / RYUUKYOKU event (XML attributes only)."""
    attr = event["attr"]
    sc = [int(x) for x in attr["sc"].split(",")]
    out = {"type": event["event"].lower(), "deltas": [sc[2 * i + 1] * 100 for i in range(4)],
           "scores_after": [(sc[2 * i] + sc[2 * i + 1]) * 100 for i in range(4)]}
    if event["event"] == "AGARI":
        fu, ten, limit = (int(x) for x in attr["ten"].split(","))
        yaku = [int(x) for x in attr.get("yaku", "").split(",") if x != ""]
        out.update({
            "who": int(attr["who"]), "from": int(attr["fromWho"]), "fu": fu, "ten": ten,
            "limit": LIMIT_NAMES.get(limit, ""),
            "yaku": [[YAKU_NAMES[yaku[i]], yaku[i + 1]] for i in range(0, len(yaku), 2)],
            "yakuman": [YAKU_NAMES[int(x)] for x in attr.get("yakuman", "").split(",") if x != ""],
            "hand": [int(x) for x in attr["hai"].split(",")],
            "melds": [_meld_dict(Naki(int(m)), int(attr["who"])) for m in attr.get("m", "").split(",") if m != ""],
            "machi": int(attr["machi"]),
            "doras": [int(x) for x in attr["doraHai"].split(",")],
            "ura": [int(x) for x in attr.get("doraHaiUra", "").split(",") if x != ""],
        })
    else:
        # hai<N> is a revealed hand: the tenpai hands of an exhaustive draw (and of a nagashi mangan), but the
        # kyuushu hand of a yao9 abort; the other aborts reveal nothing. `tenpai` is None where the XML cannot say.
        reason = attr.get("type", "exhaustive")
        revealed = [p for p in range(4) if f"hai{p}" in attr]
        out["reason"], out["revealed"] = reason, revealed
        out["tenpai"] = revealed if reason in ("exhaustive", "nm") else None
    if "owari" in attr:
        owari = [int(float(x)) for x in attr["owari"].split(",")]
        out["final_scores"] = [owari[2 * i] * 100 for i in range(4)]
    return out


def _player_names(path: str) -> list:
    """The four names of the first <UN> tag (URL-encoded in the log)."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", "replace")
    i = raw.find("<UN ")
    if i < 0:
        return [f"seat {p}" for p in range(4)]
    tag = raw[i:raw.index(">", i)]
    names = []
    for p in range(4):
        j = tag.find(f'n{p}="')
        names.append(urllib.parse.unquote(tag[j + 4:tag.index('"', j + 4)]) if j >= 0 else f"seat {p}")
    return names


def _next_turn(halfturns: list, hi: int, cls, seat: int) -> HalfTurn:
    assert hi < len(halfturns), f"the parser has no halfturn left for a {cls.__name__} by {seat}"
    turn = halfturns[hi]
    assert isinstance(turn, cls) and turn.player == seat, f"expected {cls.__name__} by {seat}, got {turn!r}"
    return turn


def export_round(events: list, halfturns: list, tokens: list, model, records: Optional[dict] = None,
                 settled: Optional[int] = None) -> dict:
    """One round: the display replay of `events` zipped with the parser's `halfturns`; both streams are consumed.
    records: the round's self-play sidecar (`sidecar_by_round`). settled (live mode, an unfinished round): only the
    decision points of the first `settled` halfturns whose records are resolved; the export stops at the first
    other one, so it is a prefix of the round's final export."""
    assert events[0]["event"] == "INIT"
    table = TableState(events[0]["attr"])
    steps, results, hi = [], [], 0

    def step(kind, actor, turn):
        index = len(steps)  # one step per halfturn, in order
        if settled is not None and index >= settled:
            raise Unsettled(f"halfturn {index} is the last one built")
        snap = table.snapshot(kind, actor)
        snap["decisions"] = [_decision(model, turn, tokens, p, records, index) for p in infer.deciding_seats(turn)]
        steps.append(snap)

    def done(round_results):
        return {"index": table.round_index, "honba": events[0]["attr"]["seed"].split(",")[1],
                "oya": table.oya, "steps": steps, "results": round_results}

    try:
        for event in events[1:]:
            tag, attr = event["event"], event["attr"]
            if tag in ("AGARI", "RYUUKYOKU"):
                results.append(_result(event))
                continue
            assert not results, f"event {tag} after the round's result"
            if tag == "DORA":
                table.dora(int(attr["hai"]))
            elif tag == "REACH":
                table.reach(int(attr["who"]), int(attr["step"]))
            elif tag == "N":
                seat, naki = int(attr["who"]), Naki(int(attr["m"]))
                table.call(seat, naki)
                if naki.is_kakan():
                    step("post", seat, _next_turn(halfturns, hi, PostTurn, seat))
                    hi += 1
            elif tag[0] in "TUVW":
                seat, tile = "TUVW".index(tag[0]), int(tag[1:])
                table.draw(seat, tile)
                step("during", seat, _next_turn(halfturns, hi, DuringTurn, seat))
                hi += 1
            elif tag[0] in "DEFG":
                seat, tile = "DEFG".index(tag[0]), int(tag[1:])
                turn = _next_turn(halfturns, hi, DiscardTurn, seat)
                assert turn.discarded_tile == tile, f"discard {tile} by {seat} vs {turn!r}"
                step("discard", seat, turn)
                hi += 1
                table.discard(seat, tile)
                step("post", seat, _next_turn(halfturns, hi, PostTurn, seat))
                hi += 1
            else:
                raise ValueError(f"unknown event {tag}")
    except Unsettled:
        if settled is None:
            raise
        return done([])
    if settled is not None:
        return done([])
    assert hi == len(halfturns), f"{len(halfturns) - hi} parser halfturns not matched by the display replay"
    return done(results)


def selfplay_meta(path: str) -> Optional[dict]:
    """meta.json when `path` is a self-play game.log (engine/game.py GameRecord.save: meta.json with `generated` and
    decisions.json beside it), else None."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")
    if os.path.basename(path) != "game.log" or not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta if meta.get("generated") else None


def selfplay_sidecar(path: str) -> Optional[tuple]:
    """(meta, decisions) when `path` is a self-play game.log (`selfplay_meta`), else None."""
    meta = selfplay_meta(path)
    if meta is None:
        return None
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "decisions.json"), encoding="utf-8") as f:
        return meta, json.load(f)


def sidecar_by_round(decisions: list) -> dict:
    """kyoku -> {(halfturn index, seat): record}; a key with two records raises."""
    out = {}
    for rec in decisions:
        rnd = out.setdefault(rec["kyoku"], {})
        key = (rec["halfturn"], rec["seat"])
        assert key not in rnd, f"two sidecar records for round {rec['kyoku']} (halfturn, seat) {key}"
        rnd[key] = rec
    return out


def export_live_round(events: list, decisions: list, finished: bool, model=None) -> dict:
    """One round of a self-play game while it plays, from the engine's own events (Round.events as parser event
    dicts) and its sidecar records so far: a finished round as `export_game` exports it from the saved log; an
    unfinished one (process_events pending mode) up to its last settled point: the last halfturn the parser built
    can still change (the call or ron that answers a discard, the action after a draw), so it is left out."""
    gameinfo = GameInfo(GO_TYPE)
    records = next(iter(sidecar_by_round(decisions).values()), {})
    if finished:
        halfturns, tokens = process_events(gameinfo, [events], admission=False)
        return export_round(events, halfturns[0], tokens[0], model, records)
    halfturns, tokens, due = process_events(gameinfo, [events], pending=True, admission=False)
    assert due is not None, "an unfinished round with its result"
    return export_round(events, halfturns[0], tokens[0], model, records, settled=max(0, len(halfturns[0]) - 1))


def export_game(path: str, model) -> dict:
    """The trajectory JSON of one log; raises InvalidGameException as `process` does. A self-play game.log takes its
    sidecar (`selfplay_sidecar`): admission filters off, names from meta.json, decisions from the records."""
    sidecar = selfplay_sidecar(path)
    all_halfturns, all_tokens = process(path, admission=sidecar is None)
    gameinfo, rounds = _get_rounds(path)
    assert len(rounds) == len(all_halfturns), (len(rounds), len(all_halfturns))
    with open(path, "rb") as f:
        n_inits = f.read().count(b"<INIT ")
    by_round = sidecar_by_round(sidecar[1]) if sidecar else {}
    # the sidecar is keyed by kyoku (INIT ordinal), the rounds below by their kept index: equal only with no round cut
    assert not (sidecar and n_inits != len(rounds)), f"a self-play game with {n_inits - len(rounds)} rounds dropped"
    out = {
        "source": selfplay_source(path) if sidecar else os.path.basename(path),
        "players": [s["label"] for s in sidecar[0]["seats"]] if sidecar else _player_names(path),
        "tonnan": bool(gameinfo.tonnan()),
        "complete": n_inits == len(rounds),
        "rounds_dropped": n_inits - len(rounds),
        "rounds": [export_round(e, h, t, model, by_round.get(i, {}) if sidecar else None)
                   for i, (e, h, t) in enumerate(zip(rounds, all_halfturns, all_tokens))],
    }
    if sidecar:
        out["selfplay"] = selfplay_summary(sidecar[0])
    return out


def selfplay_source(path: str) -> str:
    """A self-play game's name: its folder (every game is a game.log)."""
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def selfplay_summary(meta: dict) -> dict:
    return {"seed": meta["seed"], "temperature": meta["temperature"], "final_scores": meta.get("final_scores"),
            "placements": meta.get("placements")}


def main(argv=None) -> Optional[int]:
    ap = argparse.ArgumentParser(description="Tenhou log + checkpoint -> trajectory JSON")
    ap.add_argument("log")
    ap.add_argument("--ckpt", help="checkpoint; without it the JSON carries legal / actual actions only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    model = infer.load_model(args.ckpt) if args.ckpt else None
    try:
        game = export_game(args.log, model)
    except InvalidGameException as e:
        print(f"rejected: {e.msg}")
        return 1
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(game, f, ensure_ascii=False)
    n_steps = sum(len(r["steps"]) for r in game["rounds"])
    print(f"{args.out}: {len(game['rounds'])} rounds, {n_steps} decision points, complete={game['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
