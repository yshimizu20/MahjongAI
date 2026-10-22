"""G2: the self-play engine's checks (T-053). Every check raises on a bad input and prints one line when clean.

    python -m MahjongAI.engine.engine_lib --test                      # scenarios, tables, corruptions, real tags
    python -m MahjongAI.engine.engine_lib --games-dir DIR [--replay N] # every generated game in DIR (G1 output)
    python -m MahjongAI.engine.engine_lib --parity N --data-root ROOT  # prefix-vs-completed parity on N real games
    python -m MahjongAI.engine.engine_lib --corpus N --data-root ROOT  # the rule oracles on N real games

Checks (each raises AssertionError / EngineCheckError with the offending numbers):
  check_kan_emission   every kan in a log follows the kan emission table [C5]; returns the per-row counts
  check_rounds         INIT invariant (100*sum(ten) + 1000*kyotaku == 100000), running scores vs every sc / REACH
                       ten / ba, the round-transition table [C8] between rounds, the owari of the last round
  check_agari_tags     every AGARI's ten / yaku / sc recomputed from hai, m, machi, doraHai (scoring.py); every
                       RYUUKYOKU against an INDEPENDENT reconstruction of the round from its events (rivers,
                       called discards, final hands): the tenpai seats (the `mahjong` library's shanten) are the
                       hai keys, the nagashi qualifiers (all-yaochu river, nothing called) decide type="nm", then
                       the sc (noten bappu / nagashi mangan; aborts: zeros)
  check_sidecar        decisions.json vs the completed parse: complete coverage (exactly one record per decision
                       point the parsed halfturns offer, forced riichi discards excluded), chosen in the offer,
                       offer == the completed halfturn's legal mask, the chosen action of an executed record ==
                       the EXACT action the parser labels executed at that halfturn (and its event), a pass /
                       pre-emption == nothing executed by that seat (pre-empted by a higher-priority executed call
                       or an abort), three rons against the shared ron3 event; with the seat's own model: the
                       recorded candidates == softmax(logits / T) within 1e-6
  check_replay         the same seed and the players rebuilt from meta.json `seats` reproduce the log and the
                       sidecar byte for byte
  check_real_fixtures  the real logs under tests/data/engine_fixtures reproduce the rules the Codex review
                       questioned (S1 West continuation, S2 pao over the whole hand, S3 owari, S5 riichi yakuman)
  check_corpus         real logs: every round transition (next_round vs the next INIT), every exhaustive draw's
                       tenpai set and nagashi type (the reconstruction above), every owari; 0 disagreements
  check_prefix_parity  the parser on every event prefix of a real round offers the same halfturn (type, player,
                       encoding index, legal masks, state tensors) as the completed parse
The scripted scenarios (`SCENARIOS`) play one round each with hand-built walls and rule players and assert the
emitted XML against expectations computed by hand in the scenario (points, deltas, tag order).
"""
import argparse
import collections
import glob
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from mahjong.shanten import Shanten

from MahjongAI import process_xml
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor
from MahjongAI.draw import Naki
from MahjongAI.gameinfo import GameInfo
from MahjongAI.process_xml import process_events
from MahjongAI.turn import DiscardTurn, DuringTurn, PostTurn
from MahjongAI.utils import agari, legal
from MahjongAI.utils.constants import DECISION_ABORT_IDX, DECISION_AGARI_IDX, DECISION_NAKI_IDX, DECISION_REACH_IDX
from MahjongAI.engine import meld_codes, scoring
from MahjongAI.engine.game import GO_TYPE, Game, final_scores, next_round
from MahjongAI.engine.players import (GreedyPlayer, ModelPlayer, UniformPlayer, build_sample, from_description,
                                      head_of, predict)
from MahjongAI.engine.round import (ANKAN_IDX, DURING_KYUUSHU, DURING_PASS, DURING_RIICHI, DURING_TSUMO, KAKAN_IDX,
                                    PON_IDX, POST_PASS, POST_RON, RED_SLOT, Choice, Round, RoundOutcome, RoundSpec,
                                    counts34, legal_mask, render)
from MahjongAI.engine.wall import DEAD_WALL, LIVE_DRAWS, N_INDICATORS, N_RINSHAN, N_TILES, Wall

ABORTS = ("yao9", "reach4", "ron3", "kan4", "kaze4")
FLAG_OF_ID = {1: "riichi", 21: "double_riichi", 2: "ippatsu", 4: "rinshan", 3: "chankan", 5: "haitei", 6: "houtei",
              37: "tenhou", 38: "chiihou"}
TAG_LETTER = {"INIT": "I", "N": "N", "DORA": "O", "REACH": "R", "AGARI": "A", "RYUUKYOKU": "Y"}
HERE = os.path.dirname(os.path.abspath(__file__))
GATE_GAMES = os.path.join(HERE, "..", "tests", "data", "gate_games")
FIXTURES = os.path.join(HERE, "..", "tests", "data", "engine_fixtures")
_processor = TransformerTensorProcessor()
_shanten = Shanten()


class EngineCheckError(AssertionError):
    pass


def fail(msg: str) -> None:
    raise EngineCheckError(msg)


# ---------------------------------------------------------------------------------------------- XML helpers
def events_of(xml: str) -> List[Tuple[str, Dict[str, str]]]:
    out = []
    for m in re.finditer(r"<(\w+)([^>]*?)/?>", xml):
        tag, body = m.group(1), m.group(2)
        if tag in ("mjloggm",):
            continue
        out.append((tag, dict(re.findall(r'(\w+)="([^"]*)"', body))))
    return out


def rounds_of(events: Sequence[Tuple[str, Dict[str, str]]]) -> List[List[Tuple[str, Dict[str, str]]]]:
    rounds: List[list] = []
    for tag, attrs in events:
        if tag == "INIT":
            rounds.append([])
        if rounds:
            rounds[-1].append((tag, attrs))
    return rounds


def letter(tag: str) -> str:
    if tag[0] in "TUVW" and tag[1:].isdigit():
        return "T"
    if tag[0] in "DEFG" and tag[1:].isdigit():
        return "D"
    return TAG_LETTER.get(tag, "?")


def letters(events: Sequence[Tuple[str, Dict[str, str]]]) -> str:
    return " ".join(letter(tag) for tag, _ in events)


def ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",")] if s else []


def init_spec(attrs: Dict[str, str]) -> RoundSpec:
    seed = ints(attrs["seed"])
    return RoundSpec(seed[0], seed[1], seed[2], tuple(ints(attrs["ten"])))


# ---------------------------------------------------------------------------------------------- check: kan emission
def check_kan_emission(xml: str, where: str = "") -> collections.Counter:
    """Every kan's emission order against the table [C5]; returns the named row counts."""
    counts: collections.Counter = collections.Counter()
    for r, events in enumerate(rounds_of(events_of(xml))):
        tags = [t for t, _ in events]
        pending = False  # an open kan's indicator not yet written
        expect: List[Tuple[str, ...]] = []
        n_dora = 0
        for i, (tag, attrs) in enumerate(events):
            lt = letter(tag)
            ctx = f"{where} round {r} event {i} ({tag})"
            if expect:
                allowed = expect.pop(0)
                if lt not in allowed:
                    fail(f"{ctx}: expected one of {allowed} after a kan, got {lt}: {letters(events[max(0, i - 4):i + 2])}")
            if lt == "O":
                n_dora += 1
                if not expect and pending:  # the stray DORA before an open-kan player's discard
                    if i + 1 >= len(events) or letter(events[i + 1][0]) != "D":
                        fail(f"{ctx}: an open kan's indicator must be followed by the discard")
                    pending = False
                    counts["open_kan: N T DORA D"] += 1
                continue
            if lt == "D" and pending:
                fail(f"{ctx}: a discard with the open kan's indicator still unwritten")
            if lt == "A":
                if len(ints(attrs["doraHai"])) != 1 + n_dora:
                    fail(f"{ctx}: doraHai lists {attrs['doraHai']} but {n_dora} DORA tags were written")
                if pending:
                    counts["open_kan rinshan tsumo / chankan: no DORA"] += 1
                continue
            if lt == "Y" and pending:
                fail(f"{ctx}: a draw with an open kan's indicator unwritten")
            if lt != "N":
                continue
            meld = meld_codes.decode_meld(int(attrs["m"]))
            if meld.kind in ("chi", "pon"):
                if pending:
                    fail(f"{ctx}: a chi / pon with an open kan's indicator unwritten")
                continue
            if meld.kind == "ankan":
                if pending:
                    counts["ankan after open kan: N DORA DORA T"] += 1
                    expect = [("O",), ("O",), ("T",)]
                    pending = False
                else:
                    chained = i >= 3 and tags[i - 3] == "N" and letter(tags[i - 2]) == "O" and letter(tags[i - 1]) == "T" \
                        and meld_codes.decode_meld(int(events[i - 3][1]["m"])).kind == "ankan"
                    counts["chained ankan: N DORA T N DORA T" if chained else "ankan: N DORA T"] += 1
                    expect = [("O",), ("T",)]
                continue
            # minkan / kakan
            nxt = letter(tags[i + 1]) if i + 1 < len(tags) else "?"
            if nxt == "A":
                if meld.kind != "kakan":
                    fail(f"{ctx}: an AGARI right after a {meld.kind}")
                counts["kakan + chankan: N AGARI"] += 1
                continue
            if pending:
                counts["second open kan: N T N DORA T DORA D"] += 1
                expect = [("O",), ("T",)]
            else:
                expect = [("T",)]
            pending = True
            counts[f"{meld.kind}: N T ..."] += 1
        if expect:
            fail(f"{where} round {r}: the log ends inside a kan sequence")
    return counts


# ---------------------------------------------------------------------------------------------- check: rounds
def outcome_of(events: Sequence[Tuple[str, Dict[str, str]]], spec: RoundSpec) -> Tuple[RoundOutcome, dict]:
    """The RoundOutcome a round's terminal tags describe, plus the running-score bookkeeping check."""
    scores = list(spec.scores)
    kyotaku = spec.kyotaku
    terminal = [(t, a) for t, a in events if t in ("AGARI", "RYUUKYOKU")]
    if not terminal:
        fail("a round without a terminal tag")
    for tag, attrs in events:
        if tag == "REACH" and attrs.get("step") == "2":
            scores[int(attrs["who"])] -= scoring.RIICHI_STICK
            kyotaku += 1
            if ints(attrs["ten"]) != scores:
                fail(f"REACH ten {attrs['ten']} != running scores {scores}")
        elif tag in ("AGARI", "RYUUKYOKU"):
            sc = ints(attrs["sc"])
            if sc[0::2] != scores:
                fail(f"{tag} sc before {sc[0::2]} != running scores {scores}")
            ba = ints(attrs["ba"])
            first = attrs is terminal[0][1]
            if ba[0] != spec.honba or ba[1] != (kyotaku if first else 0):
                fail(f"{tag} ba {ba} != (honba {spec.honba}, sticks {kyotaku if first else 0})")
            scores = [s + d for s, d in zip(sc[0::2], sc[1::2])]
            if tag == "AGARI":
                kyotaku = 0
    tag, attrs = terminal[-1]
    if tag == "AGARI":
        winners = [int(a["who"]) for _, a in terminal]
        out = RoundOutcome("agari", scores, 0, dealer_won=spec.oya in winners, winners=winners,
                           from_who=int(attrs["fromWho"]))
    elif attrs.get("type") in ABORTS:
        out = RoundOutcome("abort", scores, kyotaku, abort_type=attrs["type"])
    else:
        out = RoundOutcome("exhaustive", scores, kyotaku, dealer_tenpai=f"hai{spec.oya}" in attrs,
                           nagashi=attrs.get("type") == "nm")
    return out, {"scores": scores, "kyotaku": kyotaku, "owari": attrs.get("owari")}


def owari_exact(where: str) -> bool:
    """A real log's name (yyyymmddhhgm-...) dated from scoring.OWARI_EXACT_FROM keeps 0.1 precision in its owari;
    the engine's own games (no date) use the rounded rule."""
    m = re.match(r"(\d{8})\d{2}gm-", os.path.basename(where))
    return bool(m) and m.group(1) >= scoring.OWARI_EXACT_FROM


def check_rounds(xml: str, where: str = "", complete: bool = True) -> collections.Counter:
    """INIT invariant, running scores, the transition table between rounds, the owari. complete=False: a round
    fragment without a game end (the scenarios)."""
    counts: collections.Counter = collections.Counter()
    rounds = rounds_of(events_of(xml))
    if not rounds:
        fail(f"{where}: no INIT")
    spec = init_spec(rounds[0][0][1])
    for r, events in enumerate(rounds):
        got = init_spec(events[0][1])
        if sum(got.scores) + 10 * got.kyotaku != 1000:
            fail(f"{where} round {r}: INIT ten {got.scores} + {got.kyotaku} sticks != 100,000")
        if got != spec:
            fail(f"{where} round {r}: INIT {got} but the transition table expects {spec}")
        if r == 0 and got.index != 0 and complete:
            fail(f"{where}: a game starting at round {got.index}")
        outcome, end = outcome_of(events, got)
        counts[f"transition: {outcome.kind}{' renchan' if outcome.renchan else ''}"] += 1
        nxt = next_round(got, outcome)
        last = r == len(rounds) - 1
        if not complete and last:
            if end["owari"] is not None:
                fail(f"{where}: an owari on a fragment")
            break
        if (nxt is None) != last:
            fail(f"{where} round {r}: table says {'game over' if nxt is None else 'continue'}, log says "
                 f"{'game over' if last else 'continue'} ({got} -> {outcome})")
        if last:
            final = final_scores(end["scores"], end["kyotaku"])
            exact = owari_exact(where)
            if end["owari"] != scoring.owari_attr(final, exact):
                fail(f"{where}: owari {end['owari']} != {scoring.owari_attr(final, exact)}")
            counts["owari"] += 1
            if min(final) < 0:
                counts["transition: tobi"] += 1
            elif got.index >= 8:
                counts["transition: extension round"] += 1
        else:
            if end["owari"] is not None:
                fail(f"{where} round {r}: an owari before the last round")
            spec = nxt
    return counts


# ---------------------------------------------------------------------------------------------- check: AGARI tags
def recompute_agari(attrs: Dict[str, str], round_index: int, oya: int,
                    first: bool = True) -> Tuple[scoring.HandScore, List[int]]:
    """first=False: the second (third) AGARI of a double ron, which collects neither honba nor sticks although its
    ba still names the honba."""
    who, from_who = int(attrs["who"]), int(attrs["fromWho"])
    ids = set(ints(attrs.get("yaku", ""))[0::2])
    yakuman_ids = set(ints(attrs.get("yakuman", "")))
    flags = {FLAG_OF_ID[i]: True for i in ids | yakuman_ids if i in FLAG_OF_ID}
    wf = scoring.WinFlags(tsumo=who == from_who, seat_wind=agari.seat_wind(who, oya),
                          round_wind=agari.round_wind(round_index), **flags)
    ura = ints(attrs.get("doraHaiUra", ""))
    score = scoring.score_hand(ints(attrs["hai"]), ints(attrs.get("m", "")), int(attrs["machi"]), wf,
                               ints(attrs["doraHai"]), ura)
    ba = ints(attrs["ba"])
    pao = {i: int(attrs["paoWho"]) for i in score.yakuman if i in scoring.PAO_YAKUMAN} if "paoWho" in attrs else None
    deltas = scoring.agari_deltas(score, who, from_who, oya, ba[0] if first else 0, ba[1], pao)
    return score, deltas


def check_agari_tag(attrs: Dict[str, str], round_index: int, oya: int, where: str = "", first: bool = True) -> str:
    """One AGARI tag against scoring.py; returns the coverage label."""
    score, deltas = recompute_agari(attrs, round_index, oya, first)
    ten = ints(attrs["ten"])
    yakuman = "yakuman" in attrs
    if yakuman:
        if tuple(sorted(ints(attrs["yakuman"]))) != score.yakuman or ten[1:] != [score.points, score.limit]:
            fail(f"{where}: yakuman {attrs['yakuman']} ten {attrs['ten']} != {score}")
    else:
        got = {i: h for i, h in zip(ints(attrs["yaku"])[0::2], ints(attrs["yaku"])[1::2]) if h > 0}
        want = {i: h for i, h in score.yaku if h > 0}
        if got != want or ten != [score.fu, score.points, score.limit]:
            fail(f"{where}: yaku {attrs['yaku']} ten {attrs['ten']} != {score}")
        if (53 in dict(score.yaku)) != ("53" in attrs["yaku"].split(",")[0::2]):
            fail(f"{where}: ura listing differs: {attrs['yaku']} vs {score.yaku}")
    sc = ints(attrs["sc"])
    if sc[1::2] != deltas:
        fail(f"{where}: sc deltas {sc[1::2]} != {deltas} ({attrs})")
    kind = "tsumo" if attrs["who"] == attrs["fromWho"] else "ron"
    extra = ("+melds" if "m" in attrs else "") + ("+riichi" if "doraHaiUra" in attrs else "") + \
        ("+yakuman" if yakuman else "") + ("+pao" if "paoWho" in attrs else "") + ("" if first else "+second") + \
        (f"+honba{attrs['ba'].split(',')[0]}" if attrs["ba"].split(",")[0] != "0" else "")
    return kind + extra


class RoundState(collections.namedtuple("RoundState", ["hands", "rivers", "called", "melds"])):
    """A round rebuilt from its events alone (no engine state): hands[q] = the concealed ids at the end, rivers[q]
    = the discards in order, called[q] = one of q's discards was chi'd / pon'd / daiminkan'd, melds[q] = codes."""


def reconstruct_round(events: Sequence[Tuple[str, Dict[str, str]]]) -> RoundState:
    init = events[0][1]
    hands = [ints(init[f"hai{q}"]) for q in range(4)]
    rivers: List[List[int]] = [[] for _ in range(4)]
    called = [False] * 4
    melds: List[List[int]] = [[] for _ in range(4)]
    last_discarder: Optional[int] = None
    for tag, attrs in events[1:]:
        lt = letter(tag)
        if lt == "T":
            hands["TUVW".index(tag[0])].append(int(tag[1:]))
        elif lt == "D":
            q, tile = "DEFG".index(tag[0]), int(tag[1:])
            hands[q].remove(tile)
            rivers[q].append(tile)
            last_discarder = q
        elif tag == "N":
            q, m = int(attrs["who"]), meld_codes.decode_meld(int(attrs["m"]))
            if m.kind in ("chi", "pon", "minkan"):
                if last_discarder is None or rivers[last_discarder][-1] != m.called:
                    fail(f"a {m.kind} of {m.called} that is not the last discard")
                called[last_discarder] = True
                for t in m.ids:
                    if t != m.called:
                        hands[q].remove(t)
            elif m.kind == "ankan":
                for t in m.ids:
                    hands[q].remove(t)
            else:
                hands[q].remove(m.added)
                melds[q] = [c for c in melds[q] if meld_codes.decode_meld(c).ids[0] // 4 != m.ids[0] // 4]
            melds[q].append(int(attrs["m"]))
        elif tag in ("AGARI", "RYUUKYOKU"):
            break
    return RoundState(hands, rivers, called, melds)


def is_tenpai(concealed_ids: Sequence[int]) -> bool:
    """The `mahjong` library's shanten on the concealed tiles (melds implied by the count): 0 = tenpai."""
    return _shanten.calculate_shanten(counts34(concealed_ids).tolist()) == 0


def nagashi_qualifiers(state: RoundState) -> List[int]:
    """Tenhou's nagashi mangan: every discard of the seat is a terminal / honor and none was called."""
    return [q for q in range(4) if state.rivers[q] and not state.called[q]
            and all(t // 4 in legal.YAOCHU_KINDS for t in state.rivers[q])]


def check_ryuukyoku_tag(events: Sequence[Tuple[str, Dict[str, str]]], attrs: Dict[str, str], oya: int,
                        where: str = "") -> str:
    """One RYUUKYOKU against the round rebuilt from its events: tenpai seats, nagashi qualifiers, then the sc."""
    sc = ints(attrs["sc"])
    type_ = attrs.get("type")
    if type_ in ABORTS:
        want = [0] * 4
    else:
        state = reconstruct_round(events)
        tenpai = [is_tenpai(state.hands[q]) for q in range(4)]
        shown = [f"hai{q}" in attrs for q in range(4)]
        if shown != tenpai:
            fail(f"{where}: tenpai seats {[q for q in range(4) if tenpai[q]]} (library shanten on the rebuilt hands) "
                 f"but hai keys for {[q for q in range(4) if shown[q]]}: {attrs}")
        for q in range(4):
            if shown[q] and ints(attrs[f"hai{q}"]) != sorted(state.hands[q]):
                fail(f"{where}: hai{q} {attrs[f'hai{q}']} != the rebuilt hand {sorted(state.hands[q])}")
        qualifiers = nagashi_qualifiers(state)
        if (type_ == "nm") != bool(qualifiers):
            fail(f"{where}: nagashi mangan qualifiers {qualifiers} from the rivers but type={type_!r}: {attrs}")
        want = scoring.nagashi_deltas(qualifiers, oya) if qualifiers else scoring.ryuukyoku_deltas(tenpai)
    if sc[1::2] != want:
        fail(f"{where}: RYUUKYOKU deltas {sc[1::2]} != {want} ({attrs})")
    return type_ or f"exhaustive tenpai {sum(f'hai{q}' in attrs for q in range(4))}"


def check_agari_tags(xml: str, where: str = "") -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for r, events in enumerate(rounds_of(events_of(xml))):
        spec = init_spec(events[0][1])
        riichi: Set[int] = set()  # accepted riichi declarations (REACH step 2), the S5 oracle for doraHaiUra
        for i, (tag, attrs) in enumerate(events):
            if tag == "REACH" and attrs["step"] == "2":
                riichi.add(int(attrs["who"]))
            if tag == "AGARI":
                first = events[i - 1][0] != "AGARI"
                ctx = f"{where} round {r} event {i}"
                if ("doraHaiUra" in attrs) != (int(attrs["who"]) in riichi):
                    fail(f"{ctx}: ura indicators {'listed' if 'doraHaiUra' in attrs else 'missing'} but the winner "
                         f"{int(attrs['who'])} {'is' if int(attrs['who']) in riichi else 'is not'} in riichi (accepted: "
                         f"{sorted(riichi)})")
                counts[check_agari_tag(attrs, spec.index, spec.oya, ctx, first)] += 1
            elif tag == "RYUUKYOKU":
                counts[check_ryuukyoku_tag(events, attrs, spec.oya, f"{where} round {r} event {i}")] += 1
    return counts


# ---------------------------------------------------------------------------------------------- check: sidecar
def parse_xml_string(xml: str, pending: bool = False):
    """The parser on a generated log held in memory (admission off)."""
    rounds = [[{"event": t, "attr": a} for t, a in r] for r in rounds_of(events_of(xml))]
    return process_events(GameInfo(GO_TYPE), rounds, pending=pending, admission=False)


def _tile_of_tag(tag: str) -> int:
    return int(tag[1:])


CALL_PRIORITY = {POST_RON: 3}  # ron > pon / daiminkan > chi; a pass has none


def call_priority(action: int) -> int:
    if action == POST_PASS:
        return 0
    return CALL_PRIORITY.get(action, 2 if action >= PON_IDX else 1)


def executed_action(ht, seat: int) -> int:
    """The action index the completed parse labels executed for `seat` at this halfturn (the training target):
    the pass index when the seat executed nothing."""
    if isinstance(ht, DuringTurn):
        idx = (DECISION_AGARI_IDX, DECISION_REACH_IDX, DECISION_NAKI_IDX, DECISION_ABORT_IDX)
        agari, reach, naki, abort = (ht.decisions[i] for i in idx)
        if agari and agari[0].executed:
            return DURING_TSUMO
        if reach and reach[0].executed:
            return DURING_RIICHI
        for d in naki:
            if d.executed:
                return d.naki.get_during_turn_filter_idx()
        if abort and abort[0].executed:
            return DURING_KYUUSHU
        return DURING_PASS
    if isinstance(ht, DiscardTurn):
        return RED_SLOT.get(ht.discarded_tile, ht.discarded_tile // 4)
    decisions = ht.decisions[seat]
    if any(d.executed for d in decisions[DECISION_AGARI_IDX]):
        return POST_RON
    for d in decisions[DECISION_NAKI_IDX]:
        if d.executed:
            return d.naki.get_post_turn_filter_idx()
    return POST_PASS


def expected_decisions(H) -> Dict[Tuple[int, int, int], str]:
    """(kyoku, halfturn, seat) -> kind for every decision point the parsed halfturns offer: a DuringTurn with any
    offer, a DiscardTurn unless forced (a riichi player's non-declaration discard: the drawn tile), a PostTurn
    once per responder with a ron or call offer. The sidecar must hold exactly one record per key."""
    out: Dict[Tuple[int, int, int], str] = {}
    for r, halfturns in enumerate(H):
        for i, ht in enumerate(halfturns):
            if isinstance(ht, DuringTurn):
                if sum(map(len, ht.decisions)):
                    out[(r, i, ht.player)] = "during"
            elif isinstance(ht, DiscardTurn):
                if not (ht.stateObj.reaches[ht.player] and not ht.is_riichi_declaration):
                    out[(r, i, ht.player)] = "discard"
            else:
                for q in range(4):
                    if q != ht.player and (ht.decisions[q][DECISION_AGARI_IDX] or ht.decisions[q][DECISION_NAKI_IDX]):
                        out[(r, i, q)] = "post"
    return out


def check_sidecar(xml: str, decisions: Sequence[dict], models: Optional[Dict[int, Tuple[object, float]]] = None,
                  where: str = "", tol: float = 1e-6) -> collections.Counter:
    """decisions.json against the completed parse of game.log (see the module docstring). models: seat -> (model,
    temperature) for the seats a ModelPlayer played: their candidates are compared with a fresh forward pass."""
    counts: collections.Counter = collections.Counter()
    H, T = parse_xml_string(xml)
    rounds = rounds_of(events_of(xml))
    expected = expected_decisions(H)
    keys = [(d["kyoku"], d["halfturn"], d["seat"]) for d in decisions]
    if len(set(keys)) != len(keys):
        dup = next(k for k in keys if keys.count(k) > 1)
        fail(f"{where}: decision coverage: {dup} recorded {keys.count(dup)} times")
    missing = sorted(set(expected) - set(keys))
    extra = sorted(set(keys) - set(expected))
    if missing or extra:
        fail(f"{where}: decision coverage: {len(expected)} decision points in the parse, {len(keys)} records; "
             f"{len(missing)} unrecorded (first {missing[:3]}), {len(extra)} without a decision point "
             f"(first {extra[:3]})")
    counts["decision coverage"] += len(expected)
    by_halfturn: Dict[Tuple[int, int], List[dict]] = collections.defaultdict(list)
    for d in decisions:
        by_halfturn[(d["kyoku"], d["halfturn"])].append(d)
    for n, d in enumerate(decisions):
        ctx = f"{where} decision k={d['k']} ({d['kind']} seat {d['seat']} kyoku {d['kyoku']} halfturn {d['halfturn']})"
        if d["k"] != n:
            fail(f"{ctx}: k out of order")
        if n and (d["kyoku"], d["halfturn"]) < (decisions[n - 1]["kyoku"], decisions[n - 1]["halfturn"]):
            fail(f"{ctx}: records out of play order")
        if d["chosen"] not in d["offer"]:
            fail(f"{ctx}: chosen {d['chosen']} not in offer {d['offer']}")
        if sorted(int(a) for a in d["candidates"]) != sorted(d["offer"]):
            fail(f"{ctx}: candidates keys {sorted(d['candidates'])} != offer {d['offer']}")
        if abs(sum(d["candidates"].values()) - 1.0) > 1e-4:
            fail(f"{ctx}: candidates sum to {sum(d['candidates'].values())}")
        if d["resolved"] is None:
            fail(f"{ctx}: unresolved")
        ht = H[d["kyoku"]][d["halfturn"]]
        if expected[(d["kyoku"], d["halfturn"], d["seat"])] != d["kind"]:
            fail(f"{ctx}: the completed halfturn is a {head_of(ht)}")
        offer = np.flatnonzero(legal_mask(ht, d["seat"])).tolist()
        if offer != d["offer"]:
            fail(f"{ctx}: offer {d['offer']} != completed legal mask {offer}")
        counts[f"offer parity: {d['kind']}"] += 1
        outcome, event = d["resolved"]["outcome"], d["resolved"]["event"]
        label = executed_action(ht, d["seat"])
        tags = [(t, a) for t, a in rounds[d["kyoku"]]]
        if outcome == "executed":
            if label != d["chosen"]:
                fail(f"{ctx}: executed action {d['chosen']} but the parse labels {label} executed")
            _check_executed(d, ht, event, tags, by_halfturn[(d["kyoku"], d["halfturn"])], ctx)
        elif outcome == "passed":
            if d["chosen"] != POST_PASS or event is not None or d["kind"] != "post":
                fail(f"{ctx}: a pass with chosen {d['chosen']} / event {event}")
            if label != POST_PASS:
                fail(f"{ctx}: executed action {label} labelled by the parse but the record says passed")
        elif outcome == "preempted":
            if d["kind"] != "post" or d["chosen"] == POST_PASS:
                fail(f"{ctx}: a pre-empted {d['kind']} pass")
            if label != POST_PASS:
                fail(f"{ctx}: executed action {label} labelled by the parse but the record says pre-empted")
            if event in ("reach4", "kaze4"):
                if tags[-1][0] != "RYUUKYOKU" or tags[-1][1].get("type") != event:
                    fail(f"{ctx}: pre-empted by a {event} abort the round does not end with")
            else:
                winners = [executed_action(ht, q) for q in range(4) if q != d["seat"]]  # the parse's labels
                winners = [a for a in winners if a != POST_PASS]
                if not winners or max(map(call_priority, winners)) < call_priority(d["chosen"]):
                    fail(f"{ctx}: {d['chosen']} pre-empted by nobody of higher priority (executed: {winners})")
        else:
            fail(f"{ctx}: unknown outcome {outcome}")
        counts[f"resolved: {d['kind']} {outcome}"] += 1
        if models and d["seat"] in models:
            model, temperature = models[d["seat"]]
            if not d["player"].startswith("model:"):
                fail(f"{ctx}: seat {d['seat']} played by {d['player']} but meta names a model")
            sample = build_sample(ht, T[d["kyoku"]], d["seat"])
            logits = predict(model, sample)
            probs = np.array([float(p) for p in (logits / temperature if temperature > 0 else logits).softmax(0)])
            diff = max(abs(probs[a] - d["candidates"][str(a)]) for a in d["offer"])
            if diff > tol:
                fail(f"{ctx}: candidates differ from a fresh forward pass by {diff:.2e}")
            counts["model distribution parity"] += 1
    return counts


def _check_executed(d: dict, ht, event: Optional[str], tags, siblings: Sequence[dict], ctx: str) -> None:
    """The recorded event of an executed record is the XML event of that exact action."""
    kind, chosen, seat = d["kind"], d["chosen"], d["seat"]
    if event is None:
        if kind != "during" or chosen != DURING_PASS:
            fail(f"{ctx}: executed without an event")
        return
    tag, attrs = events_of(event)[0]
    without_owari = [(t, {k: v for k, v in a.items() if k != "owari"}) for t, a in tags]  # the owari is added later
    if (tag, attrs) not in without_owari:
        fail(f"{ctx}: event {event} not in the round's log")
    if kind == "discard":
        if letter(tag) != "D" or "DEFG".index(tag[0]) != seat:
            fail(f"{ctx}: discard event {event}")
        tile = _tile_of_tag(tag)
        if ht.discarded_tile != tile or RED_SLOT.get(tile, tile // 4) != chosen:
            fail(f"{ctx}: discard slot {chosen} vs tile {tile} (completed: {ht.discarded_tile})")
    elif kind == "during":
        if chosen == DURING_TSUMO:
            if tag != "AGARI" or attrs["who"] != attrs["fromWho"] or int(attrs["who"]) != seat:
                fail(f"{ctx}: tsumo event {event}")
        elif chosen == DURING_RIICHI:
            if tag != "REACH" or attrs["step"] != "1" or int(attrs["who"]) != seat:
                fail(f"{ctx}: riichi event {event}")
        elif chosen == DURING_KYUUSHU:
            if tag != "RYUUKYOKU" or attrs.get("type") != "yao9":
                fail(f"{ctx}: kyuushu event {event}")
        else:
            naki = Naki(int(attrs["m"]))
            if tag != "N" or int(attrs["who"]) != seat or not (naki.is_ankan() or naki.is_kakan()) \
                    or naki.get_during_turn_filter_idx() != chosen:
                fail(f"{ctx}: kan event {event} vs chosen {chosen}")
    else:
        if chosen == POST_RON:
            rons = [o for o in siblings if o["chosen"] == POST_RON and o["resolved"]["outcome"] == "executed"]
            if len(rons) == 3:  # triple ron: the three records share the ron3 abort event
                if tag != "RYUUKYOKU" or attrs.get("type") != "ron3" or f"hai{seat}" not in attrs \
                        or any(o["resolved"]["event"] != event for o in rons):
                    fail(f"{ctx}: ron3 event {event}")
            elif tag != "AGARI" or int(attrs["who"]) != seat or attrs["who"] == attrs["fromWho"]:
                fail(f"{ctx}: ron event {event}")
        else:
            naki = Naki(int(attrs["m"]))
            if tag != "N" or int(attrs["who"]) != seat or naki.get_post_turn_filter_idx() != chosen:
                fail(f"{ctx}: call event {event} vs chosen {chosen}")


# ---------------------------------------------------------------------------------------------- check: replay
def check_replay(xml: str, decisions: Sequence[dict], seed: int, players: Sequence, labels: Sequence[str],
                 temperature: float = 0.0, where: str = "") -> None:
    """The same seed with fresh players reproduces game.log and decisions.json (time_ms aside)."""
    for p in players:
        p.reseed(seed)
    record = Game(players, labels, seed, temperature).play()
    if record.xml != xml:
        i = next((i for i, (a, b) in enumerate(zip(record.xml, xml)) if a != b), min(len(record.xml), len(xml)))
        fail(f"{where}: the replay diverges at char {i}: ...{record.xml[max(0, i - 60):i + 40]}...")
    strip = lambda ds: [{k: v for k, v in d.items() if k != "time_ms"} for d in ds]  # noqa: E731
    if strip(record.decisions) != strip(decisions):
        fail(f"{where}: the replayed sidecar differs")


# ---------------------------------------------------------------------------------------------- check: prefix parity
def _state_arrays(halfturn, seat: int):
    obj = halfturn.responder_states[seat] if isinstance(halfturn, PostTurn) else halfturn.stateObj
    return [t.numpy() for t in _processor._convert_state_obj_to_tensors(obj)]


def _same_halfturn(pending, full, ctx: str) -> None:
    if type(pending) is not type(full) or pending.player != full.player or pending.encoding_idx != full.encoding_idx:
        fail(f"{ctx}: prefix {type(pending).__name__} seat {pending.player} idx {pending.encoding_idx} vs completed "
             f"{type(full).__name__} seat {full.player} idx {full.encoding_idx}")
    seats = [q for q in range(4) if q != full.player] if isinstance(full, PostTurn) else [full.player]
    for q in seats:
        a, b = legal_mask(pending, q), legal_mask(full, q)
        if not np.array_equal(a, b):
            fail(f"{ctx}: seat {q} legal mask differs: prefix {np.flatnonzero(a).tolist()} vs completed "
                 f"{np.flatnonzero(b).tolist()}")
        for x, y in zip(_state_arrays(pending, q), _state_arrays(full, q)):
            if not np.array_equal(x, y):
                fail(f"{ctx}: seat {q} state tensors differ")


def check_prefix_parity(path: str, prefix_parser: Callable = process_events) -> collections.Counter:
    """Every strict event prefix of every round of one real game: the due halfturns equal the completed ones."""
    counts: collections.Counter = collections.Counter()
    gameinfo, rounds = process_xml._get_rounds(path)
    H_full, T_full = process_events(gameinfo, rounds, admission=False)
    for r, events in enumerate(rounds):
        for j in range(1, len(events)):
            if events[j - 1]["event"] in ("AGARI", "RYUUKYOKU"):
                continue  # a double ron: the prefix after the first AGARI is already a finished round
            H, T, due = prefix_parser(gameinfo, [events[:j]], pending=True, admission=False)
            ctx = f"{os.path.basename(path)} round {r} prefix {j} ({events[j - 1]['event']})"
            if T[0] != T_full[r][:len(T[0])]:
                fail(f"{ctx}: token prefix differs")
            nxt = events[j]["event"]
            discard_next = nxt[0] in "DEFG" and nxt[1:].isdigit()
            for h in due:
                i = len(H[0]) - 1 if H[0] and H[0][-1] is h else len(H[0])
                if isinstance(h, DiscardTurn) and i == len(H[0]) and not discard_next:
                    # the seat does something else first (riichi, a kan, tsumo, kyuushu): the completed round has
                    # no DiscardTurn here, or a later one (the riichi declaration discard, built after REACH step 1,
                    # one token later); the engine re-parses after every such event
                    counts["parity: discard pending, no discard follows"] += 1
                    continue
                if i >= len(H_full[r]):
                    fail(f"{ctx}: due halfturn {i} beyond the completed round ({len(H_full[r])})")
                _same_halfturn(h, H_full[r][i], ctx)
                counts[f"parity: {head_of(h)}{' pending' if i == len(H[0]) else ''}"] += 1
            counts["prefixes"] += 1
    return counts


# ---------------------------------------------------------------------------------------------- scripted scenarios
SUITS = {"m": 0, "p": 9, "s": 18, "z": 27}


def kinds(spec: str) -> List[int]:
    """'234m567p1z' -> kinds; z: 1-4 winds E S W N, 5 haku 6 hatsu 7 chun."""
    out, digits = [], ""
    for ch in spec:
        if ch.isdigit():
            digits += ch
        elif ch in SUITS:
            out += [SUITS[ch] + int(d) - 1 for d in digits]
            digits = ""
        elif ch != " ":
            raise ValueError(spec)
    assert not digits, spec
    return out


class Deck:
    """Hands out ids per kind, the red five last."""

    def __init__(self):
        self.pool = {k: [4 * k + 1, 4 * k + 2, 4 * k + 3, 4 * k] for k in range(34)}

    def take(self, kind: int) -> int:
        assert self.pool[kind], f"a fifth tile of kind {kind}"
        return self.pool[kind].pop(0)

    def rest(self) -> List[int]:
        """Honors first, then terminals and the rest: the filler no scripted hand wants."""
        order = sorted(range(34), key=lambda k: (k < 27, k % 9 not in (0, 8), -k))
        return [t for k in order for t in self.pool[k]]


class ScriptPlayer:
    """A rule player for the scenarios: `discards` (kinds, in order; then tsumogiri, then `spares`, then the
    highest legal slot), `take` (what it accepts, in preference order: tsumo riichi kyuushu ankan kakan ron pon
    chi minkan), `ron_from` (seats whose discard it rons; None = any)."""
    label = "script"

    def __init__(self, discards: str = "", take: Sequence[str] = (), ron_from: Optional[Iterable[int]] = None,
                 spares: str = ""):
        self.discards = kinds(discards)
        self.take = tuple(take)
        self.ron_from = None if ron_from is None else set(ron_from)
        self.spares = kinds(spares)
        self.round: Optional[Round] = None

    def reseed(self, seed: int) -> None:
        pass

    def __call__(self, halfturn, tokens, seat) -> Choice:
        legal = set(np.flatnonzero(legal_mask(halfturn, seat)).tolist())
        if isinstance(halfturn, DuringTurn):
            action = DURING_PASS
            for want in self.take:
                cand = {"tsumo": [DURING_TSUMO], "riichi": [DURING_RIICHI], "kyuushu": [DURING_KYUUSHU],
                        "ankan": range(ANKAN_IDX, KAKAN_IDX), "kakan": range(KAKAN_IDX, DURING_KYUUSHU)}.get(want, [])
                hit = [a for a in cand if a in legal]
                if hit:
                    action = hit[0]
                    break
        elif isinstance(halfturn, DiscardTurn):
            action = None
            while self.discards and action is None:
                k = self.discards.pop(0)
                if k in legal:
                    action = k
            if action is None:
                drawn = self.round.drawn
                slot = RED_SLOT.get(drawn, drawn // 4) if drawn is not None else None
                spares = [k for k in self.spares if k in legal]
                action = slot if slot in legal else spares[0] if spares else max(legal)
        else:
            action = POST_PASS
            if "ron" in self.take and POST_RON in legal and (self.ron_from is None or halfturn.player in self.ron_from):
                action = POST_RON
            else:
                for want in self.take:
                    for d in halfturn.decisions[seat][DECISION_NAKI_IDX]:
                        n = d.naki
                        if (want == "pon" and n.is_pon()) or (want == "chi" and n.is_chi()) or \
                                (want == "minkan" and n.is_minkan()):
                            action = n.get_post_turn_filter_idx()
                            break
                    if action != POST_PASS:
                        break
        return Choice(action, {a: float(a == action) for a in sorted(legal)}, False, 0.0)


@dataclass
class Scenario:
    name: str
    hands: Dict[int, str]  # seat -> 13 kinds; other seats are filled
    players: Dict[int, ScriptPlayer] = field(default_factory=dict)
    draws: Dict[int, object] = field(default_factory=dict)  # live draw position (0 = the dealer's first) -> kind (str or int)
    rinshan: str = ""  # rinshan tiles 0.. (the rest filled)
    indicators: str = "4z"  # dora indicators 0.. (the rest filled); North -> dora East
    spec: RoundSpec = RoundSpec(0, 0, 0, (250, 250, 250, 250))
    check: Optional[Callable[["Played"], None]] = None


@dataclass
class Played:
    scenario: Scenario
    round: Round
    outcome: RoundOutcome
    wall_order: List[int]
    hands: List[List[int]]  # the dealt hands
    draws: List[int]
    rinshan: List[int]
    indicators: List[int]

    @property
    def events(self):
        return self.round.events

    @property
    def xml(self) -> str:
        return self.round.xml()

    @property
    def letters(self) -> str:
        return letters(self.round.events)

    def tags(self, name: str) -> List[Dict[str, str]]:
        return [a for t, a in self.round.events if t == name]


def build_wall(sc: Scenario) -> Tuple[List[int], List[List[int]], List[int], List[int], List[int]]:
    deck = Deck()
    hands: List[Optional[List[int]]] = [None] * 4
    for q, spec in sc.hands.items():
        ks = kinds(spec)
        assert len(ks) == 13, (sc.name, q, len(ks))
        hands[q] = [deck.take(k) for k in ks]
    draws: List[Optional[int]] = [None] * LIVE_DRAWS
    for pos, spec in sc.draws.items():
        draws[pos] = deck.take(spec if isinstance(spec, int) else kinds(spec)[0])
    rinshan = [deck.take(k) for k in kinds(sc.rinshan)]
    indicators = [deck.take(k) for k in kinds(sc.indicators)]
    rest = deck.rest()  # honors first, simples last: the dead wall fills from the simples' end, the draws from the
    rinshan += [rest.pop() for _ in range(N_RINSHAN - len(rinshan))]  # honors' end, the hands from what is left
    indicators += [rest.pop() for _ in range(N_INDICATORS - len(indicators))]
    ura = [rest.pop() for _ in range(DEAD_WALL - N_RINSHAN - N_INDICATORS)]
    draws = [t if t is not None else rest.pop(0) for t in draws]
    for q in range(4):
        if hands[q] is None:
            hands[q] = [rest.pop(0) for _ in range(13)]
    assert not rest, len(rest)
    oya = sc.spec.oya
    order: List[int] = []
    for r in range(3):
        for k in range(4):
            order += hands[(oya + k) % 4][4 * r:4 * r + 4]
    for k in range(4):
        order.append(hands[(oya + k) % 4][12])
    order += draws + rinshan + indicators + ura
    assert len(order) == N_TILES, len(order)
    return order, [list(h) for h in hands], draws, rinshan, indicators


def play_scenario(sc: Scenario) -> Played:
    order, hands, draws, rinshan, indicators = build_wall(sc)
    players = [sc.players.get(q, ScriptPlayer()) for q in range(4)]
    rnd = Round(sc.spec, Wall(order), (0, 0), GameInfo(GO_TYPE), labels=["script"] * 4)
    for p in players:
        p.round = rnd
    gen = rnd.play()
    try:
        point = next(gen)
        while True:
            point = gen.send(players[point.seat](point.halfturn, point.tokens, point.seat))
    except StopIteration as stop:
        outcome = stop.value
    played = Played(sc, rnd, outcome, order, hands, draws, rinshan, indicators)
    parse_xml_string(played.xml)  # the parser accepts the generated round
    return played


def expect(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def _ids(played: Played, seat: int, spec: str) -> List[int]:
    """The ids dealt to `seat` for these kinds (first copies)."""
    out, hand = [], list(played.hands[seat])
    for k in kinds(spec):
        t = next(t for t in hand if t // 4 == k)
        hand.remove(t)
        out.append(t)
    return out


# --- the scenarios ----------------------------------------------------------------------------------------------
def sc_ankan_chained() -> Scenario:
    """Dealer: two ankan at the first turn -> N DORA T N DORA T, then tsumogiri to the exhaustive draw."""
    def check(p: Played):
        expect(p.letters.startswith("I T N O T N O T D"), p.letters[:40])
        n = p.tags("N")
        expect([int(a["m"]) for a in n[:2]] == [meld_codes.encode_ankan(0), meld_codes.encode_ankan(10)], n)
        expect([int(a["hai"]) for a in p.tags("DORA")[:2]] == p.indicators[1:3], (p.tags("DORA"), p.indicators))
        draws = [int(t[1:]) for t, _ in p.events if letter(t) == "T"]
        expect(draws[1:3] == p.rinshan[:2], (draws[:3], p.rinshan))
        recs = [d for d in p.round.decisions if d["kind"] == "during"][:2]
        expect([d["resolved"]["event"] for d in recs] == [render("N", a) for a in n[:2]], recs)
        expect(p.outcome.kind == "exhaustive", p.outcome)
    return Scenario("ankan, chained ankan", {0: "1111m2222p345s7p9s"}, {0: ScriptPlayer(take=["ankan"])}, check=check)


def sc_ankan_after_open_kan() -> Scenario:
    """Seat 1 daiminkans the dealer's 3m, draws the 4th 2p from the rinshan, ankans: N T N DORA DORA T D."""
    def check(p: Played):
        expect(p.letters.startswith("I T D N T N O O T D"), p.letters[:40])
        n = p.tags("N")
        tile_3m = _ids(p, 0, "3m")[0]
        expect(int(n[0]["m"]) == meld_codes.encode_minkan(tile_3m, 3), (n[0], tile_3m))
        expect(int(n[1]["m"]) == meld_codes.encode_ankan(10), n[1])
        expect([int(a["hai"]) for a in p.tags("DORA")[:2]] == p.indicators[1:3], (p.tags("DORA"), p.indicators))
    return Scenario("daiminkan, then ankan after the open kan",
                    {0: "3m2468m468p9s1234z", 1: "333m222p19s12345z"},
                    {0: ScriptPlayer(discards="3m"), 1: ScriptPlayer(take=["minkan", "ankan"])},
                    rinshan="2p", check=check)


def sc_pon_minkan_kakan() -> Scenario:
    """Seat 1 pons 6s, later daiminkans 3m, the rinshan tile is the 4th 6s -> kakan: N T N DORA T DORA D."""
    def check(p: Played):
        expect("D N D T D T D T D N T N O T O D" in p.letters, p.letters[:60])
        n = p.tags("N")
        pon, minkan, kakan = (int(a["m"]) for a in n[:3])
        expect(Naki(pon).is_pon() and Naki(minkan).is_minkan() and Naki(kakan).is_kakan(), n[:3])
        expect(kakan == meld_codes.encode_kakan(pon) == Naki.kakan_of(Naki(pon)).naki_code, (pon, kakan))
        expect(int(n[1]["m"]) == meld_codes.encode_minkan(_ids(p, 0, "3m")[0], 3), n[1])
        expect([int(a["hai"]) for a in p.tags("DORA")[:2]] == p.indicators[1:3], (p.tags("DORA"), p.indicators))
    return Scenario("pon, daiminkan, kakan on the rinshan draw (second open kan)",
                    {0: "6s3m2468m248p9s123z", 1: "66s333m19p19s1234z"},
                    {0: ScriptPlayer(discards="6s3m"), 1: ScriptPlayer(discards="9p", take=["minkan", "kakan", "pon"])},
                    rinshan="6s", check=check)


def sc_kakan_chankan() -> Scenario:
    """As above, but seat 2 waits on 6s (45s) and rons the kakan tile: N AGARI with chankan (yaku 3)."""
    def check(p: Played):
        expect(p.letters.endswith("N T N A"), p.letters[-30:])
        a = p.tags("AGARI")[0]
        expect(a["who"] == "2" and a["fromWho"] == "1", a)
        expect(int(a["machi"]) == p.rinshan[0], (a, p.rinshan))
        got = dict(zip(ints(a["yaku"])[0::2], ints(a["yaku"])[1::2]))
        expect(got.get(3) == 1, got)
        expect("doraHaiUra" not in a and len(ints(a["doraHai"])) == 1, a)
        rec = [d for d in p.round.decisions if d["kind"] == "post" and d["chosen"] == POST_RON]
        expect(len(rec) == 1 and rec[0]["resolved"] == {"outcome": "executed", "event": render("AGARI", a)}, rec)
    return Scenario("kakan answered by chankan",
                    {0: "6s3m2468m248p9s123z", 1: "66s333m19p19s1234z", 2: "456m456p789p45s77z"},
                    {0: ScriptPlayer(discards="6s3m"), 1: ScriptPlayer(discards="9p", take=["minkan", "kakan", "pon"]),
                     2: ScriptPlayer(take=["ron"], ron_from=[1])},
                    rinshan="6s", check=check)


def sc_rinshan_tsumo() -> Scenario:
    """Seat 1 daiminkans 3m; the rinshan tile 5z completes 234p 567p 789s 5z-tanki: N T AGARI.
    Independent score: open hand, rinshan 1 han (the haku is the pair: no yakuhai); fu 20 + tsumo 2 + tanki 2 +
    haku pair 2 + open kan of simples 8 = 34 -> 40; non-dealer tsumo 1 han 40 fu = 400 / 700 (total 1500)."""
    def check(p: Played):
        expect(p.letters.endswith("D N T A"), p.letters[-30:])
        a = p.tags("AGARI")[0]
        expect(a["who"] == a["fromWho"] == "1", a)
        expect(a["ten"] == "40,1500,0", a["ten"])
        got = dict(zip(ints(a["yaku"])[0::2], ints(a["yaku"])[1::2]))
        expect(got == {4: 1}, got)
        expect(a["sc"] == "250,-7,250,15,250,-4,250,-4", a["sc"])
        expect(len(ints(a["doraHai"])) == 1 and len(p.tags("DORA")) == 0, a)
        expect(p.outcome.scores == [243, 265, 246, 246] and p.outcome.winners == [1], p.outcome)
    return Scenario("daiminkan, rinshan tsumo", {0: "3m2468m248p9s11z23z", 1: "333m234p567p789s5z"},
                    {0: ScriptPlayer(discards="3m"), 1: ScriptPlayer(take=["minkan", "tsumo"])}, rinshan="5z",
                    check=check)


def sc_ron_honba() -> Scenario:
    """Seat 1 rons the dealer's first discard 5m (tanyao, closed, 333s pon 4 fu, tanki 2 fu -> 40 fu, 1 han = 1300)
    with 2 honba and 1 stick: the discarder pays 1300 + 600, the winner also takes the 1000-point stick."""
    def check(p: Played):
        expect(p.letters == "I T D A", p.letters)
        a = p.tags("AGARI")[0]
        expect(a["ba"] == "2,1" and a["ten"] == "40,1300,0" and a["yaku"] == "8,1", a)
        expect(a["sc"] == "250,-19,250,29,240,0,250,0", a["sc"])
        expect(a["who"] == "1" and a["fromWho"] == "0" and int(a["machi"]) == _ids(p, 0, "5m")[0], a)
        expect(p.outcome.kyotaku == 0 and not p.outcome.renchan, p.outcome)
    return Scenario("ron with honba and a riichi stick", {0: "5m2468m248p9s11z23z", 1: "234m567p678s333s5m"},
                    {0: ScriptPlayer(discards="5m"), 1: ScriptPlayer(take=["ron"])},
                    spec=RoundSpec(0, 2, 1, (250, 250, 240, 250)), check=check)


def sc_double_ron() -> Scenario:
    """Seats 1 and 2 both ron the dealer's 5m (1300 each): two AGARI tags head-first; honba and the stick go
    to seat 1 only, the second tag has ba='1,0'."""
    def check(p: Played):
        expect(p.letters == "I T D A A", p.letters)
        a, b = p.tags("AGARI")
        expect(a["who"] == "1" and b["who"] == "2" and a["fromWho"] == b["fromWho"] == "0", (a, b))
        expect(a["ba"] == "1,1" and b["ba"] == "1,0", (a["ba"], b["ba"]))
        expect(a["sc"] == "250,-16,250,26,250,0,240,0", a["sc"])
        expect(b["sc"] == "234,-13,276,0,250,13,240,0", b["sc"])
        expect(p.outcome.winners == [1, 2] and p.outcome.scores == [221, 276, 263, 240], p.outcome)
    return Scenario("double ron", {0: "5m2468m248p9s11z23z", 1: "234m567p678s333s5m", 2: "234p567s678m444s5m"},
                    {0: ScriptPlayer(discards="5m"), 1: ScriptPlayer(take=["ron"]), 2: ScriptPlayer(take=["ron"])},
                    spec=RoundSpec(0, 1, 1, (250, 250, 250, 240)), check=check)


def sc_ron3() -> Scenario:
    def check(p: Played):
        expect(p.letters == "I T D Y", p.letters)
        y = p.tags("RYUUKYOKU")[0]
        expect(y.get("type") == "ron3" and y["ba"] == "0,0" and y["sc"] == "250,0,250,0,250,0,250,0", y)
        expect(set(y) == {"type", "ba", "sc", "hai1", "hai2", "hai3"}, sorted(y))
        expect(p.outcome.kind == "abort" and p.outcome.renchan, p.outcome)
    return Scenario("triple ron", {0: "5m2468m248p9s11z23z", 1: "234m567p678s333s5m", 2: "234p567s678m444s5m",
                                   3: "345p456s678m222s5m"},
                    {0: ScriptPlayer(discards="5m"), 1: ScriptPlayer(take=["ron"]), 2: ScriptPlayer(take=["ron"]),
                     3: ScriptPlayer(take=["ron"])}, check=check)


def sc_reach4() -> Scenario:
    """Four tenpai hands, each declares riichi on its first discard (a drawn honor): REACH 1/2 x 4, then reach4
    with all four hands shown, 4 sticks in ba, every score 1000 down."""
    def check(p: Played):
        expect(p.letters == "I T R D R T R D R T R D R T R D R Y", p.letters)
        y = p.tags("RYUUKYOKU")[0]
        expect(y.get("type") == "reach4" and y["ba"] == "0,4" and y["sc"] == "240,0,240,0,240,0,240,0", y)
        expect(all(f"hai{q}" in y for q in range(4)), sorted(y))
        reach2 = [a for a in p.tags("REACH") if a["step"] == "2"]
        want = ["240,250,250,250", "240,240,250,250", "240,240,240,250", "240,240,240,240"]
        expect([a["ten"] for a in reach2] == want, reach2)
        expect(p.outcome.kyotaku == 4 and p.outcome.renchan, p.outcome)
    tenpai = {0: "234m567p678s33s45p", 1: "345m678p789s44s56m", 2: "456m789p234s55s67p", 3: "567m234p345s66s78m"}
    return Scenario("four riichi", tenpai, {q: ScriptPlayer(take=["riichi"]) for q in range(4)},
                    draws={0: "5z", 1: "6z", 2: "7z", 3: "3z"}, check=check)


def sc_kaze4() -> Scenario:
    def check(p: Played):
        expect(p.letters == "I T D T D T D T D Y", p.letters)
        y = p.tags("RYUUKYOKU")[0]
        expect(y == {"type": "kaze4", "ba": "0,0", "sc": "250,0,250,0,250,0,250,0"}, y)
    return Scenario("four same winds", {q: "1z2468m2468p2468s" for q in range(4)},
                    {q: ScriptPlayer(discards="1z") for q in range(4)}, check=check)


def sc_kan4() -> Scenario:
    """Two ankan by the dealer, two by seat 1: the fourth kan's discard passes -> kan4 (two owners)."""
    def check(p: Played):
        expect(p.letters == "I T N O T N O T D T N O T N O T D Y", p.letters)
        y = p.tags("RYUUKYOKU")[0]
        expect(y == {"type": "kan4", "ba": "0,0", "sc": "250,0,250,0,250,0,250,0"}, y)
        expect([int(a["hai"]) for a in p.tags("DORA")] == p.indicators[1:5], (p.tags("DORA"), p.indicators))
    return Scenario("four kans by two players", {0: "1111m2222p678s7p9s", 1: "3333s4444s567m1z2z"},
                    {0: ScriptPlayer(take=["ankan"]), 1: ScriptPlayer(take=["ankan"])}, check=check)


def sc_yao9() -> Scenario:
    def check(p: Played):
        expect(p.letters == "I T Y", p.letters)
        y = p.tags("RYUUKYOKU")[0]
        expect(y.get("type") == "yao9" and y["ba"] == "0,0" and len(ints(y["hai0"])) == 14, y)
        expect(set(y) == {"type", "ba", "sc", "hai0"}, sorted(y))
        rec = p.round.decisions[0]
        expect(rec["chosen"] == DURING_KYUUSHU and rec["resolved"]["event"] == render("RYUUKYOKU", y), rec)
    return Scenario("nine terminals", {0: "19m19p19s123z2468m"}, {0: ScriptPlayer(take=["kyuushu"])},
                    draws={0: "4z"}, check=check)


def sc_exhaustive() -> Scenario:
    """Everyone tsumogiris to the wall's end: seats 0 and 2 tenpai, 1 and 3 noten (12 distinct yaochu + 1)."""
    def check(p: Played):
        expect(p.letters.endswith("T D Y") and p.letters.count("T") == LIVE_DRAWS, p.letters.count("T"))
        y = p.tags("RYUUKYOKU")[0]
        expect("type" not in y and y["ba"] == "0,0" and y["sc"] == "250,15,250,-15,250,15,250,-15", y)
        expect(set(y) == {"ba", "sc", "hai0", "hai2"}, sorted(y))
        expect(p.outcome.kind == "exhaustive" and p.outcome.dealer_tenpai and p.outcome.renchan, p.outcome)
    return Scenario("exhaustive draw", {0: "234m567p678s33s45p", 2: "456m789p234s55s67p",
                                        1: "19m19p19s1234z56z2m", 3: "19m19p19s12345z28p"}, check=check)


def sc_nagashi() -> Scenario:
    """Seat 1 draws (and tsumogiris) only terminals and honors: nagashi mangan, paid as a non-dealer mangan tsumo
    (dealer 4000, others 2000), no honba, the stick stays."""
    def check(p: Played):
        y = p.tags("RYUUKYOKU")[0]
        expect(y.get("type") == "nm" and y["ba"] == "1,1", y)
        expect(y["sc"] == "250,-40,240,80,250,-20,250,-20", y["sc"])
        expect(p.outcome.kind == "exhaustive" and p.outcome.nagashi and p.outcome.kyotaku == 1, p.outcome)
        expect(not p.outcome.renchan, p.outcome)
    yaochu = "19m19p19s1234z567z19m19p19s1z"  # 18 draws for seat 1 (positions 1, 5, ..., 69)
    draws = {1 + 4 * i: k for i, k in enumerate(kinds(yaochu)[:18])}
    return Scenario("nagashi mangan", {0: "2468m2468p2468s1z", 2: "2468m2468p2468s2z", 3: "2468m2468p2468s3z",
                                       1: "2345678m234567p"}, draws=draws,
                    spec=RoundSpec(0, 1, 1, (250, 240, 250, 250)), check=check)


def sc_pao() -> Scenario:
    """Seat 1 pons haku, hatsu and chun from the dealer's discards, then tsumos 4m: daisangen with pao on the
    dealer, who pays all 32000; honba 0."""
    def check(p: Played):
        a = p.tags("AGARI")[0]
        expect(a.get("yakuman") == "39" and a.get("paoWho") == "0", a)
        expect(a["who"] == a["fromWho"] == "1", a)
        expect(a["ten"].split(",")[1:] == ["32000", "5"], a["ten"])
        expect(a["sc"] == "250,-320,250,320,250,0,250,0", a["sc"])
        expect(len(p.tags("N")) == 3, p.tags("N"))
    return Scenario("pao: daisangen tsumo", {0: "567z2468m248p9s12z", 1: "55z66z77z55s23m19p1s"},
                    {0: ScriptPlayer(discards="5z6z7z"), 1: ScriptPlayer(discards="1p9p1s", take=["pon", "tsumo"])},
                    draws={10: "4m"}, rinshan="5z6z7z", check=check)


def sc_pao_compound() -> Scenario:
    """As sc_pao, but the concealed part is 11z22z: the tsumo 1z makes daisangen + tsuuiisou (two yakuman,
    64,000) with 2 honba. Tenhou's pao covers the WHOLE hand and the honba (real log 2019103023gm-00a9-0000-f90e0fe9,
    3 honba: sc 204,649,314,0,98,-649,384,0): the liable dealer alone pays 640 + 6."""
    def check(p: Played):
        a = p.tags("AGARI")[0]
        expect(a.get("yakuman") == "39,42" and a.get("paoWho") == "0" and a["ba"] == "2,0", a)
        expect(a["who"] == a["fromWho"] == "1" and a["ten"].split(",")[1:] == ["64000", "5"], a)
        expect(a["sc"] == "250,-646,250,646,250,0,250,0", a["sc"])
    return Scenario("pao: compound yakuman tsumo (daisangen + tsuuiisou), 2 honba",
                    {0: "567z2468m248p9s12z", 1: "55z66z77z11z22z19p1s", 2: "2z2468m2468p2468s"},
                    {0: ScriptPlayer(discards="5z6z7z"), 1: ScriptPlayer(discards="1p9p1s", take=["pon", "tsumo"])},
                    draws={10: "1z"}, rinshan="5z6z7z", spec=RoundSpec(0, 2, 0, (250,) * 4), check=check)


def sc_pao_compound_ron_third() -> Scenario:
    """The same hand rons seat 2's 1z (seat 0 is liable): the liable seat and the discarder pay half of the whole
    64,000 each, the liable seat adds the 2 honba (real logs 2020040823gm-00a9-0000-44be7808: ba 2,0 48000 ->
    liable -246, discarder -240; 2020110507gm-00a9-0000-990e14a5: ba 1,2 -> liable -163, discarder -160)."""
    def check(p: Played):
        a = p.tags("AGARI")[0]
        expect(a.get("yakuman") == "39,42" and a.get("paoWho") == "0" and a["ba"] == "2,1", a)
        expect(a["who"] == "1" and a["fromWho"] == "2" and a["ten"].split(",")[1:] == ["64000", "5"], a)
        expect(a["sc"] == "250,-326,240,656,250,-320,250,0", a["sc"])
    return Scenario("pao: compound yakuman ron by a third seat, 2 honba + 1 stick",
                    {0: "567z2468m248p9s12z", 1: "55z66z77z11z22z19p1s", 2: "1z99m99p99s2468m24p"},
                    {0: ScriptPlayer(discards="5z6z7z"), 1: ScriptPlayer(discards="1p9p1s", take=["pon", "ron"]),
                     2: ScriptPlayer(discards="9m9m1z")},
                    rinshan="5z6z7z", spec=RoundSpec(0, 2, 1, (250, 240, 250, 250)), check=check)


def sc_riichi_yakuman() -> Scenario:
    """Seat 1 (111m 999p 111s 999s + 5z) declares riichi on its first draw 6z and tsumos 5z: suuankou tanki. A
    yakuman's yaku names carry no Riichi, yet the winner IS in riichi, so the tag must list doraHaiUra (the real
    riichi suuankou tag of 2021010111gm-00a9-0000-5390436b: ... yakuman="41" doraHai="61,119" doraHaiUra="35,107")."""
    def check(p: Played):
        expect(p.letters == "I T D T R D R T D T D T D T A", p.letters)
        a = p.tags("AGARI")[0]
        expect(a.get("yakuman") == "41" and a["who"] == a["fromWho"] == "1", a)
        expect(a["ten"].split(",")[1:] == ["32000", "5"], a["ten"])
        expect(ints(a["doraHaiUra"]) == p.round.wall.ura_indicators(1) and len(ints(a["doraHai"])) == 1, a)
        expect(set(a) == {"ba", "hai", "machi", "ten", "yakuman", "doraHai", "doraHaiUra", "who", "fromWho", "sc"},
               sorted(a))
        expect(a["sc"] == "250,-160,240,330,250,-80,250,-80", a["sc"])
    return Scenario("riichi yakuman keeps its ura indicators (suuankou tanki)",
                    {0: "2468m2468p2468s1z", 1: "111m999p111s999s5z", 2: "2468m2468p2468s2z", 3: "2468m2468p2468s3z"},
                    {1: ScriptPlayer(take=["tsumo", "riichi"])}, draws={1: "6z", 5: "5z"}, check=check)


SCENARIOS: List[Callable[[], Scenario]] = [sc_ankan_chained, sc_ankan_after_open_kan, sc_pon_minkan_kakan,
                                           sc_kakan_chankan, sc_rinshan_tsumo, sc_ron_honba, sc_double_ron, sc_ron3,
                                           sc_reach4, sc_kaze4, sc_kan4, sc_yao9, sc_exhaustive, sc_nagashi, sc_pao,
                                           sc_pao_compound, sc_pao_compound_ron_third, sc_riichi_yakuman]


def check_scenario(sc: Scenario, played: Played) -> collections.Counter:
    """The scenario's own expectations, then every generic check on its round, the sidecar included."""
    sc.check(played)
    counts = check_rounds(played.xml, sc.name, complete=False) + check_agari_tags(played.xml, sc.name)
    counts += check_kan_emission(played.xml, sc.name)
    counts += check_sidecar(played.xml, played.round.decisions, where=sc.name)
    return counts


def run_scenarios(names: Optional[Sequence[str]] = None) -> List[Played]:
    out = []
    for make in SCENARIOS:
        sc = make()
        if names and sc.name not in names:
            continue
        t0 = time.perf_counter()
        played = play_scenario(sc)
        check_scenario(sc, played)
        print(f"scenario OK: {sc.name:70s} {played.letters.count(' ') + 1:3d} events {time.perf_counter() - t0:5.1f} s")
        out.append(played)
    return out


# ---------------------------------------------------------------------------------------------- pure tables
def check_tables() -> None:
    """Settlement [C7][C9] and transition [C8] rows with hand-computed expectations."""
    ron13 = scoring.HandScore(1, 40, 1300, 0, ((8, 1),), (), 0, 0)
    expect(scoring.agari_deltas(ron13, 1, 0, 0, 2, 1) == [-19, 29, 0, 0], "ron 1300 + 2 honba + 1 stick")
    mangan_ko = scoring.HandScore(5, 30, 8000, 1, (), (), 40, 20)
    expect(scoring.agari_deltas(mangan_ko, 1, 1, 0, 1, 0) == [-41, 83, -21, -21], "non-dealer mangan tsumo, 1 honba")
    mangan_oya = scoring.HandScore(5, 30, 12000, 1, (), (), 0, 40)
    expect(scoring.agari_deltas(mangan_oya, 0, 0, 0, 0, 2) == [140, -40, -40, -40], "dealer mangan tsumo, 2 sticks")
    ykm = scoring.HandScore(13, 40, 32000, 5, (), (39,), 160, 80)
    expect(scoring.agari_deltas(ykm, 1, 1, 0, 1, 0, {39: 2}) == [0, 323, -323, 0], "pao tsumo: liable pays all + honba")
    expect(scoring.agari_deltas(ykm, 1, 2, 0, 0, 0, {39: 2}) == [0, 320, -320, 0], "pao ron by the liable seat")
    expect(scoring.agari_deltas(ykm, 1, 3, 0, 1, 0, {39: 2}) == [0, 323, -163, -160],
           "pao ron by a third seat: halves, the honba on the liable seat (2020110507gm-00a9-0000-990e14a5)")
    expect(scoring.agari_deltas(ykm, 1, 3, 0, 0, 0) == [0, 320, 0, -320], "yakuman ron without pao")
    ykm2 = scoring.HandScore(26, 50, 64000, 5, (), (39, 42), 0, 320)
    expect(scoring.agari_deltas(ykm2, 0, 0, 0, 3, 0, {39: 2}) == [649, 0, -649, 0],
           "compound yakuman pao tsumo: the whole hand + honba (2019103023gm-00a9-0000-f90e0fe9)")
    ykm2k = scoring.HandScore(26, 50, 64000, 5, (), (39, 42), 320, 160)
    expect(scoring.agari_deltas(ykm2k, 1, 3, 0, 2, 0, {39: 2}) == [0, 646, -326, -320],
           "compound yakuman pao ron by a third seat: half of the whole hand each, honba on the liable seat")
    expect(scoring.ryuukyoku_deltas([True, False, False, False]) == [30, -10, -10, -10], "1 tenpai")
    expect(scoring.ryuukyoku_deltas([True, True, False, False]) == [15, 15, -15, -15], "2 tenpai")
    expect(scoring.ryuukyoku_deltas([True, True, True, False]) == [10, 10, 10, -30], "3 tenpai")
    expect(scoring.ryuukyoku_deltas([True] * 4) == [0] * 4 == scoring.ryuukyoku_deltas([False] * 4), "0 / 4 tenpai")
    expect(scoring.nagashi_deltas([1], 0) == [-40, 80, -20, -20], "nagashi non-dealer")
    expect(scoring.nagashi_deltas([0], 0) == [120, -40, -40, -40], "nagashi dealer")
    expect(scoring.nagashi_deltas([0, 1], 0) == [80, 40, -60, -60], "two nagashi")
    expect(scoring.owari_points([413, 336, 280, -29]) == [51.0, 14.0, -12.0, -53.0], "owari with tobi")
    expect(scoring.owari_points([250, 250, 250, 250]) == [35.0, 5.0, -15.0, -25.0], "owari ties: seat order")
    expect(scoring.owari_points([305, 295, 205, 195]) == [39.0, 10.0, -19.0, -30.0], "owari rounding half away")
    expect(scoring.owari_points([265, 448, -8, 295]) == [-13.0, 54.0, -51.0, 10.0],
           "owari of 2021081109gm-00a9-0000-fc96a2e9")
    expect(scoring.owari_points([-5, 423, 269, 313], exact=True) == [-50.5, 52.3, -13.1, 11.3],
           "owari from 2022-01-22 keeps 0.1 precision (2022051222gm-00a9-0000-b02c0edb)")
    expect(final_scores([300, 300, 200, 180], 2) == [320, 300, 200, 180], "unclaimed sticks to first")

    def o(kind, scores, **kw):
        return RoundOutcome(kind, list(scores), kw.pop("kyotaku", 0), **kw)
    e1 = RoundSpec(0, 0, 0, (250,) * 4)
    expect(next_round(e1, o("agari", [330, 210, 230, 230], dealer_won=True)) == RoundSpec(0, 1, 0, (330, 210, 230, 230)),
           "dealer win: renchan, honba + 1")
    expect(next_round(RoundSpec(0, 2, 1, (250,) * 4), o("agari", [240, 280, 240, 240]))
           == RoundSpec(1, 0, 0, (240, 280, 240, 240)), "non-dealer win: pass, honba 0, sticks taken")
    expect(next_round(e1, o("exhaustive", [240, 240, 260, 260], kyotaku=1)) == RoundSpec(1, 1, 1, (240, 240, 260, 260)),
           "draw, dealer noten: pass, honba + 1, sticks stay")
    expect(next_round(e1, o("exhaustive", [260, 240, 260, 240], dealer_tenpai=True))
           == RoundSpec(0, 1, 0, (260, 240, 260, 240)), "draw, dealer tenpai: renchan")
    expect(next_round(e1, o("abort", [250] * 4, abort_type="kaze4")) == RoundSpec(0, 1, 0, (250,) * 4), "abort: renchan")
    expect(next_round(e1, o("agari", [600, 420, -20, 0])) is None, "tobi below zero ends")
    expect(next_round(e1, o("agari", [600, 400, 0, 0])) == RoundSpec(1, 0, 0, (600, 400, 0, 0)), "zero continues")
    s4 = RoundSpec(7, 0, 0, (250,) * 4)
    expect(next_round(s4, o("agari", [240, 310, 230, 220])) is None, "S4 pass with a 30,000 leader ends")
    expect(next_round(s4, o("agari", [240, 290, 240, 230])) == RoundSpec(8, 0, 0, (240, 290, 240, 230)), "S4 -> W1")
    expect(next_round(s4, o("agari", [240, 240, 240, 280], dealer_won=True)) == RoundSpec(7, 1, 0, (240, 240, 240, 280)),
           "S4 dealer win, not first: renchan")
    expect(next_round(s4, o("agari", [240, 240, 210, 310], dealer_won=True)) is None, "S4 dealer win as leader: agari-yame")
    expect(next_round(s4, o("exhaustive", [240, 240, 210, 310], dealer_tenpai=True)) is None, "S4 tenpai-yame")
    expect(next_round(s4, o("exhaustive", [240, 240, 210, 310], dealer_tenpai=False)) is None, "S4 pass, leader 31,000")
    expect(next_round(s4, o("abort", [240, 240, 210, 310], abort_type="ron3")) == RoundSpec(7, 1, 0, (240, 240, 210, 310)),
           "S4 abort never ends")
    w1 = RoundSpec(8, 0, 0, (290, 240, 240, 230))
    expect(next_round(w1, o("agari", [320, 240, 240, 200], dealer_won=True)) is None, "West renchan with 30,000: ends")
    expect(next_round(w1, o("agari", [260, 240, 270, 230], dealer_won=True)) == RoundSpec(8, 1, 0, (260, 240, 270, 230)),
           "West renchan under 30,000 continues")
    west_tenpai = o("exhaustive", [243, 183, 305, 259], kyotaku=1, dealer_tenpai=True)
    expect(next_round(RoundSpec(8, 1, 0, (243, 173, 295, 289)), west_tenpai)
           == RoundSpec(8, 2, 1, (243, 183, 305, 259)), "West dealer tenpai with another seat at 30,000 continues "
           "(2021081622gm-00a9-0000-17dc8b4d)")
    expect(next_round(w1, o("abort", [320, 240, 240, 200], abort_type="kan4")) == RoundSpec(8, 1, 0, (320, 240, 240, 200)),
           "West abort never ends")
    expect(next_round(RoundSpec(11, 0, 0, (290,) * 2 + (210,) * 2), o("agari", [290, 295, 210, 205])) is None, "W4 ends")
    expect(next_round(RoundSpec(10, 0, 0, (290,) * 2 + (210,) * 2), o("agari", [290, 295, 210, 205]))
           == RoundSpec(11, 0, 0, (290, 295, 210, 205)), "W3 -> W4 under 30,000")
    codes = list(meld_codes.every_4p_code())
    expect(len(codes) == len(set(codes)) == 7024, f"{len(codes)} 4p meld codes (expected 7024)")
    bad = [c for c in codes if meld_codes.encode_meld(meld_codes.decode_meld(c)) != meld_codes.canonical(c)]
    expect(not bad, f"encode(decode(c)) != canonical(c) for {len(bad)} codes, e.g. {bad[:5]}")
    non_ankan = [c for c in codes if meld_codes.decode_meld(c).kind != "ankan"]
    bad = [c for c in non_ankan if meld_codes.canonical(c) != c]
    expect(not bad, f"{len(bad)} chi / pon / kakan / daiminkan codes do not round-trip, e.g. {bad[:5]}")
    ankan = sorted({meld_codes.canonical(c) for c in codes if meld_codes.decode_meld(c).kind == "ankan"})
    expect(ankan == [meld_codes.encode_ankan(k) for k in range(34)], "ankan canonical codes are not one per kind")
    code_set = set(codes)
    aliases = 0
    for m in meld_codes.every_alias():
        c = meld_codes.canonical(m)
        expect(c in code_set and meld_codes.decode_meld(c)[:5] == meld_codes.decode_meld(m)[:5],
               f"alias {m} canonicalises to {c}, outside the 7024 codes or a different meld")
        aliases += 1
    print(f"tables OK: settlement 17 rows, transition 22 rows, owari 5, meld codes {len(codes)} round-trip, "
          f"{aliases} aliases canonicalise into them")


# ---------------------------------------------------------------------------------------------- real tags
def gate_logs() -> List[str]:
    return sorted(glob.glob(os.path.join(GATE_GAMES, "**", "*.log"), recursive=True))


def parity_logs(n: int) -> List[str]:
    """The first n gate logs the parser accepts without the admission filter (the fixtures also hold the malformed
    logs the parser must reject, which no prefix can parse)."""
    out = []
    for f in gate_logs():
        try:
            process_xml.process(f, admission=False)
        except process_xml.InvalidGameException:
            continue
        out.append(f)
        if len(out) == n:
            break
    return out


def check_real_tags(files: Sequence[str], n_agari: int = 30, n_ryuukyoku: int = 10) -> collections.Counter:
    """The first n_agari AGARI and n_ryuukyoku RYUUKYOKU tags of real logs, reproduced by scoring.py."""
    counts: collections.Counter = collections.Counter()
    seen_a = seen_r = 0
    for f in files:
        xml = open(f, encoding="utf-8", errors="ignore").read()
        for r, events in enumerate(rounds_of(events_of(xml))):
            spec = init_spec(events[0][1])
            for i, (tag, attrs) in enumerate(events):
                where = f"{os.path.basename(f)} round {r} event {i}"
                if tag == "AGARI" and seen_a < n_agari:
                    first = events[i - 1][0] != "AGARI"
                    counts["AGARI " + check_agari_tag(attrs, spec.index, spec.oya, where, first)] += 1
                    seen_a += 1
                elif tag == "RYUUKYOKU" and seen_r < n_ryuukyoku:
                    counts["RYUUKYOKU " + check_ryuukyoku_tag(events, attrs, spec.oya, where)] += 1
                    seen_r += 1
        if seen_a >= n_agari and seen_r >= n_ryuukyoku:
            break
    expect(seen_a == n_agari and seen_r == n_ryuukyoku, f"only {seen_a} AGARI / {seen_r} RYUUKYOKU found")
    return counts


# ---------------------------------------------------------------------------------------------- real fixtures
def fixture_xml(name: str) -> str:
    return open(os.path.join(FIXTURES, name + ".log"), encoding="utf-8").read()


def check_real_fixtures() -> collections.Counter:
    """The real logs under tests/data/engine_fixtures, expectations from their own tags (never from the engine):
    S1 West dealer continuation, S2 pao over the whole hand (compound tsumo + honba, ron by a third seat + honba
    / sticks), S3 owari, S5 the riichi-yakuman tag shape. Every full log also passes check_rounds (every
    transition, every running score, the owari) and check_agari_tags (every AGARI / RYUUKYOKU)."""
    counts: collections.Counter = collections.Counter()
    xml = fixture_xml("2021081622gm-00a9-0000-17dc8b4d")
    rounds = rounds_of(events_of(xml))
    spec = RoundSpec(8, 1, 0, (243, 173, 295, 289))
    r = next(i for i, events in enumerate(rounds) if init_spec(events[0][1]) == spec)
    out, _ = outcome_of(rounds[r], spec)
    expect(out.kind == "exhaustive" and out.dealer_tenpai and out.scores == [243, 183, 305, 259] and out.kyotaku == 1,
           (spec, out))
    expect(next_round(spec, out) == init_spec(rounds[r + 1][0][1]) == RoundSpec(8, 2, 1, (243, 183, 305, 259)),
           f"S1: West 1 must continue after dealer tenpai with seat 2 at 30,500: {next_round(spec, out)}")
    counts["S1 West continuation after dealer tenpai with a 30,000 non-dealer"] += 1
    pao = {"2019103023gm-00a9-0000-f90e0fe9": ([649, 0, -649, 0], "compound tsumo, 3 honba"),
           "2020040823gm-00a9-0000-44be7808": ([-240, -246, 0, 486], "ron by a third seat, 2 honba, dealer winner"),
           "2020110507gm-00a9-0000-990e14a5": ([343, -163, -160, 0], "ron by a third seat, 1 honba, 2 sticks"),
           "2021012523gm-00a9-0000-0da1f584": ([-160, 0, 330, -160], "ron by a third seat, 1 stick")}
    for name, (want, label) in pao.items():
        xml = fixture_xml(name)
        for r, events in enumerate(rounds_of(events_of(xml))):
            spec = init_spec(events[0][1])
            for tag, attrs in events:
                if tag == "AGARI" and "paoWho" in attrs:
                    _, deltas = recompute_agari(attrs, spec.index, spec.oya)
                    expect(ints(attrs["sc"])[1::2] == want, f"{name}: the log's pao sc {attrs['sc']} is not {want}")
                    expect(deltas == want, f"S2 {name}: scoring.agari_deltas {deltas} != the log's {want} ({attrs})")
                    counts[f"S2 pao {label}: sc reproduced"] += 1
    expect(scoring.owari_points([265, 448, -8, 295]) == [-13.0, 54.0, -51.0, 10.0], "S3 owari formula")
    xml = fixture_xml("2021081109gm-00a9-0000-fc96a2e9")
    expect('owari="265,-13.0,448,54.0,-8,-51.0,295,10.0"' in xml, "S3: the fixture's owari")
    counts["S3 owari of 2021081109gm-00a9-0000-fc96a2e9 reproduced"] += 1
    for f in sorted(glob.glob(os.path.join(FIXTURES, "*.log"))):
        xml = open(f, encoding="utf-8").read()
        name = os.path.basename(f)
        check_rounds(xml, name)
        counts["real fixture: every transition, running score and owari reproduced"] += 1
        tags = check_agari_tags(xml, name)
        counts["real fixture AGARI / RYUUKYOKU tags reproduced"] += sum(tags.values())
    riichi_ykm = [a for f in gate_logs() for _, a in events_of(open(f, encoding="utf-8").read())
                  if "yakuman" in a and "doraHaiUra" in a]
    expect(riichi_ykm, "no real riichi yakuman tag among the gate logs")
    for a in riichi_ykm:
        expect(set(a) - {"m", "owari", "paoWho"}
               == {"ba", "hai", "machi", "ten", "yakuman", "doraHai", "doraHaiUra", "who", "fromWho", "sc"}, sorted(a))
    counts["S5 real riichi yakuman tags listing doraHaiUra"] += len(riichi_ykm)
    return counts


def check_corpus_log(xml: str, where: str = "") -> collections.Counter:
    """One real log against the rule oracles: next_round vs the log's next INIT (or its end), every exhaustive
    draw's tenpai seats / nagashi type / sc from the reconstruction, the owari. Raises on the first disagreement."""
    counts: collections.Counter = collections.Counter()
    rounds = rounds_of(events_of(xml))
    for r, events in enumerate(rounds):
        spec = init_spec(events[0][1])
        out, end = outcome_of(events, spec)
        last = r == len(rounds) - 1
        nxt = next_round(spec, out)
        log_next = None if last else init_spec(rounds[r + 1][0][1])
        if nxt != log_next:
            fail(f"{where} round {r}: next_round says {nxt}, the log says {log_next} ({spec} -> {out})")
        counts[f"transition: {out.kind}{' renchan' if out.renchan else ''}{' West' if spec.index >= 8 else ''}"] += 1
        for i, (tag, attrs) in enumerate(events):
            if tag == "RYUUKYOKU":
                counts["RYUUKYOKU " + check_ryuukyoku_tag(events, attrs, spec.oya, f"{where} round {r} event {i}")] += 1
        if last:
            final = final_scores(end["scores"], end["kyotaku"])
            exact = owari_exact(where)
            if end["owari"] != scoring.owari_attr(final, exact):
                fail(f"{where}: owari {end['owari']} != {scoring.owari_attr(final, exact)}")
            counts["owari"] += 1
    counts["games"] += 1
    return counts


GO_HANCHAN, GO_SANMA = 0x08, 0x10  # GO type bits: 169 = 4p red kuitan hanchan; 0xe1 is a tonpuu (East-only) game


def check_corpus(files: Sequence[str]) -> collections.Counter:
    """Every complete (owari, no BYE) 4-player hanchan log in `files` through check_corpus_log (tonpuu / sanma GO
    types are skipped: their round table differs); the counts name every disagreement class with its first example,
    and `disagreements` sums them."""
    counts: collections.Counter = collections.Counter()
    examples: Dict[str, str] = {}
    for f in files:
        xml = open(f, encoding="utf-8", errors="ignore").read()
        if "owari=" not in xml or "<BYE" in xml:
            counts["skipped: no owari / BYE"] += 1
            continue
        go = int(re.search(r'<GO type="(\d+)"', xml).group(1))
        if not go & GO_HANCHAN or go & GO_SANMA:
            counts[f"skipped: GO type {go} is not a 4-player hanchan"] += 1
            continue
        try:
            counts += check_corpus_log(xml, os.path.basename(f))
        except EngineCheckError as e:
            key = str(e).split(":")[1].strip()[:30] if ":" in str(e) else str(e)[:30]
            counts[f"DISAGREEMENT {key}"] += 1
            counts["disagreements"] += 1
            examples.setdefault(key, str(e)[:300])
    for key, ex in examples.items():
        print(f"  first disagreement ({key}): {ex}")
    return counts


# ---------------------------------------------------------------------------------------------- corruptions
def must_raise(name: str, fn: Callable[[], object], match: str) -> None:
    """fn must raise EngineCheckError, and from the intended check: `match` is in the message."""
    try:
        fn()
    except EngineCheckError as e:
        if match not in str(e):
            fail(f"corruption {name!r} raised from another check: wanted {match!r}, got {str(e)[:200]}")
        print(f"corruption raises OK: {name:55s} -> {str(e)[:80]}")
        return
    fail(f"corruption NOT caught: {name}")


def _normalised(d: dict) -> dict:
    """The record with its candidates spread uniformly over its offer: the probability-sum check stays quiet so a
    corruption of the offer / chosen / resolution trips its own check."""
    d["candidates"] = {str(a): 1.0 / len(d["offer"]) for a in d["offer"]}
    return d


def check_corruptions(played: Sequence[Played], game_xml: str, game_decisions: List[dict]) -> None:
    by_name = {p.scenario.name: p for p in played}
    xml = by_name["ankan, chained ankan"].xml
    must_raise("ankan written as N T DORA", lambda: check_kan_emission(
        re.sub(r"(<N [^>]*/>)(<DORA [^>]*/>)(<T\d+/>)", r"\1\3\2", xml, count=1)), "after a kan")
    kakan = by_name["pon, daiminkan, kakan on the rinshan draw (second open kan)"].xml
    must_raise("open kan's DORA before the discard dropped", lambda: check_kan_emission(
        re.sub(r"(<[TUVW]\d+/>)(<DORA [^>]*/>)(<[DEFG]\d+/>)", r"\1\3", kakan, count=1)), "indicator still unwritten")
    must_raise("AGARI doraHai with an unflipped indicator", lambda: check_kan_emission(
        by_name["daiminkan, rinshan tsumo"].xml.replace('doraHai="', 'doraHai="0,')), "doraHai lists")
    must_raise("INIT ten not summing to 100,000", lambda: check_rounds(
        game_xml.replace('ten="250,250,250,250"', 'ten="250,250,250,240"', 1)), "INIT ten")
    must_raise("second INIT honba off by one", lambda: check_rounds(_bump_second_init(game_xml)), "transition table")
    must_raise("owari altered", lambda: check_rounds(re.sub(r'owari="(\d+),', r'owari="\g<1>0,', game_xml)), "owari")
    must_raise("REACH ten altered", lambda: check_rounds(
        re.sub(r'(<REACH who="\d" ten=")(\d+)', lambda m: m.group(1) + str(int(m.group(2)) + 1), game_xml, count=1)),
        "REACH ten")
    west = fixture_xml("2021081622gm-00a9-0000-17dc8b4d")
    must_raise("West 1 continuation cut after dealer tenpai", lambda: check_rounds(
        west[:west.index('<INIT seed="8,2,1')] + "</mjloggm>"), "table says continue")
    ron = by_name["ron with honba and a riichi stick"].xml
    must_raise("AGARI ten points altered", lambda: check_agari_tags(ron.replace("40,1300,0", "40,2000,0")), "ten")
    must_raise("AGARI sc delta altered", lambda: check_agari_tags(ron.replace("250,-19,250,29", "250,-16,250,26")),
               "sc deltas")
    must_raise("AGARI yaku altered", lambda: check_agari_tags(ron.replace('yaku="8,1"', 'yaku="7,1"')), "yaku")
    compound = by_name["pao: compound yakuman tsumo (daisangen + tsuuiisou), 2 honba"].xml
    must_raise("compound pao charged for one yakuman only", lambda: check_agari_tags(
        compound.replace("250,-646,250,646,250,0,250,0", "250,-326,250,646,250,-160,250,-160")), "sc deltas")
    must_raise("riichi yakuman without doraHaiUra", lambda: check_agari_tags(
        re.sub(r' doraHaiUra="[^"]*"', "", by_name["riichi yakuman keeps its ura indicators (suuankou tanki)"].xml)),
        "ura")
    draw = by_name["exhaustive draw"].xml
    must_raise("RYUUKYOKU noten bappu altered", lambda: check_agari_tags(draw.replace("250,15,250,-15", "250,30,250,-30")),
               "RYUUKYOKU deltas")
    must_raise("RYUUKYOKU relabelled nagashi mangan", lambda: check_agari_tags(
        draw.replace('<RYUUKYOKU ba="0,0" sc="250,15,250,-15,250,15,250,-15"',
                     '<RYUUKYOKU type="nm" ba="0,0" sc="250,-40,250,80,250,-20,250,-20"')), "nagashi")
    must_raise("RYUUKYOKU tenpai seat dropped", lambda: check_agari_tags(
        re.sub(r'(<RYUUKYOKU [^>]*?) hai2="[^"]*"', r"\1", draw)), "tenpai seats")  # the tag's key, not INIT's
    must_raise("RYUUKYOKU tenpai hand altered", lambda: check_agari_tags(
        re.sub(r'(<RYUUKYOKU [^>]*? hai0=")(\d+),', r"\1\2,\2,", draw)), "rebuilt hand")
    nagashi = by_name["nagashi mangan"].xml
    must_raise("nagashi mangan paid as noten bappu", lambda: check_agari_tags(
        nagashi.replace(' type="nm"', "").replace("250,-40,240,80,250,-20,250,-20", "250,0,240,0,250,0,250,0")),
        "nagashi")
    bad = json.loads(json.dumps(game_decisions))
    _normalised(bad[0])["chosen"] = next(a for a in bad[0]["offer"] if a != bad[0]["chosen"])
    must_raise("sidecar chosen flipped", lambda: check_sidecar(game_xml, bad), "executed action")
    bad = json.loads(json.dumps(game_decisions))
    bad[0]["offer"] = [a for a in bad[0]["offer"] if a != bad[0]["chosen"]][:-1] + [bad[0]["chosen"]]
    _normalised(bad[0])
    must_raise("sidecar offer missing an action", lambda: check_sidecar(game_xml, bad), "legal mask")
    bad = json.loads(json.dumps(game_decisions))
    post = next(d for d in bad if d["kind"] == "post" and d["resolved"]["outcome"] == "passed")
    post["resolved"] = {"outcome": "executed", "event": None}
    must_raise("sidecar pass relabelled executed", lambda: check_sidecar(game_xml, bad), "executed without an event")
    bad = json.loads(json.dumps(game_decisions))
    ron = next(d for d in bad if d["kind"] == "post" and d["chosen"] == POST_RON and d["resolved"]["outcome"] == "executed")
    _normalised(ron).update(chosen=POST_PASS, resolved={"outcome": "passed", "event": None})
    must_raise("sidecar executed ron relabelled a pass", lambda: check_sidecar(game_xml, bad), "executed action")
    bad = json.loads(json.dumps(game_decisions))
    call = next(d for d in bad if d["kind"] == "post" and d["chosen"] >= 2 and d["resolved"]["outcome"] == "executed"
                and any(a >= 2 and a != d["chosen"] for a in d["offer"]))
    other = next(a for a in call["offer"] if a >= 2 and a != call["chosen"])
    same_class = ("chi" if call["chosen"] < PON_IDX else "pon") == ("chi" if other < PON_IDX else "pon")
    _normalised(call)["chosen"] = other
    must_raise(f"sidecar executed call changed to another offered {'call of the same class' if same_class else 'call'}",
               lambda: check_sidecar(game_xml, bad), "executed action")
    must_raise("sidecar empty", lambda: check_sidecar(game_xml, []), "decision coverage")
    bad = json.loads(json.dumps(game_decisions))
    del bad[len(bad) // 2]
    for n, d in enumerate(bad):
        d["k"] = n
    must_raise("sidecar one record dropped and renumbered", lambda: check_sidecar(game_xml, bad), "decision coverage")
    bad = json.loads(json.dumps(game_decisions))
    bad.append(dict(bad[-1]))
    bad[-1]["k"] = len(bad) - 1
    must_raise("sidecar one record duplicated", lambda: check_sidecar(game_xml, bad), "decision coverage")
    must_raise("replay against an altered log", lambda: check_replay(
        re.sub(r"<T(\d+)/>", r"<U\1/>", game_xml, count=1), game_decisions, 0,
        [UniformPlayer(), UniformPlayer(), GreedyPlayer(), GreedyPlayer()], ["uniform", "uniform", "greedy", "greedy"]),
        "replay diverges")

    def dropping_parser(gameinfo, rounds, pending, admission):
        H, T, due = process_events(gameinfo, rounds, pending=pending, admission=admission)
        for h in due:
            if isinstance(h, DuringTurn) and h.decisions[DECISION_REACH_IDX]:
                h.decisions[DECISION_REACH_IDX] = []
                return H, T, due
        return H, T, due
    must_raise("prefix parser dropping a riichi offer", lambda: check_prefix_parity(parity_logs(1)[0], dropping_parser),
               "legal mask differs")


def _bump_second_init(xml: str) -> str:
    inits = list(re.finditer(r'<INIT seed="(\d+),(\d+),', xml))
    m = inits[1]
    return xml[:m.start(2)] + str(int(m.group(2)) + 1) + xml[m.end(2):]


# ---------------------------------------------------------------------------------------------- generated games
def players_of(meta: dict) -> list:
    """The four players a generated game's meta.json describes, rebuilt exactly (players.from_description)."""
    expect("seats" in meta, "meta.json without `seats`: the per-seat player specification is missing")
    return [from_description(d) for d in meta["seats"]]


def check_game_dir(game_dir: str, replay: bool = True) -> collections.Counter:
    """Every check on one saved game; the players are rebuilt from meta.json `seats` for the distribution parity
    of every model seat (its own checkpoint) and, with replay=True, for the byte-identical replay."""
    xml = open(os.path.join(game_dir, "game.log"), encoding="utf-8").read()
    decisions = json.load(open(os.path.join(game_dir, "decisions.json")))
    meta = json.load(open(os.path.join(game_dir, "meta.json")))
    where = os.path.basename(game_dir)
    players = players_of(meta)
    expect([p.label for p in players] == meta["players"], f"{where}: rebuilt labels {[p.label for p in players]} != "
           f"meta players {meta['players']}")
    models = {q: (p.model, p.temperature) for q, p in enumerate(players) if isinstance(p, ModelPlayer)}
    counts = check_rounds(xml, where) + check_agari_tags(xml, where) + check_kan_emission(xml, where)
    counts += check_sidecar(xml, decisions, models, where)
    H, _ = parse_xml_string(xml)
    expect(len(H) == meta["rounds"], f"{where}: parsed {len(H)} rounds, meta says {meta['rounds']}")
    expect(len(decisions) == meta["n_decisions"], f"{where}: {len(decisions)} decisions, meta {meta['n_decisions']}")
    for q, p in enumerate(players):
        counts[f"seat played by {meta['seats'][q]['kind']}"] += 1
    if replay:
        check_replay(xml, decisions, meta["seed"], players, meta["players"], meta["temperature"], where)
        counts["replay identical"] += 1
    counts["games"] += 1
    return counts


def print_counts(title: str, counts: collections.Counter) -> None:
    print(title)
    for k in sorted(counts):
        print(f"  {k:60s} {counts[k]}")


# ---------------------------------------------------------------------------------------------- entry points
def self_test(parity_games: int = 3) -> None:
    t0 = time.perf_counter()
    check_tables()
    played = run_scenarios()
    kan_counts = collections.Counter()
    for p in played:
        kan_counts += check_kan_emission(p.xml, p.scenario.name)
    print_counts("kan emission rows covered by the scenarios:", kan_counts)
    for row in ("ankan: N DORA T", "chained ankan: N DORA T N DORA T", "open_kan: N T DORA D",
                "open_kan rinshan tsumo / chankan: no DORA", "kakan + chankan: N AGARI",
                "second open kan: N T N DORA T DORA D", "ankan after open kan: N DORA DORA T"):
        expect(kan_counts[row] >= 1, f"kan emission row not covered: {row}")
    corpus = collections.Counter()
    for f in gate_logs():
        corpus += check_kan_emission(open(f, encoding="utf-8").read(), os.path.basename(f))
    print_counts(f"kan emission rows in {len(gate_logs())} real logs (0 violations):", corpus)
    print_counts("real tags reproduced by scoring.py:", check_real_tags(gate_logs()))
    print_counts("real-log fixtures (tests/data/engine_fixtures):", check_real_fixtures())
    corpus = check_corpus(gate_logs())
    print_counts(f"rule oracles on the {len(gate_logs())} gate logs:", corpus)
    expect(corpus["disagreements"] == 0, f"{corpus['disagreements']} rule disagreements on the gate logs")
    players = [UniformPlayer(), UniformPlayer(), GreedyPlayer(), GreedyPlayer()]
    labels = [p.label for p in players]
    for p in players:
        p.reseed(0)
    record = Game(players, labels, 0).play()
    counts = check_rounds(record.xml, "game 0") + check_agari_tags(record.xml, "game 0")
    counts += check_kan_emission(record.xml, "game 0") + check_sidecar(record.xml, record.decisions, where="game 0")
    check_replay(record.xml, record.decisions, 0, players_of(record.meta), labels, where="game 0")
    counts["replay identical"] += 1
    print_counts(f"one uniform/greedy game (seed 0, {record.meta['rounds']} rounds, {record.meta['duration_s']} s):", counts)
    check_corruptions(played, record.xml, record.decisions)
    parity = collections.Counter()
    for f in parity_logs(parity_games):
        parity += check_prefix_parity(f)
    print_counts(f"prefix-vs-completed parity on {parity_games} real logs:", parity)
    print(f"engine self-test OK in {time.perf_counter() - t0:.0f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--games-dir", help="a match out-dir (manifest.txt) or one game dir")
    ap.add_argument("--replay", type=int, default=1,
                    help="replay the first N games with the players rebuilt from their meta.json")
    ap.add_argument("--parity", type=int, default=0, help="prefix parity on the first N logs under --data-root")
    ap.add_argument("--corpus", type=int, default=0, help="the rule oracles on the first N logs under --data-root")
    ap.add_argument("--data-root", help="processed logs root (<root>/<year>/<day>/<day>_<nn>/*.log)")
    ap.add_argument("--days", nargs="*", default=["2022/202201%02d" % d for d in range(1, 8)])
    args = ap.parse_args(argv)
    if args.test:
        self_test()
    if args.games_dir:
        manifest = os.path.join(args.games_dir, "manifest.txt")
        dirs = [os.path.dirname(x.strip()) for x in open(manifest)] if os.path.exists(manifest) else [args.games_dir]
        counts = collections.Counter()
        t0 = time.perf_counter()
        for i, d in enumerate(dirs):
            counts += check_game_dir(d, replay=i < args.replay)
        print_counts(f"{len(dirs)} generated games checked in {time.perf_counter() - t0:.0f} s:", counts)
    if args.parity:
        files = sorted(f for day in args.days for f in glob.glob(os.path.join(args.data_root, day, "*", "*.log")))
        counts = collections.Counter()
        t0 = time.perf_counter()
        for f in files[:args.parity]:
            try:
                process_xml.process(f, admission=False)
            except Exception as e:  # a log the parser rejects even without the admission filter: no prefix parses
                counts[f"skipped, parser rejects the whole log: {type(e).__name__}"] += 1
                continue
            counts += check_prefix_parity(f)
        print_counts(f"prefix-vs-completed parity on the first {min(args.parity, len(files))} real logs "
                     f"in {time.perf_counter() - t0:.0f} s:", counts)
    if args.corpus:
        files = sorted(f for day in args.days for f in glob.glob(os.path.join(args.data_root, day, "*", "*gm-00a9*.log")))
        t0 = time.perf_counter()
        counts = check_corpus(files[:args.corpus])
        print_counts(f"rule oracles on the first {min(args.corpus, len(files))} real 4p hanchan logs in "
                     f"{time.perf_counter() - t0:.0f} s:", counts)
        if counts["disagreements"]:
            raise SystemExit(f"{counts['disagreements']} disagreements")


if __name__ == "__main__":
    main()
