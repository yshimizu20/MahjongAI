"""One round of a Tenhou 4-player hanchan as a generator (the chess `_game_flow` shape): it yields each decision
point and receives the seat's Choice, and appends Tenhou XML events as it goes.

Legality is the parser's: before every decision the round parses its own event prefix with
`process_xml.process_events(pending=True, admission=False)` and offers exactly the pending halfturn's decisions.
The round owns the wall, the real 136-id hands, call priority, the kan emission order, the terminal tags and the
settlement.

Kan emission table [C5] (every row copied from the 2021 4p logs; see engine_lib.py --test for the fixtures):
  ankan                       N DORA T            the indicator turns at once, before the rinshan draw
  chained ankan               N DORA T N DORA T   (a second ankan after the first rinshan draw)
  open kan (daiminkan, kakan) N T DORA D          the indicator is written before the kan player's discard
  open kan, rinshan tsumo     N T AGARI           no DORA: the indicator never turns
  kakan answered by chankan   N AGARI
  second open kan             N DORA T DORA D     the first kan's indicator turns right after the second N
  ankan after an open kan     N DORA DORA T       the open kan's indicator, then the ankan's
Call priority after a discard: every responder with an offer is asked and recorded; ron > pon / daiminkan > chi;
two rons = a double ron (two AGARI tags, head-first order); three rons = `ron3`. A riichi declaration's stick is
paid (REACH step 2) before a call on the declaration discard. After the discard passes: reach4, kaze4 and kan4
are checked before a call is honoured.
Terminal tags [C6]: AGARI ba hai [m] machi ten yaku|yakuman doraHai [doraHaiUra] who fromWho [paoWho] sc [owari];
RYUUKYOKU [type] ba sc [hai<seat>...]: exhaustive draws show every tenpai hand (nagashi mangan too, type="nm"),
yao9 the aborting seat's 14 tiles, ron3 the three hands, reach4 all four, kan4 and kaze4 none.
"""
from dataclasses import dataclass, field, replace
from typing import Dict, Generator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from MahjongAI.draw import Naki
from MahjongAI.gameinfo import GameInfo
from MahjongAI.process_xml import process_events
from MahjongAI.turn import DiscardTurn, DuringTurn, HalfTurn, PostTurn
from MahjongAI.utils import agari, legal
from MahjongAI.utils.constants import DECISION_ABORT_IDX, DECISION_AGARI_IDX, DECISION_NAKI_IDX, DECISION_REACH_IDX
from MahjongAI.utils.legal import legal_discard_mask
from MahjongAI.engine import meld_codes, scoring
from MahjongAI.engine.wall import Wall

DRAW_TAGS = "TUVW"
DISCARD_TAGS = "DEFG"
RED_SLOT = {16: 34, 52: 35, 88: 36}
SLOT_RED = {34: 16, 35: 52, 36: 88}
DURING_PASS, DURING_TSUMO, DURING_RIICHI, ANKAN_IDX, KAKAN_IDX, DURING_KYUUSHU = 0, 1, 2, 3, 37, 71
DURING_DIM, DISCARD_DIM, POST_DIM = 72, 37, 154
POST_PASS, POST_RON, PON_IDX, MINKAN_IDX = 0, 1, 83, 120
WINDS = frozenset({27, 28, 29, 30})
DRAGONS = frozenset({31, 32, 33})


class Choice(NamedTuple):
    """A player's answer: the action index in the head's layout, the play-time distribution over the legal
    actions, whether it was sampled (else argmax / rule), and the time the player took."""
    action: int
    candidates: Dict[int, float]
    sampled: bool
    time_ms: float


class DecisionPoint(NamedTuple):
    """kind: 'during' | 'discard' | 'post'; halfturn: the pending halfturn (a PostTurn: the responder is `seat`);
    tokens: the round's history tokens so far; halfturn_index: the halfturn's index in the completed round."""
    kind: str
    seat: int
    halfturn: HalfTurn
    tokens: List[int]
    halfturn_index: int


@dataclass(frozen=True)
class RoundSpec:
    index: int  # 0-3 East, 4-7 South, 8-11 West
    honba: int
    kyotaku: int
    scores: Tuple[int, int, int, int]  # hundreds

    @property
    def oya(self) -> int:
        return self.index % 4


@dataclass
class RoundOutcome:
    kind: str  # 'agari' | 'exhaustive' | 'abort'
    scores: List[int]
    kyotaku: int  # sticks left on the table
    dealer_won: bool = False
    dealer_tenpai: bool = False
    abort_type: Optional[str] = None
    nagashi: bool = False
    winners: List[int] = field(default_factory=list)
    from_who: Optional[int] = None

    @property
    def renchan(self) -> bool:
        """Dealer continuation (round-transition table row 2): dealer win, dealer tenpai at a draw, any abort."""
        return self.dealer_won or (self.kind == "exhaustive" and self.dealer_tenpai) or self.kind == "abort"


def render(tag: str, attrs: Dict[str, str]) -> str:
    """One Tenhou XML tag, spaced as Tenhou writes it."""
    if not attrs:
        return f"<{tag}/>"
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    if tag in ("INIT", "REACH"):
        return f"<{tag} {body}/>"
    return f"<{tag} {body} />"


def counts34(ids: Sequence[int]) -> np.ndarray:
    return np.bincount([t // 4 for t in ids], minlength=34).astype(int)


def legal_mask(halfturn: HalfTurn, seat: int) -> np.ndarray:
    """The head's legal-action mask of a pending halfturn, as the tensor builders write it (encoder_conv.py)."""
    if isinstance(halfturn, DuringTurn):
        mask = np.zeros(DURING_DIM, dtype=np.float32)
        mask[DURING_PASS] = 1.0
        if halfturn.decisions[DECISION_AGARI_IDX]:
            mask[DURING_TSUMO] = 1.0
        if halfturn.decisions[DECISION_REACH_IDX]:
            mask[DURING_RIICHI] = 1.0
        for d in halfturn.decisions[DECISION_NAKI_IDX]:
            mask[d.naki.get_during_turn_filter_idx()] = 1.0
        if halfturn.decisions[DECISION_ABORT_IDX]:
            mask[DURING_KYUUSHU] = 1.0
        return mask
    if isinstance(halfturn, DiscardTurn):
        return legal_discard_mask(halfturn.stateObj.hand_tensor, halfturn.is_riichi_declaration, halfturn.after_call)
    assert isinstance(halfturn, PostTurn)
    mask = np.zeros(POST_DIM, dtype=np.float32)
    decisions = halfturn.decisions[seat]
    if decisions[DECISION_AGARI_IDX] or decisions[DECISION_NAKI_IDX]:
        mask[POST_PASS] = 1.0
    if decisions[DECISION_AGARI_IDX]:
        mask[POST_RON] = 1.0
    for d in decisions[DECISION_NAKI_IDX]:
        mask[d.naki.get_post_turn_filter_idx()] = 1.0
    return mask


class Round:
    """The state of one round and its event loop (`play`). Scores and deltas in hundreds."""

    def __init__(self, spec: RoundSpec, wall: Wall, dice: Tuple[int, int], gameinfo: GameInfo,
                 labels: Sequence[str] = ("", "", "", ""), k0: int = 0, kyoku: int = 0):
        self.spec = spec
        self.kyoku = kyoku  # the round's position in the game (the sidecar's key into the parsed game)
        self.wall = wall
        self.dice = dice
        self.gameinfo = gameinfo
        self.labels = list(labels)
        self.k = k0
        self.oya = spec.oya
        self.scores = list(spec.scores)
        self.honba = spec.honba
        self.kyotaku = spec.kyotaku
        self.events: List[Tuple[str, Dict[str, str]]] = []
        self.decisions: List[dict] = []  # the sidecar records of this round
        self.hands: List[List[int]] = [[] for _ in range(4)]
        self.melds: List[List[int]] = [[] for _ in range(4)]  # Tenhou codes, in call order
        self.discards: List[List[int]] = [[] for _ in range(4)]
        self.river_called = [False] * 4  # a discard of this seat was called: no nagashi mangan
        self.any_call = False
        self.riichi = [False] * 4
        self.revealed: List[int] = []  # indicators the XML has turned (INIT's first)
        self.pending_indicator: Optional[int] = None  # an open kan's indicator, written before the next discard / N
        self.n_indicators = 0
        self.pao: List[Dict[int, int]] = [{} for _ in range(4)]  # seat -> {yakuman id: liable seat}
        self.drawn: Optional[int] = None

    # ------------------------------------------------------------------ XML
    def xml(self) -> str:
        return "".join(render(tag, attrs) for tag, attrs in self.events)

    def _emit(self, tag: str, **attrs) -> str:
        attrs = {k: str(v) for k, v in attrs.items()}
        self.events.append((tag, attrs))
        return render(tag, attrs)

    def _parse(self):
        rounds = [[{"event": tag, "attr": attrs} for tag, attrs in self.events]]
        H, T, due = process_events(self.gameinfo, rounds, pending=True, admission=False)
        return H[0], T[0], due

    @staticmethod
    def _index_of(halfturn: HalfTurn, halfturns: list) -> int:
        """The halfturn's index in the completed round: built already (the last one) or the pending discard."""
        return len(halfturns) - 1 if halfturns and halfturns[-1] is halfturn else len(halfturns)

    # ------------------------------------------------------------------ decisions
    def _ask(self, kind: str, seat: int, halfturn: HalfTurn, halfturns: list, tokens: List[int]):
        """Yield the decision point, record the answer; returns (action, record)."""
        mask = legal_mask(halfturn, seat)
        offer = np.flatnonzero(mask).tolist()
        choice: Choice = yield DecisionPoint(kind, seat, halfturn, tokens, self._index_of(halfturn, halfturns))
        assert choice.action in offer, f"{kind} by seat {seat}: action {choice.action} not in the offer {offer}"
        record = {"kyoku": self.kyoku, "round": self.spec.index, "honba": self.honba, "k": self.k, "kind": kind,
                  "seat": seat,
                  "halfturn": self._index_of(halfturn, halfturns), "offer": offer,
                  "candidates": {str(a): float(p) for a, p in sorted(choice.candidates.items())},
                  "chosen": int(choice.action), "resolved": None, "sampled": bool(choice.sampled),
                  "time_ms": float(choice.time_ms), "player": self.labels[seat]}
        self.k += 1
        self.decisions.append(record)
        return choice.action, record

    @staticmethod
    def _resolve(record: Optional[dict], outcome: str, event: Optional[str]) -> None:
        if record is not None:
            record["resolved"] = {"outcome": outcome, "event": event}

    # ------------------------------------------------------------------ tiles
    def _pick(self, seat: int, kind: int, red: bool) -> int:
        """One id of `kind` from the seat's hand: the red five, or the lowest plain copy."""
        if red:
            tile = next(t for t in RED_SLOT if t // 4 == kind)
            assert tile in self.hands[seat], (seat, tile)
            return tile
        return min(t for t in self.hands[seat] if t // 4 == kind and t not in RED_SLOT)

    def _tile_for_slot(self, seat: int, slot: int) -> int:
        """The discard for a 37-slot action: a red slot is its red id; a kind slot prefers the tile just drawn
        (tsumogiri), else the lowest plain copy."""
        if slot in SLOT_RED:
            tile = SLOT_RED[slot]
            assert tile in self.hands[seat], (seat, tile)
            return tile
        copies = [t for t in self.hands[seat] if t // 4 == slot and t not in RED_SLOT]
        assert copies, (seat, slot)
        return self.drawn if self.drawn in copies else min(copies)

    def _take(self, seat: int, tiles: Sequence[int]) -> None:
        for t in tiles:
            self.hands[seat].remove(t)

    def _hai(self, seat: int) -> str:
        return ",".join(map(str, sorted(self.hands[seat])))

    # ------------------------------------------------------------------ indicators
    def _flush_pending(self) -> None:
        if self.pending_indicator is not None:
            self._emit("DORA", hai=self.pending_indicator)
            self.revealed.append(self.pending_indicator)
            self.pending_indicator = None

    def _new_indicator(self) -> int:
        tile = self.wall.indicator(self.n_indicators)
        self.n_indicators += 1
        return tile

    # ------------------------------------------------------------------ calls
    def _note_pao(self, seat: int, kind: int, from_who: int) -> None:
        """After a called set: the discarder is liable for a daisangen (3rd dragon set) / daisuushii (4th wind set)."""
        if kind not in DRAGONS and kind not in WINDS:
            return
        kinds = [meld_codes.decode_meld(c).ids[0] // 4 for c in self.melds[seat]]
        if kind in DRAGONS and sum(k in DRAGONS for k in kinds) == 3:
            self.pao[seat][39] = from_who
        if kind in WINDS and sum(k in WINDS for k in kinds) == 4:
            self.pao[seat][49] = from_who

    def _call(self, caller: int, discarder: int, tile: int, naki: Naki) -> int:
        """Execute an offered chi / pon / daiminkan on `tile`: the real code, hand and melds updated."""
        from_rel = (discarder - caller) % 4
        if naki.is_chi():
            color, number, which, has_red, *_ = naki.pattern_chi()
            kinds = [9 * color + number + i for i in range(3) if i != which]
            ids = [self._pick(caller, k, has_red and k % 9 == 4) for k in kinds]
            code = meld_codes.encode_chi(ids + [tile], tile)
        elif naki.is_pon():
            kind = tile // 4
            own_red = PON_IDX + 27 <= naki.get_post_turn_filter_idx() < PON_IDX + 30
            ids = [self._pick(caller, kind, True), self._pick(caller, kind, False)] if own_red else sorted(
                t for t in self.hands[caller] if t // 4 == kind and t not in RED_SLOT)[:2]
            code = meld_codes.encode_pon(ids + [tile], tile, from_rel)
        else:
            assert naki.is_minkan()
            ids = [t for t in self.hands[caller] if t // 4 == tile // 4]
            assert len(ids) == 3, ids
            code = meld_codes.encode_minkan(tile, from_rel)
        self._take(caller, ids)
        self.melds[caller].append(code)
        self.river_called[discarder] = True
        self.any_call = True
        self._emit("N", who=caller, m=code)
        if not naki.is_chi():
            self._note_pao(caller, tile // 4, discarder)
        return code

    def _ankan(self, seat: int, kind: int) -> str:
        """N, then the indicators: an open kan's pending one first, then the ankan's own. Returns the N tag."""
        ids = [t for t in self.hands[seat] if t // 4 == kind]
        assert len(ids) == 4, ids
        self._take(seat, ids)
        self.melds[seat].append(meld_codes.encode_ankan(kind))
        self.any_call = True
        n_tag = self._emit("N", who=seat, m=self.melds[seat][-1])
        self._flush_pending()
        indicator = self._new_indicator()
        self._emit("DORA", hai=indicator)
        self.revealed.append(indicator)
        return n_tag

    def _kakan(self, seat: int, naki: Naki) -> None:
        """The offered kakan is the exact log code (Naki.kakan_of on the seat's log pon)."""
        code = naki.naki_code
        meld = meld_codes.decode_meld(code)
        pon_code = (code & ~0x10) | 0x8
        assert pon_code in self.melds[seat], (pon_code, self.melds[seat])
        assert meld.added in self.hands[seat], (meld.added, self.hands[seat])
        self._take(seat, [meld.added])
        self.melds[seat][self.melds[seat].index(pon_code)] = code
        self.any_call = True
        self._emit("N", who=seat, m=code)

    # ------------------------------------------------------------------ results
    def _flags(self, winner: int, from_who: int, names: Sequence[str]) -> scoring.WinFlags:
        return scoring.flags_from_yaku(names, tsumo=winner == from_who, seat_wind=agari.seat_wind(winner, self.oya),
                                       round_wind=agari.round_wind(self.spec.index))

    def _agari(self, wins: List[Tuple[int, Sequence[str]]], from_who: int, win_tile: int) -> RoundOutcome:
        """One AGARI tag per winner, head-first order; honba and sticks go to the first."""
        for i, (winner, names) in enumerate(wins):
            flags = self._flags(winner, from_who, names)
            concealed = list(self.hands[winner]) + ([] if winner == from_who else [win_tile])
            # ura from the ACCEPTED riichi state: a yakuman hand's yaku names carry no Riichi (the parser lists
            # yakuman only), so the names cannot say whether the winner was in riichi
            riichi = self.riichi[winner]
            assert riichi or not (flags.riichi or flags.double_riichi), (winner, names, self.riichi)
            if riichi and not (flags.riichi or flags.double_riichi):
                flags = replace(flags, riichi=True)
            ura = self.wall.ura_indicators(len(self.revealed)) if riichi else []
            score = scoring.score_hand(concealed, self.melds[winner], win_tile, flags, self.revealed, ura)
            honba, kyotaku = (self.honba, self.kyotaku) if i == 0 else (0, 0)
            pao = {yid: liable for yid, liable in self.pao[winner].items() if yid in score.yakuman}
            deltas = scoring.agari_deltas(score, winner, from_who, self.oya, honba, kyotaku, pao)
            attrs = {"ba": f"{self.honba},{kyotaku}", "hai": ",".join(map(str, sorted(concealed)))}
            if self.melds[winner]:
                attrs["m"] = ",".join(map(str, self.melds[winner]))
            attrs["machi"] = win_tile
            attrs["ten"] = score.ten_attr()
            key, value = score.yaku_attr()
            attrs[key] = value
            attrs["doraHai"] = ",".join(map(str, self.revealed))
            if riichi:
                attrs["doraHaiUra"] = ",".join(map(str, ura))
            attrs["who"] = winner
            attrs["fromWho"] = from_who
            if pao:
                attrs["paoWho"] = next(iter(pao.values()))
            attrs["sc"] = scoring.sc_attr(self.scores, deltas)
            self._emit("AGARI", **attrs)
            self.scores = [s + d for s, d in zip(self.scores, deltas)]
            self.kyotaku = 0
        return RoundOutcome("agari", self.scores, 0, dealer_won=any(w == self.oya for w, _ in wins),
                            winners=[w for w, _ in wins], from_who=from_who)

    def _abort(self, type_: str, shown: Sequence[int]) -> RoundOutcome:
        attrs = {"type": type_, "ba": f"{self.honba},{self.kyotaku}", "sc": scoring.sc_attr(self.scores, [0] * 4)}
        for q in shown:
            attrs[f"hai{q}"] = self._hai(q)
        self._emit("RYUUKYOKU", **attrs)
        return RoundOutcome("abort", self.scores, self.kyotaku, abort_type=type_)

    def _exhaustive(self) -> RoundOutcome:
        qualifiers = [q for q in range(4)
                      if not self.river_called[q] and all(t // 4 in legal.YAOCHU_KINDS for t in self.discards[q])]
        tenpai = [bool(agari.waits(counts34(self.hands[q]))) for q in range(4)]
        deltas = scoring.nagashi_deltas(qualifiers, self.oya) if qualifiers else scoring.ryuukyoku_deltas(tenpai)
        attrs = {}
        if qualifiers:
            attrs["type"] = "nm"
        attrs["ba"] = f"{self.honba},{self.kyotaku}"
        attrs["sc"] = scoring.sc_attr(self.scores, deltas)
        for q in range(4):
            if tenpai[q]:
                attrs[f"hai{q}"] = self._hai(q)
        self._emit("RYUUKYOKU", **attrs)
        self.scores = [s + d for s, d in zip(self.scores, deltas)]
        return RoundOutcome("exhaustive", self.scores, self.kyotaku, dealer_tenpai=tenpai[self.oya],
                            nagashi=bool(qualifiers))

    # ------------------------------------------------------------------ the event loop
    def play(self) -> Generator[DecisionPoint, Choice, RoundOutcome]:
        oya = self.oya
        self.hands = self.wall.deal(oya)
        self.revealed = [self._new_indicator()]
        self._emit("INIT", seed=f"{self.spec.index},{self.honba},{self.kyotaku},{self.dice[0]},{self.dice[1]},"
                   f"{self.revealed[0]}", ten=",".join(map(str, self.scores)), oya=oya,
                   **{f"hai{q}": ",".join(map(str, self.hands[q])) for q in range(4)})
        seat = oya
        rinshan = False
        while True:
            # ---- draw
            tile = self.wall.rinshan_draw() if rinshan else self.wall.draw()
            rinshan = False
            self.hands[seat].append(tile)
            self.drawn = tile
            self._emit(DRAW_TAGS[seat] + str(tile))
            halfturns, tokens, due = self._parse()
            during, discard_turn = due
            assert isinstance(during, DuringTurn) and isinstance(discard_turn, DiscardTurn)
            action, record = DURING_PASS, None
            if sum(map(len, during.decisions)):
                action, record = yield from self._ask("during", seat, during, halfturns, tokens)
            if action == DURING_TSUMO:
                names = during.decisions[DECISION_AGARI_IDX][0].yaku
                outcome = self._agari([(seat, names)], seat, tile)
                self._resolve(record, "executed", self.xml_last())
                return outcome
            if action == DURING_KYUUSHU:
                outcome = self._abort("yao9", [seat])
                self._resolve(record, "executed", self.xml_last())
                return outcome
            if ANKAN_IDX <= action < KAKAN_IDX:
                n_tag = self._ankan(seat, action - ANKAN_IDX)
                self._resolve(record, "executed", n_tag)
                rinshan = True
                continue
            if KAKAN_IDX <= action < DURING_KYUUSHU:
                naki = next(d.naki for d in during.decisions[DECISION_NAKI_IDX]
                            if d.naki.get_during_turn_filter_idx() == action)
                self._kakan(seat, naki)
                self._resolve(record, "executed", self.xml_last())
                kakan_tile = meld_codes.decode_meld(naki.naki_code).added
                halfturns, tokens, due = self._parse()
                (post,) = due
                assert isinstance(post, PostTurn) and post.player == seat
                rons = yield from self._post_window(post, halfturns, tokens, seat, kakan_tile, chankan=True)
                if rons is not None:
                    return rons
                self._flush_pending()
                self.pending_indicator = self._new_indicator()
                rinshan = True
                continue
            declaring = action == DURING_RIICHI
            if declaring:
                self._emit("REACH", who=seat, step=1)
                self._resolve(record, "executed", self.xml_last())
                halfturns, tokens, due = self._parse()
                (discard_turn,) = due
                assert isinstance(discard_turn, DiscardTurn) and discard_turn.is_riichi_declaration
            elif record is not None:
                self._resolve(record, "executed", None)
            # ---- discard
            outcome = yield from self._discard_and_window(seat, discard_turn, halfturns, tokens, declaring)
            if isinstance(outcome, RoundOutcome):
                return outcome
            seat, rinshan = outcome
            while rinshan is None:  # a chi / pon: the caller discards without a draw
                halfturns, tokens, due = self._parse()
                (discard_turn,) = due
                assert isinstance(discard_turn, DiscardTurn) and discard_turn.after_call is not None
                outcome = yield from self._discard_and_window(seat, discard_turn, halfturns, tokens, False)
                if isinstance(outcome, RoundOutcome):
                    return outcome
                seat, rinshan = outcome

    def xml_last(self) -> str:
        return render(*self.events[-1])

    def _discard_and_window(self, seat: int, discard_turn: DiscardTurn, halfturns: list, tokens: List[int],
                            declaring: bool):
        """The seat's discard, then the responders' window. Returns a RoundOutcome, or (next seat, rinshan) where
        rinshan is None after a chi / pon (the caller discards next), True after a daiminkan, False otherwise."""
        after_call = discard_turn.after_call is not None
        if self.riichi[seat] and not declaring:
            tile, record = self.drawn, None  # a riichi player's discard is forced: the drawn tile
            assert tile is not None and tile in self.hands[seat]
        else:
            slot, record = yield from self._ask("discard", seat, discard_turn, halfturns, tokens)
            tile = self._tile_for_slot(seat, slot)
        self._flush_pending()
        self._take(seat, [tile])
        self.discards[seat].append(tile)
        self._emit(DISCARD_TAGS[seat] + str(tile))
        self._resolve(record, "executed", self.xml_last())
        self.drawn = None
        halfturns, tokens, due = self._parse()
        (post,) = due
        assert isinstance(post, PostTurn) and post.player == seat
        result = yield from self._post_window(post, halfturns, tokens, seat, tile, chankan=False)
        if isinstance(result, RoundOutcome):
            return result
        if declaring:  # the declaration discard passed: the stick is paid before any call
            self.riichi[seat] = True
            self.scores[seat] -= scoring.RIICHI_STICK
            self.kyotaku += 1
            self._emit("REACH", who=seat, ten=",".join(map(str, self.scores)), step=2)
            if all(self.riichi):
                self._preempt(result, "reach4")
                return self._abort("reach4", range(4))
        n_discards = [len(d) for d in self.discards]
        if n_discards == [1] * 4 and not self.any_call and not after_call:
            kinds = {d[0] // 4 for d in self.discards}
            if len(kinds) == 1 and kinds <= WINDS:
                self._preempt(result, "kaze4")
                return self._abort("kaze4", [])
        n_kans, owners = legal.kans_on_table([[Naki(c) for c in m] for m in self.melds])
        if legal.suukaikan_pending(n_kans, owners):
            assert not result, "a call was offered with suukaikan pending"
            return self._abort("kan4", [])
        if result:  # the winning call: pon / daiminkan over chi
            caller, action, naki, record = max(result, key=lambda r: r[1] >= PON_IDX)
            for q, a, _, rec in result:
                if q != caller:
                    self._resolve(rec, "preempted", None)
            self._call(caller, seat, tile, naki)
            self._resolve(record, "executed", self.xml_last())
            if naki.is_minkan():
                self._flush_pending()
                self.pending_indicator = self._new_indicator()
                return caller, True
            return caller, None
        if self.wall.draws_left == 0:
            return self._exhaustive()
        return (seat + 1) % 4, False

    @staticmethod
    def _preempt(calls, reason: str) -> None:
        for _, _, _, rec in calls:
            Round._resolve(rec, "preempted", reason)

    def _post_window(self, post: PostTurn, halfturns: list, tokens: List[int], from_who: int, tile: int,
                     chankan: bool):
        """Ask every responder with an offer (head-first order). Returns a RoundOutcome (ron(s), ron3), None
        (chankan window passed) or the list of chosen calls [(seat, action, naki, record)]."""
        order = [(from_who + i) % 4 for i in (1, 2, 3)]
        answers = []
        for q in order:
            decisions = post.decisions[q]
            if not decisions[DECISION_AGARI_IDX] and not decisions[DECISION_NAKI_IDX]:
                continue
            action, record = yield from self._ask("post", q, post, halfturns, tokens)
            answers.append((q, action, record))
        rons = [(q, rec) for q, a, rec in answers if a == POST_RON]
        calls = [(q, a, rec) for q, a, rec in answers if a >= 2]
        passes = [rec for q, a, rec in answers if a == POST_PASS]
        for rec in passes:
            self._resolve(rec, "passed", None)
        if len(rons) == 3:
            outcome = self._abort("ron3", [q for q, _ in rons])
            for _, rec in rons:
                self._resolve(rec, "executed", self.xml_last())
            return outcome
        if rons:
            for _, _, rec in calls:
                self._resolve(rec, "preempted", None)
            wins = [(q, post.decisions[q][DECISION_AGARI_IDX][0].yaku) for q, _ in rons]
            n_before = len(self.events)
            outcome = self._agari(wins, from_who, tile)
            for (_, rec), (tag, attrs) in zip(rons, self.events[n_before:]):
                self._resolve(rec, "executed", render(tag, attrs))
            return outcome
        if chankan:
            return None
        result = []
        for q, a, rec in calls:
            naki = next(d.naki for d in post.decisions[q][DECISION_NAKI_IDX] if d.naki.get_post_turn_filter_idx() == a)
            result.append((q, a, naki, rec))
        return result
