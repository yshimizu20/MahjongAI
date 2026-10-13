"""Pre-training gate checks: the parsed data must agree with the XML ground truth and with itself.

    python -m MahjongAI.utils.gates --games 150      # random games from DEFAULT_DAYS; every gate is an assert
    python -m MahjongAI.utils.gates --test           # self-test: passes on a clean game, raises on corrupted ones
Run from the directory that contains MahjongAI/ (python -I would drop it from sys.path).

A clean run prints ONE line. A failing gate raises AssertionError with the offending counts.
Gate groups (counted over every game, round, halfturn and built tensor):
  WINS    executed tsumo / ron labels == <AGARI> events in the XML (who == fromWho is a tsumo), plus 3 rons
          per triple-ron abort (<RYUUKYOKU type="ron3">)
  RIICHI  executed riichi labels == <REACH step="1"> events: the same seats in the same order per round, and
          the during-head riichi targets (index 2) == their count
  CALLS   executed chi / pon / minkan / ankan / kakan labels == <N> events by kind
  MASK    every label index is inside its action mask (during 71, discard 37, post 154); chi/pon indices injective
  POV     a PostTurn sample's state belongs to the responder; unseen counts + own hand <= 135 and no negative
          count in EVERY state (each PostTurn responder state included); no two StateObjects of a round share
          a mutable buffer (remaining_tiles_pov, hand, remaining_tiles, discards, safe tiles)
  TOKENS  the history tokens of every round == the tokens replayed from the XML alone (INIT / DORA indicator,
          REACH step 1, N, discards with the tsumogiri bit); a riichi declaration discard follows its own reach
          token; a discard by a player in riichi is the drawn tile; seat bits rotate so that the actor is seat 0
  SCALE   remaining_tsumo in [0, 70] and == 0 at an exhaustive draw; |x2|, |x3| <= 5 (normalised inputs;
          the whole 100,000-point pool on one seat is exactly 4.0)
  HISTORY in every state a model head reads, double_reaches, ippatsu and safe_tiles of all four seats == the
          values rebuilt from its token history alone (history_facts): double riichi = a reach before any call
          and before the seat's own first discard; ippatsu = no call since the reach (the kakan a chankan
          responder answers still counts) and no second draw; safe = the seat's own discards and every discard
          after its reach, per tile kind (a five marks its plain and its red slot), except the discard a
          responder is answering (not yet safe against anyone but its discarder)
  IPPATSU a riichi declarer's next DuringTurn has ippatsu = 1 (0 if a call came between); his next discard
          and his DuringTurn after that have 0
  RED     every state's red-five slots 34-36 of remaining_tiles_pov == 1 - (that red visible to the player:
          own hand, any discard, any meld, a dora indicator)
  TENSORS the BUILT tensors agree with the parser and the XML: win / riichi / call targets per head == XML events
          and every during / post target == the executed action of its sample (exact index, not only the kind);
          discard targets == TILE2IDX (red five -> slot 34-36) and the plain-five slot is enabled only
          with a plain copy; EVERY input of every sample == its StateObject and history: all 10 x1 planes,
          all 7 actor-first x2 rows, all 4 x3 scalars, the full rotated history prefix with zeros after it, and
          the padding mask; the post sample set follows the priority rule (executors always, pre-empted
          responders never); the last history token of a post sample is exactly the discard / kakan it
          answers; the tensor builder skipped nothing
"""
import argparse
import collections
import glob
import os
import random
import re
from unittest import mock

import numpy as np
import torch

from MahjongAI import process_xml
from MahjongAI.process_xml import process, InvalidGameException, discard2idx, naki2idx, reach2idx, new_dora2idx
from MahjongAI.turn import DuringTurn, DiscardTurn, PostTurn
from MahjongAI.draw import Naki
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_NAKI_IDX, DECISION_REACH_IDX, EventStateTypes, EventTypes, TILE2IDX
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor, N_X1_PLANES

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PACKAGE_DIR, "data", "processed")
SAMPLE_LOG = os.path.join(PACKAGE_DIR, "data", "sample", "sample_haifu.xml")
# Named self-test fixtures, (day, file, regex its XML must match): the 2021-01-01 pool lacks these events
RON3_LOG = ("20210105", "2021010521gm-00a9-0000-e76b4550.log", r'<RYUUKYOKU [^>]*type="ron3"')  # round 5: triple ron
RED_KAN_DORA_LOG = ("20210101", "2021010100gm-00a9-0000-3ceec6a2.log", r'<DORA hai="(16|52|88)"')  # red kan indicator
CHANKAN_IPPATSU_LOG = ("20210101", "2021010110gm-00a9-0000-85d17b74.log", r'<REACH [^>]*step="2"')  # kakan in ippatsu
EARLY_ANKAN_LOG = ("20210109", "2021010919gm-00a9-0000-b25465a8.log", r'<REACH [^>]*step="1"')  # ankan, then riichi
DEFAULT_DAYS = ["20210103", "20210214", "20210322", "20210430", "20210518", "20210626",
                "20210709", "20210815", "20210921", "20211030", "20211112", "20211225"]
KINDS = ["chi", "pon", "minkan", "ankan", "kakan"]
LABELLED = ["tsumo", "ron", "riichi"] + KINDS  # lab_<k> (parser) == tensor_<k> (built targets) == xml_<k>
HEADS = ["during", "discard", "post"]
ZERO_GATES = [
    "parse_error", "tensor_build_error",
    "post_wrong_pov", "post_answered_event_hidden", "pov_negative", "pov_sum_too_big", "state_shares_buffer",
    "index_collisions", "index_spill_into_pon",
    "first_token_not_dora", "token_xml_mismatch", "riichi_label_mismatch",
    "riichi_declaration_unflagged", "riichi_discard_not_tsumogiri", "rotation_mismatch",
    "tsumo_negative", "tsumo_over_70", "exhaustive_draw_tsumo_not_zero", "x_scale_over_5", "x2_not_actor_first",
    "x1_plane_count", "safe_plane_mismatch", "double_riichi_mismatch", "ippatsu_history_mismatch",
    "discard_sample_count_mismatch", "tensor_skipped", "discard_label_mismatch", "discard_mask_red_rule",
    "hand_plane_mismatch", "x1_plane_mismatch", "x2_row_mismatch", "x3_scalar_mismatch",
    "history_prefix_mismatch", "padding_mask_mismatch",
    "post_sample_count_mismatch", "post_history_last_token",
    "during_sample_count_mismatch", "during_target_mismatch", "post_target_mismatch",
    "ippatsu_window_wrong", "red_pov_mismatch",
]
TARGET_GATE = {"during": "during_target_mismatch", "discard": "discard_label_mismatch", "post": "post_target_mismatch"}
RED_TO_PLAIN = {34: 4, 35: 13, 36: 22}
PLAIN_TO_RED = {plain: red for red, plain in RED_TO_PLAIN.items()}
BUFFER_FIELDS = ("remaining_tiles_pov", "hand_tensor", "remaining_tiles", "sutehai_tensors", "safe_tiles")
XML_EVENT = re.compile(r"<(INIT|DORA|REACH|N|AGARI|RYUUKYOKU|[TUVWDEFG]\d+)\b([^>]*)>")


def kind_of(nk: Naki) -> str:
    return "chi" if nk.is_chi() else "pon" if nk.is_pon() else "kakan" if nk.is_kakan() else "minkan" if nk.is_minkan() else "ankan"


def _event(tok): return (tok >> 13) & 7
def _etype(tok): return (tok >> 10) & 7
def _who(tok): return (tok >> 8) & 3
def _tile(tok): return tok & 0x7F
def _is_discard(tok): return _event(tok) == EventStateTypes.DISCARD
def _is_reach(tok): return _event(tok) == EventStateTypes.DURING and _etype(tok) == EventTypes.RIICHI
def _has_seat(tok): return _event(tok) in (EventStateTypes.DISCARD, EventStateTypes.DURING, EventStateTypes.POST)
def _is_call(tok): return _event(tok) in (EventStateTypes.DURING, EventStateTypes.POST) and _etype(tok) == EventTypes.NAKI


def rotate(tokens, actor):
    """Tokens with their seat bits relative to actor (0 = actor), restated from the token layout."""
    return [(t & ~(3 << 8)) | (((_who(t) - actor) % 4) << 8) if _has_seat(t) else t for t in tokens]


def history_facts(tokens, answering):
    """Per seat q, what the token history alone says: None if q has no reach token, else
    (double, called, own_after, safe):
      double    q's reach came before any call and before q's own first discard (double riichi)
      called    a call token sits after q's reach, the last token excluded when answering (a chankan responder
                answers the kakan token: ippatsu still counts for that ron)
      own_after q's discard tokens after the reach (1 = the declaration discard: q has not drawn again yet)
      safe      37-index slots safe against q: q's own discards and every discard after the reach, by tile kind
                (a five marks its plain and its red slot); when answering, the answered discard (the last
                token) is not yet safe against anyone but its discarder
    answering: the history of a PostTurn responder state, whose last token is the discard / kakan answered."""
    pending = len(tokens) - 1 if answering else -1
    facts = []
    for q in range(4):
        r = next((i for i, t in enumerate(tokens) if _is_reach(t) and _who(t) == q), None)
        if r is None:
            facts.append(None)
            continue
        double = not any(_is_call(t) or (_is_discard(t) and _who(t) == q) for t in tokens[:r])
        called = any(_is_call(tokens[i]) for i in range(r + 1, len(tokens)) if i != pending)
        own_after = sum(1 for t in tokens[r + 1:] if _is_discard(t) and _who(t) == q)
        kinds = {RED_TO_PLAIN.get(_tile(t), _tile(t)) for i, t in enumerate(tokens)
                 if _is_discard(t) and (_who(t) == q or r < i != pending)}
        facts.append((double, called, own_after, kinds | {PLAIN_TO_RED[k] for k in kinds if k in PLAIN_TO_RED}))
    return facts


def check_history_fields(so, facts, discarding, C):
    """double_reaches, ippatsu and safe_tiles of all four seats in so == history_facts of its history.
    discarding: so is a DiscardTurn state; q's own discard after the declaration follows q's next draw."""
    for q in range(4):
        double, called, own_after, safe = facts[q] or (False, True, 0, set())
        ippatsu = 0 if facts[q] is None or called else int(own_after == 0 or (own_after == 1 and not (discarding and so.player == q)))
        if so.double_reaches[q] != int(double):
            C["double_riichi_mismatch"] += 1
        if so.ippatsu[q] != ippatsu:
            C["ippatsu_history_mismatch"] += 1
        if set(np.flatnonzero(so.safe_tiles[q]).tolist()) != safe:
            C["safe_plane_mismatch"] += 1


def model_states(halfturns, tokens):
    """(state, history prefix, answering, discarding) for every StateObject a model head reads: the decider's state
    of each DuringTurn / DiscardTurn and EVERY responder state of each PostTurn (PostTurn.stateObj is the
    discarder's debug snapshot)."""
    for t in halfturns:
        hist = tokens[: t.encoding_idx]
        if isinstance(t, PostTurn):
            for rs in t.responder_states.values():
                yield rs, hist, True, False
        else:
            yield t.stateObj, hist, False, isinstance(t, DiscardTurn)


def _buffer_id(a):
    while isinstance(a, np.ndarray) and a.base is not None:
        a = a.base
    return id(a)


def shared_buffers(halfturns) -> int:
    """Buffers shared by two distinct StateObjects of one round, over BUFFER_FIELDS (the round loop mutates
    these in place, so a shared buffer shows a later value). PostTurn.stateObj may BE an earlier state object."""
    states = {id(s): s for t in halfturns for s in [t.stateObj] + list(getattr(t, "responder_states", {}).values())}
    shared = 0
    for field in BUFFER_FIELDS:
        owners = collections.Counter(_buffer_id(getattr(s, field)) for s in states.values() if getattr(s, field) is not None)
        shared += sum(n - 1 for n in owners.values() if n > 1)
    return shared


def ippatsu_windows(halfturns, tokens):
    """Per riichi declaration discard: (declarer, called, first, disc, nxt). first = the declarer's next
    DuringTurn, called = a call token sits between the declaration discard and first, disc = the declarer's
    next DiscardTurn after first, nxt = his DuringTurn after first; None when the round ends before."""
    for i, d in enumerate(halfturns):
        if not (isinstance(d, DiscardTurn) and d.is_riichi_declaration):
            continue
        p = d.player
        mine = [t for t in halfturns[i + 1:] if t.player == p and isinstance(t, (DuringTurn, DiscardTurn))]
        first = next((t for t in mine if isinstance(t, DuringTurn)), None)
        if first is None:
            continue
        rest = mine[mine.index(first) + 1:]
        called = any(_is_call(tok) for tok in tokens[d.encoding_idx:first.encoding_idx])
        disc = next((t for t in rest if isinstance(t, DiscardTurn)), None)
        nxt = next((t for t in rest if isinstance(t, DuringTurn)), None)
        yield p, called, first, disc, nxt


def red_visible(so, s) -> bool:
    """True iff red five s (tile 16 + 36 s) is visible to so.player: own hand, any discard, any meld, a dora indicator."""
    red, kind = 16 + 36 * s, 4 + 9 * s
    if so.hand_tensor[34 + s] > 0 or (so.sutehai_tensors[:, 34 + s] > 0).any() or red in so.dora:
        return True
    for meld in (m for ms in so.melds for m in ms):
        exposed, acquired = meld.get_exposed()
        if meld.is_chi() or meld.is_pon():
            if red in exposed + [acquired]:
                return True
        elif exposed[0] // 4 == kind:  # a kan of fives holds all four copies (a kakan meld lists only the added one)
            return True
    return False


def check_red_pov(so, C):
    for s in range(3):
        if so.remaining_tiles_pov[34 + s] != 1 - int(red_visible(so, s)):
            C["red_pov_mismatch"] += 1


def check_bounds(so, C):
    """POV bounds of one StateObject: no negative unseen count, unseen + own hand <= 135, tsumo in [0, 70], red slots."""
    if (so.remaining_tiles_pov < 0).any():
        C["pov_negative"] += 1
    if so.remaining_tsumo < 0:
        C["tsumo_negative"] += 1
    if so.remaining_tsumo > 70:
        C["tsumo_over_70"] += 1
    if so.remaining_tiles_pov[:34].sum() + so.hand_tensor[:34].sum() > 135:
        C["pov_sum_too_big"] += 1
    check_red_pov(so, C)


def thermometer(counts) -> torch.Tensor:
    """(37,) counts -> (37, 4) plane with column j = 1 iff count > j; the layout of every x1 plane."""
    return (torch.arange(4) < torch.as_tensor(np.asarray(counts), dtype=torch.long).unsqueeze(1)).float()


def expected_inputs(so):
    """x1, x2, x3 of one StateObject restated from the layout documented in encoder_conv, seats relative to
    so.player (0 = me, 1 = shimocha, 2 = toimen, 3 = kamicha). x1 planes: 0 own hand, 1 unseen counts, 2 own
    discards, 3 drawn tile, 4-6 opponents' discards, 7-9 tiles safe against each opponent. x2 rows: scores/250,
    dealer, dealer rounds left/2, double riichi, riichi, ippatsu, menzen. x3: remaining_tsumo/70, kyotaku/4,
    honba/8, rounds_remaining/8."""
    seats = [(so.player + d) % 4 for d in range(4)]
    drawn = np.zeros(37)
    if so.drawn_tile is not None:
        drawn[so.drawn_tile] = 1
    planes = [so.hand_tensor, so.remaining_tiles_pov, so.sutehai_tensors[seats[0]], drawn]
    planes += [so.sutehai_tensors[q] for q in seats[1:]] + [so.safe_tiles[q] for q in seats[1:]]
    rows = [(so.scores, 250.0), (so.parent_tensor, 1.0), (so.parent_rounds_remaining, 2.0), (so.double_reaches, 1.0),
            (so.reaches, 1.0), (so.ippatsu, 1.0), (so.is_menzen, 1.0)]
    x1 = torch.stack([thermometer(c) for c in planes])
    x2 = torch.tensor([[float(v[q]) / scale for q in seats] for v, scale in rows])
    x3 = torch.tensor([so.remaining_tsumo / 70.0, so.kyotaku / 4.0, so.honba / 8.0, so.rounds_remaining / 8.0])
    return x1, x2, x3


def answered_token(halfturns, i, tokens):
    """The exact token PostTurn halfturns[i] answers, from the halfturn before it: the preceding DiscardTurn's own
    discard token (same seat, same 37-slot tile, right before the PostTurn's history end), or the token of the
    kakan the preceding DuringTurn executed. None when the PostTurn lines up with neither."""
    t, prev = halfturns[i], halfturns[i - 1] if i else None
    if isinstance(prev, DiscardTurn) and prev.player == t.player and t.encoding_idx == prev.encoding_idx + 1:
        tok = tokens[prev.encoding_idx]
        if _is_discard(tok) and _who(tok) == t.player and _tile(tok) == TILE2IDX[prev.discarded_tile][-1]:
            return tok
    if isinstance(prev, DuringTurn) and prev.player == t.player and t.encoding_idx > 0:
        kakan = [d for d in prev.decisions[DECISION_NAKI_IDX] if d.executed and d.naki.is_kakan()]
        if len(kakan) == 1 and tokens[t.encoding_idx - 1] == naki2idx(t.player, kakan[0].naki):
            return tokens[t.encoding_idx - 1]
    return None


def expected_responders(t: PostTurn):
    """Responders that must be post samples, in turn order, with the exact target of each: executors always;
    a non-executor only when no other responder executed something of higher priority than ANY action this
    responder was offered (ron 3 > pon/kan 2 > chi 1), because it may have chosen the weaker one and lost.
    This restates the rule of TransformerTensorProcessor._post_turn_responders, so it catches a regression
    of that code, not a wrong rule; the XML-count gates (tensor_ron == xml_ron, ...) are the independent check."""
    lowest, done, target = {}, {}, {}
    for p in range(4):
        decs = t.decisions[p]
        kinds = [(3, d.executed, 1) for d in decs[DECISION_AGARI_IDX]]
        kinds += [(1 if d.naki.is_chi() else 2, d.executed, d.naki.get_post_turn_filter_idx()) for d in decs[DECISION_NAKI_IDX]]
        if not kinds:
            continue
        lowest[p] = min(k for k, _, _ in kinds)
        done[p] = max((k for k, x, _ in kinds if x), default=0)
        target[p] = next((idx for _, x, idx in kinds if x), 0)
    top = max(done.values(), default=0)
    return [(p, target[p]) for p in ((t.player + k) % 4 for k in (1, 2, 3)) if p in lowest and (done[p] or top <= lowest[p])]


def expected_during_target(t: DuringTurn) -> int:
    """The executed action of a DuringTurn in the 71-way layout: 1 tsumo, 2 riichi, 3-70 ankan / kakan, else 0."""
    if any(d.executed for d in t.decisions[DECISION_AGARI_IDX]):
        return 1
    if any(d.executed for d in t.decisions[DECISION_REACH_IDX]):
        return 2
    return next((d.naki.get_during_turn_filter_idx() for d in t.decisions[DECISION_NAKI_IDX] if d.executed), 0)


def _attr(attrs, name):
    return re.search(rf'\b{name}="([^"]*)"', attrs).group(1)


def xml_rounds(raw: str):
    """Per round, replayed from the XML alone, in log order: end ('agari' | 'exhaustive' | 'abortive'), the INIT
    dora indicator, the seats of the riichi declarations (REACH step 1) in order, and the history tokens the
    parser must emit: the INIT indicator, then DORA -> new dora, REACH step 1 -> reach, N -> call, D/E/F/G ->
    discard (tsumogiri iff it is the tile just drawn); nothing after the round's first AGARI / RYUUKYOKU."""
    rounds, acquired = [], None
    for m in XML_EVENT.finditer(raw):
        tag, attrs = m.group(1), m.group(2)
        if tag == "INIT":
            dora = int(_attr(attrs, "seed").split(",")[5])
            rounds.append({"end": None, "dora": dora, "reach": [], "tokens": [new_dora2idx(dora // 4)]})
            acquired = None
            continue
        r = rounds[-1]
        if r["end"] is not None:
            continue
        if tag == "AGARI":
            r["end"] = "agari"
        elif tag == "RYUUKYOKU":
            r["end"] = "abortive" if "type=" in attrs else "exhaustive"
        elif tag == "DORA":
            r["tokens"].append(new_dora2idx(int(_attr(attrs, "hai")) // 4))
        elif tag == "REACH":
            if _attr(attrs, "step") == "1":
                r["reach"].append(int(_attr(attrs, "who")))
                r["tokens"].append(reach2idx(int(_attr(attrs, "who"))))
        elif tag == "N":
            r["tokens"].append(naki2idx(int(_attr(attrs, "who")), Naki(int(_attr(attrs, "m")))))
            acquired = None
        elif tag[0] in "TUVW":
            acquired = int(tag[1:])
        else:
            tile = int(tag[1:])
            r["tokens"].append(discard2idx("DEFG".index(tag[0]), TILE2IDX[tile][-1], tile == acquired))
            acquired = None
    return rounds


def check_state(so, C, tp):
    """Built-tensor invariants of one StateObject: plane count, actor-first rows, input scale."""
    x1, x2, x3 = tp._convert_state_obj_to_tensors(so)
    me = so.player
    if x1.shape[1] != N_X1_PLANES:
        C["x1_plane_count"] += 1
    if x2[0, 4, 0].item() != so.reaches[me] or x2[0, 1, 0].item() != so.parent_tensor[me]:
        C["x2_not_actor_first"] += 1
    if max(x2.abs().max().item(), x3.abs().max().item()) > 5:
        C["x_scale_over_5"] += 1


def check_samples(head, tt, expected, tokens, C):
    """EVERY input of every built sample of one head against the StateObject and the history it was built from.
    expected: per sample, in build order, (target, state, history length, actor, answered token or None)."""
    enc, (x1, x2, x3), mask, y, pad = tt
    for i, (target, so, eidx, actor, answer) in enumerate(expected):
        if y[i].item() != target:
            C[TARGET_GATE[head]] += 1
        w1, w2, w3 = expected_inputs(so)
        if not torch.equal(x1[i, 0], w1[0]):
            C["hand_plane_mismatch"] += 1
        if not torch.equal(x1[i, 1:], w1[1:]):
            C["x1_plane_mismatch"] += 1
        if not torch.allclose(x2[i], w2, rtol=0, atol=1e-6):
            C["x2_row_mismatch"] += 1
        if not torch.allclose(x3[i], w3, rtol=0, atol=1e-6):
            C["x3_scalar_mismatch"] += 1
        hist = enc[i].tolist()
        if hist != rotate(tokens[:eidx], actor) + [0] * (len(hist) - eidx):
            C["history_prefix_mismatch"] += 1
        if pad[i].tolist() != [j < eidx for j in range(len(hist))]:
            C["padding_mask_mismatch"] += 1
        if head == "discard":
            hand = so.hand_tensor
            for red, plain in RED_TO_PLAIN.items():
                if mask[i, plain].item() != float(hand[plain] - hand[red] > 0) or mask[i, red].item() != float(hand[red] > 0):
                    C["discard_mask_red_rule"] += 1
        if head == "post" and (answer is None or eidx == 0 or hist[eidx - 1] != rotate([answer], actor)[0]):
            C["post_history_last_token"] += 1


def count_game(path, H, T, C, tp):
    """Add every counter of one parsed game (H = halfturns per round, T = tokens per round) to C."""
    raw = open(path, "rb").read().decode("utf-8", "ignore")
    ag = re.findall(r'<AGARI [^>]*?who="(\d)"[^>]*?fromWho="(\d)"', raw)
    C["xml_tsumo"] += sum(1 for w, fw in ag if w == fw)
    C["xml_ron"] += sum(1 for w, fw in ag if w != fw)
    C["xml_ron"] += 3 * len(re.findall(r'<RYUUKYOKU [^>]*type="ron3"', raw))  # triple ron: three rons, no AGARI
    for m in re.findall(r'<N who="\d" m="(\d+)"', raw):
        C["xml_" + kind_of(Naki(int(m)))] += 1
    rounds = xml_rounds(raw)
    assert len(rounds) == len(H), f"{path}: {len(rounds)} rounds in XML, {len(H)} parsed"

    for xr, halfturns, tokens in zip(rounds, H, T):
        C["rounds"] += 1
        if not (_event(tokens[0]) == EventStateTypes.NEW_DORA and _tile(tokens[0]) == xr["dora"] // 4):
            C["first_token_not_dora"] += 1
        if list(tokens) != xr["tokens"]:
            C["token_xml_mismatch"] += 1
        riichi = [t.player for t in halfturns
                  if isinstance(t, DuringTurn) and any(d.executed for d in t.decisions[DECISION_REACH_IDX])]
        C["xml_riichi"] += len(xr["reach"])
        C["lab_riichi"] += len(riichi)
        if riichi != xr["reach"]:
            C["riichi_label_mismatch"] += 1
        durings = [t for t in halfturns if isinstance(t, DuringTurn)]
        if xr["end"] == "exhaustive" and durings and durings[-1].stateObj.remaining_tsumo != 0:
            C["exhaustive_draw_tsumo_not_zero"] += 1
        if shared_buffers(halfturns):
            C["state_shares_buffer"] += 1
        tok_t = torch.tensor(tokens, dtype=torch.int64)
        for actor in range(4):
            if tp._relative_seats(tok_t, actor).tolist() != rotate(tokens, actor):
                C["rotation_mismatch"] += 1

        facts = {}  # (history length, answering) -> history_facts: states with the same prefix share it
        for so, hist, answering, discarding in model_states(halfturns, tokens):
            key = (len(hist), answering)
            if key not in facts:
                facts[key] = history_facts(hist, answering)
            check_history_fields(so, facts[key], discarding, C)
            check_bounds(so, C)

        exp = {h: [] for h in HEADS}  # per head, in build order: (target, state, history length, actor, answered)
        for i, t in enumerate(halfturns):
            so = t.stateObj
            if isinstance(t, DuringTurn):
                if sum(map(len, t.decisions)):
                    exp["during"].append((expected_during_target(t), so, t.encoding_idx, t.player, None))
                if t.decisions[DECISION_AGARI_IDX] and t.decisions[DECISION_AGARI_IDX][0].executed:
                    C["lab_tsumo"] += 1
                for d in t.decisions[DECISION_NAKI_IDX]:
                    if d.executed:
                        C["lab_" + kind_of(d.naki)] += 1
                check_state(so, C, tp)

            elif isinstance(t, DiscardTurn):
                if not (so.reaches[t.player] and not t.is_riichi_declaration):
                    exp["discard"].append((TILE2IDX[t.discarded_tile][-1], so, t.encoding_idx, t.player, None))
                if t.is_riichi_declaration:
                    prev = tokens[t.encoding_idx - 1]
                    if not (so.reaches[t.player] == 1 and _is_reach(prev) and _who(prev) == t.player):
                        C["riichi_declaration_unflagged"] += 1
                elif so.reaches[t.player]:
                    tok = tokens[t.encoding_idx]  # the discard token itself
                    drawn = so.drawn_tile
                    if not (_is_discard(tok) and _who(tok) == t.player and (tok >> 7) & 1 == 1
                            and drawn is not None and _tile(tok) == drawn[-1]):
                        C["riichi_discard_not_tsumogiri"] += 1
                check_state(so, C, tp)

            elif isinstance(t, PostTurn):
                check_bounds(so, C)  # the discarder's debug snapshot; the responder states are checked above
                answer = answered_token(halfturns, i, tokens)
                if answer is None:
                    C["post_history_last_token"] += 1
                for p in range(4):
                    if sum(map(len, t.decisions[p])) == 0:
                        continue
                    C["post_samples_counted"] += 1
                    rs = t.responder_states[p]
                    if rs.player != p:
                        C["post_wrong_pov"] += 1
                    tok = tokens[t.encoding_idx - 1] if t.encoding_idx > 0 else 0
                    is_disc = _is_discard(tok) and _who(tok) == t.player
                    is_kakan = _event(tok) == EventStateTypes.DURING and _etype(tok) == EventTypes.NAKI and _who(tok) == t.player
                    if not (is_disc or is_kakan):
                        C["post_answered_event_hidden"] += 1
                    if t.decisions[p][DECISION_AGARI_IDX] and t.decisions[p][DECISION_AGARI_IDX][0].executed:
                        C["lab_ron"] += 1
                    for d in t.decisions[p][DECISION_NAKI_IDX]:
                        if d.executed:
                            C["lab_" + kind_of(d.naki)] += 1
                    check_state(rs, C, tp)
                for p, target in expected_responders(t):
                    exp["post"].append((target, t.responder_states[p], t.encoding_idx, p, answer))

        for p, called, first, disc, nxt in ippatsu_windows(halfturns, tokens):
            C["ippatsu_windows"] += 1
            if first.stateObj.ippatsu[p] != (0 if called else 1):
                C["ippatsu_window_wrong"] += 1
            if any(t is not None and t.stateObj.ippatsu[p] != 0 for t in (disc, nxt)):
                C["ippatsu_window_wrong"] += 1

        try:
            td, tdisc, tpost = tp.build_tensors(halfturns, tokens)
        except Exception as e:
            C["tensor_build_error"] += 1
            print(f"tensor build error in {path}: {type(e).__name__}: {str(e)[:200]}")
            continue
        C["tensor_skipped"] = sum(tp.skipped.values())
        for name, tt in zip(HEADS, (td, tdisc, tpost)):
            if len(tt[3]):
                C[f"{name}_label_outside_mask"] += int((tt[2][np.arange(len(tt[3])), tt[3]] != 1).sum().item())
                C[f"{name}_samples"] += len(tt[3])
            if len(tt[3]) != len(exp[name]):
                C[f"{name}_sample_count_mismatch"] += 1
            else:
                check_samples(name, tt, exp[name], tokens, C)
        tensor_counts(td[3].tolist(), tpost[3].tolist(), C)


def tensor_counts(y_during, y_post, C):
    """Count the built targets per action kind (the head layouts: during 71, post 154)."""
    C["tensor_tsumo"] += sum(1 for v in y_during if v == 1)
    C["tensor_riichi"] += sum(1 for v in y_during if v == 2)
    C["tensor_ankan"] += sum(1 for v in y_during if 3 <= v <= 36)
    C["tensor_kakan"] += sum(1 for v in y_during if 37 <= v <= 70)
    C["tensor_ron"] += sum(1 for v in y_post if v == 1)
    C["tensor_chi"] += sum(1 for v in y_post if 2 <= v <= 82)
    C["tensor_pon"] += sum(1 for v in y_post if 83 <= v <= 119)
    C["tensor_minkan"] += sum(1 for v in y_post if 120 <= v <= 153)


def index_checks(C):
    """Every chi/pon the mask can build must have its own post-turn action index, and chi never lands on pon slots."""
    idx_map = collections.defaultdict(set)
    for color in range(3):
        for number in range(7):
            for which in range(3):
                for red in (False, True):
                    if red and not (number <= 4 <= number + 2):
                        continue
                    nk = Naki.from_chi_info(color, number, which, red)
                    idx_map[nk.get_post_turn_filter_idx()].add(("chi", color, number, which, red and number + which != 4))
    for tile in range(34):
        for which in (0, 1):
            for red in (False, True):
                if red and not (tile < 27 and tile % 9 == 4):
                    continue
                nk = Naki.from_pon_info(tile, which, red, 1)
                _, _, _, has_red, _, acq = nk.pattern_pon()
                idx_map[nk.get_post_turn_filter_idx()].add(("pon", tile, bool(has_red and acq % 4 != 0)))
    C["index_collisions"] += sum(1 for v in idx_map.values() if len(v) > 1)
    C["index_spill_into_pon"] += sum(1 for k, v in idx_map.items() if 83 <= k <= 119 and any(x[0] == "chi" for x in v))


def run(files, tp=None):
    tp = tp or TransformerTensorProcessor()
    C = collections.Counter()
    for f in files:
        try:
            H, T = process(f)
        except InvalidGameException as e:
            C["skipped_invalid"] += 1
            continue
        except Exception as e:
            C["parse_error"] += 1
            print(f"parse error in {f}: {type(e).__name__}: {str(e)[:200]}")
            continue
        C["games"] += 1
        count_game(f, H, T, C, tp)
    index_checks(C)
    return C


def failed_gates(C) -> dict:
    """{gate: offending value} for every gate that fails; empty when all pass."""
    bad = {}
    if C["games"] == 0:
        bad["games"] = "no game parsed"
    for k in LABELLED:
        for side in ("lab", "tensor"):
            if C[f"{side}_{k}"] != C[f"xml_{k}"]:
                bad[f"{side}_{k}"] = f"{C[f'{side}_{k}']} != xml_{k}={C[f'xml_{k}']}"
    for h in HEADS:
        if C[f"{h}_samples"] == 0:
            bad[f"{h}_samples"] = 0
        if C[f"{h}_label_outside_mask"]:
            bad[f"{h}_label_outside_mask"] = C[f"{h}_label_outside_mask"]
    bad.update({k: C[k] for k in ZERO_GATES if C[k] != 0})
    return bad


def check(C):
    """Raise AssertionError naming every failed gate; print one line when every gate passes."""
    bad = failed_gates(C)
    assert not bad, f"gates failed: {bad}"
    print(
        f"GATES OK | games {C['games']} (skipped {C['skipped_invalid']}) rounds {C['rounds']} | "
        f"wins tsumo {C['lab_tsumo']}/{C['xml_tsumo']} ron {C['lab_ron']}/{C['xml_ron']} | "
        f"riichi {C['lab_riichi']}/{C['xml_riichi']} | "
        f"calls " + " ".join(f"{k} {C['lab_' + k]}/{C['xml_' + k]}" for k in KINDS) + " | "
        f"samples " + " ".join(f"{h} {C[h + '_samples']}" for h in HEADS) + " | tensors == parser == xml"
    )


def _first_valid_game():
    for f in [SAMPLE_LOG] + sorted(glob.glob(os.path.join(DATA_ROOT, "2021", "20210101", "*", "*.log")))[:20]:
        try:
            process(f)
            return f
        except Exception:
            continue
    raise RuntimeError("no valid game found for the self-test")


def _game_with(pred, limit=60):
    """First 2021-01-01 game whose parsed rounds satisfy pred(H, T)."""
    for f in sorted(glob.glob(os.path.join(DATA_ROOT, "2021", "20210101", "*", "*.log")))[:limit]:
        try:
            H, T = process(f)
        except Exception:
            continue
        if pred(H, T):
            return f
    raise RuntimeError("no game found for the self-test predicate")


def _named_game(fixture):
    """A named fixture game (day, file, regex): the self-test pool and the default --games sample lack its event."""
    day, name, pattern = fixture
    found = glob.glob(os.path.join(DATA_ROOT, "2021", day, "*", name))
    assert found, f"self-test needs the game {name} under {DATA_ROOT}/2021/{day}"
    raw = open(found[0], "rb").read().decode("utf-8", "ignore")
    assert re.search(pattern, raw), f"{found[0]} does not match {pattern}"
    return found[0]


def _count(game, H=None, T=None, tp=None):
    """Every counter of one game, as run() would count it."""
    tp = tp or TransformerTensorProcessor()
    if H is None:
        H, T = process(game)
    C = collections.Counter(games=1)
    count_game(game, H, T, C, tp)
    index_checks(C)
    return C


def _states(H):
    """Every distinct StateObject of a game: each halfturn's stateObj and every PostTurn responder state."""
    seen = {}
    for halfturns in H:
        for t in halfturns:
            for so in [t.stateObj] + list(getattr(t, "responder_states", {}).values()):
                seen.setdefault(id(so), so)
    return list(seen.values())


def _patch_tensors(tp, edit):
    """Corrupt the BUILT tensors: edit(td, tdisc, tpost) runs on every build_tensors output."""
    orig = tp.build_tensors

    def bad(halfturns, tokens):
        out = orig(halfturns, tokens)
        edit(*out)
        return out

    tp.build_tensors = bad


def _patch_inputs(tp, edit):
    """Corrupt every converted state: edit(x1, x2, x3) runs on clones of the converter output."""
    orig = tp._convert_state_obj_to_tensors

    def bad(so):
        x1, x2, x3 = (x.clone() for x in orig(so))
        edit(x1, x2, x3)
        return x1, x2, x3

    tp._convert_state_obj_to_tensors = bad


def self_test():
    """Clean games pass every gate; each corruption, applied to a game that passes every gate, must fail its NAMED
    gate (another gate failing as well is allowed). Also: a one-loader cluster rolls over at the epoch boundary."""
    f = _first_valid_game()
    ron3 = _named_game(RON3_LOG)
    red_kan_dora = _named_game(RED_KAN_DORA_LOG)
    chankan_ippatsu = _named_game(CHANKAN_IPPATSU_LOG)
    early_ankan = _named_game(EARLY_ANKAN_LOG)
    check(run([f, ron3, red_kan_dora, chankan_ippatsu, early_ankan], TransformerTensorProcessor()))

    def expect_failure(name, gate, corrupt, game=f):
        baseline = failed_gates(_count(game))
        if baseline:
            raise SystemExit(f"self-test FAILED: {name}: the clean game {game} already fails {baseline}")
        tp = TransformerTensorProcessor()
        H, T = process(game)
        corrupt(H, T, tp)
        bad = failed_gates(_count(game, H, T, tp))
        if gate not in bad:
            raise SystemExit(f"self-test FAILED: {name} did not fail gate {gate} (failed: {bad or 'none'})")
        print(f"self-test OK: {name} -> {gate}={bad[gate]}")

    def found(n, what):
        if not n:
            raise SystemExit(f"self-test needs {what}")

    # ---- parser labels
    def unlabel_ron(H, T, tp):
        for t in (t for halfturns in H for t in halfturns if isinstance(t, PostTurn)):
            for p in range(4):
                a = t.decisions[p][DECISION_AGARI_IDX]
                if a and a[0].executed:
                    a[0].executed = False
                    return
        found(0, "a game with a ron")

    def skip_ron3_marking(H, T, tp):  # re-parse with the triple-ron labelling switched off
        with mock.patch.object(process_xml, "_mark_ron3_executed", lambda halfturns: None):
            H[:], T[:] = process(ron3)

    def erase_riichi_labels(H, T, tp):
        n = 0
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn)):
            for d in t.decisions[DECISION_REACH_IDX]:
                n += d.executed
                d.executed = False
        found(n, "a game with a riichi")

    def has_riichi(H, T):
        return any(isinstance(t, DiscardTurn) and t.is_riichi_declaration for halfturns in H for t in halfturns)

    def flip_tsumogiri_bit(H, T, tp):
        i = next(i for i, tok in enumerate(T[0]) if _is_discard(tok))
        T[0][i] ^= 1 << 7

    # ---- POV / buffers
    def wrong_pov(H, T, tp):
        for t in (t for halfturns in H for t in halfturns if isinstance(t, PostTurn)):
            for p in range(4):
                if sum(map(len, t.decisions[p])):
                    t.responder_states[p].player = (p + 1) % 4
                    return

    def responder_unseen_negative(H, T, tp):  # a responder state with NO decision: only the bounds sweep sees it
        for t in (t for halfturns in H for t in halfturns if isinstance(t, PostTurn)):
            for p, rs in t.responder_states.items():
                if not sum(map(len, t.decisions[p])):
                    rs.remaining_tiles_pov[0] = -1
                    return
        found(0, "a PostTurn responder without a decision")

    def shared_pov_buffer(H, T, tp):  # the later of two DuringTurn states views the earlier one's unseen counts
        durings = [t for t in H[0] if isinstance(t, DuringTurn)]
        durings[-1].stateObj.remaining_tiles_pov = durings[0].stateObj.remaining_tiles_pov[:]

    # ---- history fields: safe tiles, ippatsu, double riichi, red dora
    def stale_safe_plane(H, T, tp):
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DiscardTurn)):
            t.stateObj.safe_tiles[(t.player + 1) % 4, 0] = 1.0
            return

    def answered_marked_safe_target(H, T):  # a responder state, a riichi player q != discarder, the answered kind
        for halfturns, tokens in zip(H, T):
            for t in (t for t in halfturns if isinstance(t, PostTurn)):
                tok = tokens[t.encoding_idx - 1]
                kind = RED_TO_PLAIN.get(_tile(tok), _tile(tok))
                for rs in t.responder_states.values():
                    for q in range(4):
                        if _is_discard(tok) and q != t.player and rs.reaches[q] and not rs.safe_tiles[q, kind]:
                            return rs, q, kind
        return None

    def answered_marked_safe(H, T, tp):  # the old timing: the discard is safe before its responders answer it
        rs, q, kind = answered_marked_safe_target(H, T)
        rs.safe_tiles[q, kind] = 1.0
        if kind in PLAIN_TO_RED:
            rs.safe_tiles[q, PLAIN_TO_RED[kind]] = 1.0

    def safe_five_target(H, T):
        return next(((so, q, plain) for so in _states(H) for q in range(4) for plain in PLAIN_TO_RED
                     if so.safe_tiles[q, plain]), None)

    def safe_five_without_red(H, T, tp):  # the old rule: a plain five made only its plain slot safe
        so, q, plain = safe_five_target(H, T)
        so.safe_tiles[q, PLAIN_TO_RED[plain]] = 0.0

    def early_ippatsu_clear(H, T, tp):  # the old order: the first draw after riichi cleared ippatsu before its snapshot
        for halfturns, tokens in zip(H, T):
            for p, called, first, _, _ in ippatsu_windows(halfturns, tokens):
                if not called:
                    first.stateObj.ippatsu = [0 if q == p else v for q, v in enumerate(first.stateObj.ippatsu)]
                    return
        found(0, "a riichi followed by the declarer's draw")

    def has_ippatsu_window(H, T):
        return any(not called for halfturns, tokens in zip(H, T) for _, called, *_ in ippatsu_windows(halfturns, tokens))

    def chankan_ippatsu_cleared(H, T, tp):  # the old order: a kakan cleared ippatsu before its chankan window
        n = 0
        for halfturns, tokens in zip(H, T):
            for t in (t for t in halfturns if isinstance(t, PostTurn) and _is_call(tokens[t.encoding_idx - 1])):
                for rs in t.responder_states.values():
                    if any(rs.ippatsu):
                        rs.ippatsu = [0] * 4
                        n += 1
        found(n, "a kakan while a riichi has ippatsu")

    def old_double_riichi_rule(H, T, tp):  # re-parse with `cycles <= 4 and min(is_menzen)`: true after an early ankan
        old = lambda uninterrupted, own_discards, cycles, is_menzen: bool(cycles <= 4 and min(is_menzen))
        with mock.patch.object(process_xml, "_double_riichi", old):
            H[:], T[:] = process(early_ankan)

    def red_kan_dora_kept(H, T, tp):  # the old DORA bug: a red kan-dora indicator decremented the kind slot only
        n = 0
        for so in _states(H):
            for s in range(3):
                if 16 + 36 * s in so.dora[1:]:  # dora[0] is the INIT indicator
                    so.remaining_tiles_pov = so.remaining_tiles_pov.copy()
                    so.remaining_tiles_pov[34 + s] += 1
                    n += 1
        found(n, "a red-five kan-dora indicator (a DORA event, not INIT)")

    # ---- built tensors
    def zero_ron_targets(H, T, tp):  # parser labels intact, built targets lose every ron
        _patch_tensors(tp, lambda td, tdisc, tpost: tpost[3].masked_fill_(tpost[3] == 1, 0))

    def zero_riichi_targets(H, T, tp):
        _patch_tensors(tp, lambda td, tdisc, tpost: td[3].masked_fill_(td[3] == 2, 0))

    def other_chi_target(H, T, tp):  # an executed chi target moved to ANOTHER enabled chi slot: same kind count
        def edit(td, tdisc, tpost):
            for i in range(len(tpost[3])):
                y = tpost[3][i].item()
                others = [j for j in range(2, 83) if tpost[2][i, j] == 1 and j != y]
                if 2 <= y <= 82 and others:
                    tpost[3][i] = others[0]
                    break
        _patch_tensors(tp, edit)

    def has_chi_choice(H, T):  # a round where an executed chi had a second chi option
        tp = TransformerTensorProcessor()
        for halfturns, tokens in zip(H, T):
            _, _, tpost = tp.build_tensors(halfturns, tokens)
            for i in range(len(tpost[3])):
                y = tpost[3][i].item()
                if 2 <= y <= 82 and int(tpost[2][i, 2:83].sum().item()) >= 2:
                    return True
        return False

    def keep_preempted(H, T, tp):  # the old rule: every responder with a decision is a sample
        tp._post_turn_responders = lambda turn: [
            p for p in ((turn.player + k) % 4 for k in (1, 2, 3)) if sum(map(len, turn.decisions[p]))
        ]

    def has_preempted(H, T):
        return any(
            len(expected_responders(t)) < sum(1 for p in range(4) if sum(map(len, t.decisions[p])))
            for halfturns in H for t in halfturns if isinstance(t, PostTurn)
        )

    def shifted_hand_plane(H, T, tp):  # every sample's hand plane is rolled by one tile
        _patch_inputs(tp, lambda x1, x2, x3: x1[:, 0].copy_(x1[:, 0].roll(1, dims=1)))

    def zero_unseen_plane(H, T, tp):  # x1 plane 1 (unseen counts) zeroed
        _patch_inputs(tp, lambda x1, x2, x3: x1[:, 1].zero_())

    def zero_score_row(H, T, tp):  # x2 row 0 (scores) zeroed
        _patch_inputs(tp, lambda x1, x2, x3: x2[:, 0].zero_())

    def zero_scalars(H, T, tp):
        _patch_inputs(tp, lambda x1, x2, x3: x3.zero_())

    def truncate_history(H, T, tp):  # post histories keep only the answered token
        def edit(td, tdisc, tpost):
            for i in range(len(tpost[3])):
                n = int(tpost[4][i].sum().item())
                tpost[0][i, : n - 1] = 0
        _patch_tensors(tp, edit)

    def full_padding_mask(H, T, tp):  # every position of every discard sample marked valid
        _patch_tensors(tp, lambda td, tdisc, tpost: tdisc[4].fill_(True))

    def stale_answered_token(H, T, tp):  # a PostTurn history stops one token early: it answers the event before
        t = next(t for halfturns in H for t in halfturns if isinstance(t, PostTurn) and expected_responders(t))
        t.encoding_idx -= 1

    expect_failure("un-labelled ron", "lab_ron", unlabel_ron)
    expect_failure("triple ron left unlabelled", "lab_ron", skip_ron3_marking, game=ron3)
    riichi_game = f if has_riichi(*process(f)) else _game_with(has_riichi)
    expect_failure("executed riichi labels erased", "riichi_label_mismatch", erase_riichi_labels, game=riichi_game)
    expect_failure("tsumogiri bit flipped in the token list", "token_xml_mismatch", flip_tsumogiri_bit)
    expect_failure("wrong responder POV", "post_wrong_pov", wrong_pov)
    expect_failure("responder unseen count -1", "pov_negative", responder_unseen_negative)
    expect_failure("two states share an unseen-count buffer", "state_shares_buffer", shared_pov_buffer)
    expect_failure("stale safe plane", "safe_plane_mismatch", stale_safe_plane)
    expect_failure("answered discard already safe (old timing)", "safe_plane_mismatch", answered_marked_safe,
                   game=_game_with(lambda H, T: answered_marked_safe_target(H, T) is not None))
    expect_failure("safe five without its red slot", "safe_plane_mismatch", safe_five_without_red,
                   game=_game_with(lambda H, T: safe_five_target(H, T) is not None))
    expect_failure("ippatsu cleared before the first draw after riichi", "ippatsu_window_wrong", early_ippatsu_clear,
                   game=_game_with(has_ippatsu_window))
    expect_failure("ippatsu cleared before the chankan window", "ippatsu_history_mismatch", chankan_ippatsu_cleared,
                   game=chankan_ippatsu)
    expect_failure("double riichi by the old cycles/menzen test after an ankan", "double_riichi_mismatch",
                   old_double_riichi_rule, game=early_ankan)
    expect_failure("red kan-dora indicator left in the red slot", "red_pov_mismatch", red_kan_dora_kept, game=red_kan_dora)
    expect_failure("zeroed ron targets", "tensor_ron", zero_ron_targets)
    expect_failure("zeroed riichi targets", "tensor_riichi", zero_riichi_targets, game=riichi_game)
    expect_failure("chi target moved to another chi", "post_target_mismatch", other_chi_target, game=_game_with(has_chi_choice))
    expect_failure("pre-empted responder kept", "post_sample_count_mismatch", keep_preempted, game=_game_with(has_preempted))
    expect_failure("shifted hand plane", "hand_plane_mismatch", shifted_hand_plane)
    expect_failure("zeroed unseen-count plane", "x1_plane_mismatch", zero_unseen_plane)
    expect_failure("zeroed score row", "x2_row_mismatch", zero_score_row)
    expect_failure("zeroed x3 scalars", "x3_scalar_mismatch", zero_scalars)
    expect_failure("post history truncated to the answered token", "history_prefix_mismatch", truncate_history)
    expect_failure("padding mask all valid", "padding_mask_mismatch", full_padding_mask)
    expect_failure("post sample answers the token before its event", "post_history_last_token", stale_answered_token)

    from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster
    dl = DataLoader(os.path.dirname(f), batch_size=1, log_func=lambda *_: None)
    dl.file_list = dl.file_list[:1] or [f]
    cluster = DataLoaderCluster([dl])
    for _ in range(3):  # 1 game, 3 batches: two epoch boundaries
        games, _tokens = next(cluster)
        assert games, "loader rollover returned an empty batch"
    print("self-test OK: one-loader cluster rolls over at the epoch boundary")
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="run the self-test and exit")
    ap.add_argument("--games", type=int, default=150, help="number of random games to check")
    ap.add_argument("--year", default="2021")
    ap.add_argument("--days", nargs="*", default=DEFAULT_DAYS, help="day directories to sample from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print every counter")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit(0)
    files = [f for d in args.days for f in sorted(glob.glob(os.path.join(DATA_ROOT, args.year, d, "*", "*.log")))]
    assert files, f"no games under {DATA_ROOT}/{args.year}/{args.days}"
    files = random.Random(args.seed).sample(files, min(args.games, len(files)))
    C = run(files)
    if args.verbose:
        for k in sorted(C):
            print(f"{k:34s} {C[k]}")
    check(C)
