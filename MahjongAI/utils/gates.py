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
  MASK    every label index is inside its action mask (during 72, discard 37, post 154); chi/pon indices injective
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
          all 7 actor-first x2 rows, all 7 x3 slots, the full rotated history prefix with zeros after it, and
          the padding mask; the post sample set follows the priority rule (executors always, pre-empted
          responders never); the last history token of a post sample is exactly the discard / kakan it
          answers; the tensor builder skipped nothing
  LEGAL   the discard mask == held tiles (red-five split) minus the illegal kinds, both slots of a five, with
          the context and the rules rebuilt WITHOUT utils/legal.py, DiscardTurn.after_call or
          is_riichi_declaration (legal_oracles.py): a declaration (the seat's own reach token right before)
          keeps only the kinds that leave tenpai by Agari wait enumeration on the full hand; a post-call
          discard (the seat's own chi / pon token right before) loses the called kind and the suji kind, from
          the latest discard token and the two tiles that left the hand; any other discard == plain presence.
          The executed discard is always enabled, no unheld slot is, no mask is empty, and every riichi offer
          has at least one tenpai-keeping discard. Coverage: declaration samples == riichi labels and post-call
          samples == chi + pon labels; is_riichi_declaration == the token context, both ways
  STATE   every state a head reads (and every DiscardTurn state) == the round replayed from the XML alone
          (xml_snapshots): the 37-slot hand, every seat's melds, the drawn tile, the live draws left, the scores,
          the riichi seats and the menzen flags. The option oracles below read only that replay.
  OPTIONS the riichi / kan / kyuushu bits of every BUILT during mask and the call bits of every BUILT post mask == the
          Tenhou option rules rebuilt WITHOUT utils/legal.py (legal_oracles.py, on the replay): riichi needs
          menzen, no riichi yet, shanten <= 0, >= 1,000 points, >= 4 live draws and no suukaikan pending; no
          kan without a live draw or with 4 kans down; in riichi only the ankan of the drawn kind that keeps the
          waits (Shanten wait set); no call in a chankan window, by a riichi player, on the last discard, with
          suukaikan pending, no daiminkan with 4 kans down, no chi / pon that leaves no legal discard; kyuushu
          kyuuhai only on the seat's first draw with no call yet and >= 9 terminal / honor kinds. A turn /
          responder that is not a sample has its parser options compared instead. One named counter per rule
          on the same bits; "options removed" per rule are diagnostics
  DORA    Tenhou flips an ankan indicator at once, a daiminkan / kakan one only after the kan player's discard (or
          right before his next rinshan draw); the XML writes the open-kan DORA before that discard. Every state
          the XML replay checks has exactly the public indicators (state_xml_dora) and unseen counts == 4 (1 for a
          red slot) - own hand - every discard - meld tiles from a hand - public indicators, remaining_tiles == 4 -
          dealt and drawn tiles - public indicators (pov_xml_mismatch); rule-free: between a seat's draw decision
          and its discard decision no indicator appears (dora_draw_to_discard); Tenhou's own data: each <AGARI>'s
          doraHai == the indicators of the winner's state at the win decision (dora_vs_agari, with coverage); the
          parser defers exactly the XML's `T DORA D` triples (open_kan_dora_deferred)
  WIN     win options rebuilt WITHOUT utils/agari.py (win_oracles.py): every executed win's yaku, mapped to Tenhou
          ids, == the <AGARI> yaku (bonus ids dropped) or yakuman ids; for every PostTurn responder whose concealed
          hand the answered tile completes (direct Agari.is_agari), the parser's furiten verdict == the one rebuilt
          from the tokens (a wait among the seat's own discard tokens; a complete tile passed at an earlier
          PostTurn after the seat's latest discard token / after its reach token); a complete hand has an offer XOR
          a block reason, an incomplete one neither; every offer has a yaku; the built during / post win slot is
          on iff the sample offers a win
  KYUUSHU executed kyuushu aborts == <RYUUKYOKU type="yao9"> (lab / tensor / xml counts), and per round the
          executed abort is exactly the yao9 seat on its latest draw (seat, draw number), none without a yao9
  ROUND   the round-wind (bakaze) one-hot of every BUILT sample row of every head (x3 slots 4-6) == the round wind
          read from the XML INIT seed (seed[0] // 4: 0 East, 1 South, 2 West), never from agari.round_wind or
          the StateObject (round_wind_xml); every built row is checked (round_wind_coverage). A North round
          (seed[0] >= 12) is rejected by the parser and counted (skipped_invalid_north, path printed)
"""
import argparse
import collections
import contextlib
import glob
import os
import random
import re
import sys
from unittest import mock

import numpy as np
import torch
from mahjong.constants import EAST

from MahjongAI import process_xml
from MahjongAI.process_xml import process, InvalidGameException, discard2idx, naki2idx, reach2idx, new_dora2idx
from MahjongAI.turn import DuringTurn, DiscardTurn, PostTurn
from MahjongAI.draw import Naki
from MahjongAI.decision import AgariDecision, NakiDecision, ReachDecision, decision_mask
from MahjongAI.utils import agari, legal, legal_oracles, win_oracles
from MahjongAI.utils.constants import (DECISION_ABORT_IDX, DECISION_AGARI_IDX, DECISION_NAKI_IDX, DECISION_REACH_IDX,
                                       EventStateTypes, EventTypes, TILE2IDX)
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor, N_X1_PLANES, DURING_TURN_ACTION_DIM

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PACKAGE_DIR, "data", "processed")
SAMPLE_LOG = os.path.join(PACKAGE_DIR, "data", "sample", "sample_haifu.xml")
# Named self-test fixtures, (day, file, regex its XML must match): the 2021-01-01 pool lacks these events
RON3_LOG = ("20210105", "2021010521gm-00a9-0000-e76b4550.log", r'<RYUUKYOKU [^>]*type="ron3"')  # round 5: triple ron
RED_KAN_DORA_LOG = ("20210101", "2021010100gm-00a9-0000-3ceec6a2.log", r'<DORA hai="(16|52|88)"')  # red kan indicator
CHANKAN_IPPATSU_LOG = ("20210101", "2021010110gm-00a9-0000-85d17b74.log", r'<REACH [^>]*step="2"')  # kakan in ippatsu
EARLY_ANKAN_LOG = ("20210109", "2021010919gm-00a9-0000-b25465a8.log", r'<REACH [^>]*step="1"')  # ankan, then riichi
# T-032 fixtures (day, file, round): the option rules they exercise are rare (T-032 plan, Problem table)
LOW_POINTS_RIICHI = ("20210103", "2021010301gm-00a9-0000-ac500e01.log", 11)  # riichi tenpai below 1,000 points
WAIT_CHANGE_ANKAN = ("20210103", "2021010313gm-00a9-0000-5120e1eb.log", 10)  # drawn-kind ankan in riichi changes waits
KAN4_ROUND = ("20210214", "2021021419gm-00a9-0000-57c92bbe.log", 3)  # 4th kan by 2 players, then a pon-able discard
DEAD_END_CALL = ("20210322", "2021032214gm-00a9-0000-6d556c69.log", 12)  # chi that leaves only kuikae kinds
RIICHI_ANKAN = ("20210101", "2021010100gm-00a9-0000-17175841.log", None)  # an executed riichi ankan
# T-034 kyuushu fixtures. Named games: (day, file, regex); parsed fixtures: (day, file, round)
YAO9 = r'<RYUUKYOKU [^>]*type="yao9"'
YAO9_LOG = ("20210101", "2021010100gm-00a9-0000-c851c43f.log", YAO9)  # round 0 seat 0: exactly 9 kinds, aborted
KYUUSHU_AFTER_CALL_LOG = ("20210101", "2021010115gm-00a9-0000-52ecaf6b.log", YAO9)  # round 8 seat 0: 9 kinds after a
#   pon (no offer); round 6 seat 0 declines an offer; round 11 seat 2 aborts
NINTH_KIND_DRAWN_ABORT_LOG = ("20210101", "2021010101gm-00a9-0000-004c8ef1.log", YAO9)  # round 1 seat 2: 8 kinds + the draw
NINTH_KIND_DRAWN_DECLINED = ("20210101", "2021010102gm-00a9-0000-54704a83.log", 2)  # seat 3: 8 kinds + the draw, declined
KYUUSHU_DECLINED = ("20210101", "2021010100gm-00a9-0000-c8ce9c2b.log", 4)  # seat 3 declines an offer
# T-033 win fixtures. YAKU(i): an <AGARI> whose yaku list holds id i
def YAKU(i): return rf'<AGARI [^>]*\byaku="(\d+,\d+,)*{i},'
SEAT_WIND_HOUTEI_LOG = ("20210101", "2021010100gm-00a9-0000-1222e403.log", YAKU(12))  # Haku + seat W; Houtei + Tanyao
RINSHAN_LOG = ("20210101", "2021010101gm-00a9-0000-004c8ef1.log", YAKU(4))  # Haku + Rinshan; a no-yaku pass (round 2)
SUUANKOU_LOG = ("20210101", "2021010111gm-00a9-0000-5390436b.log", r'yakuman="([^"]*,)?4[01]\b')
CHIIHOU_LOG = ("20210219", "2021021911gm-00a9-0000-9ac4312b.log", r'yakuman="([^"]*,)?38\b')
TENHOU_LOG = ("20210406", "2021040609gm-00a9-0000-6d605f30.log", r'yakuman="([^"]*,)?37\b')
LAST_RINSHAN_TSUMO_LOG = ("20210117", "2021011702gm-00a9-0000-6c2bc28d.log", YAKU(4))  # rinshan on the 70th draw
LAST_RINSHAN_HOUTEI_LOG = ("20210102", "2021010221gm-00a9-0000-f90354ec.log", YAKU(6))  # houtei-only ron after it
TEMP_FURITEN_LOG = ("20210102", "2021010213gm-00a9-0000-5ed455ef.log", r'<AGARI ')
RIICHI_FURITEN_LOG = ("20210101", "2021010100gm-00a9-0000-3b235798.log", r'<REACH [^>]*step="1"')
# A passed chankan followed by a same-turn chance: no real log has one with the passer outside discard furiten (all of
# 2013-2023 scanned, 2026-10-15), so this game is synthetic: seat 1 (45m, waits 3m / 6m) passes seat 0's kakan of 3m,
# then seat 0 discards 6m from the rinshan draw before seat 1 discards again (Tenhou: same-turn furiten).
CHANKAN_PASS_GAME = os.path.join(PACKAGE_DIR, "data", "sample", "chankan_pass_synthetic.xml")
# T-035 fixtures: open-kan indicators. OPEN_KAN_DORA_RE = an open kan's DORA between its rinshan draw and discard
OPEN_KAN_DORA_RE = r'<[TUVW]\d+/><DORA hai="\d+"\s*/><[DEFG]\d+/>'
OPEN_KAN_RON_LOG = ("20210101", "2021010101gm-00a9-0000-6422c07e.log",  # red-five indicator, then a ron on that discard
                    r'<[TUVW]\d+/><DORA hai="16"\s*/><[DEFG]\d+/><AGARI ')
OPEN_KAN_LOG = ("20210101", "2021010105gm-00a9-0000-0e0fb57b.log",  # 2 kakan; the 1st indicator flips before the 2nd rinshan
                    r'<[TUVW]\d+/><N [^>]*/><DORA hai="\d+"\s*/><[TUVW]\d+/><DORA hai="\d+"\s*/><[DEFG]\d+/>')
PENDING_FLIP_TSUMO_LOG = ("20210102", "2021010214gm-00a9-0000-7ed48ecb.log",  # 2 kakan, rinshan tsumo counts the 1st
                          r'<N [^>]*/><DORA hai="31"\s*/><[TUVW]\d+/><AGARI [^>]*doraHai="121,31"')
OPEN_KAN_THEN_ANKAN_LOG = ("20210108", "2021010823gm-00a9-0000-725fdda5.log",  # daiminkan, ankan: both flip at once
                           r'<N [^>]*/><DORA hai="\d+"\s*/><DORA hai="\d+"\s*/><[TUVW]\d+/><[DEFG]\d+/>')
# T-038 bakaze fixtures: a tonpuu game that reaches South (E1-E4, S1, S2), and a hanchan game that reaches West
TONPUU_SOUTH_LOG = ("20210101", "2021010100gm-00e1-0000-4762443c.log", r'<GO type="225"[\s\S]*<INIT seed="[4-7],')
HANCHAN_WEST_LOG = ("20210101", "2021010100gm-00a9-0000-0af99d68.log", r'<GO type="169"[\s\S]*<INIT seed="(8|9|10|11),')
FULL_WALL = np.array([4] * 34 + [1] * 3)  # copies per 37-slot index (slots 34-36: the red fives)
DEFAULT_DAYS = ["20210103", "20210214", "20210322", "20210430", "20210518", "20210626",
                "20210709", "20210815", "20210921", "20211030", "20211112", "20211225"]
KINDS = ["chi", "pon", "minkan", "ankan", "kakan"]
LABELLED = ["tsumo", "ron", "riichi", "kyuushu"] + KINDS  # lab_<k> (parser) == tensor_<k> (built targets) == xml_<k>
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
    "discard_target_illegal", "discard_mask_unrestricted_mismatch", "discard_mask_superset", "discard_mask_empty",
    "riichi_mask_mismatch", "kuikae_mask_mismatch", "riichi_offer_without_legal_discard",
    "state_xml_mismatch", "during_mask_mismatch", "post_mask_mismatch",
    "riichi_offer_low_points", "riichi_offer_few_draws", "riichi_offer_after_kan4", "riichi_okurikan_offer",
    "riichi_ankan_wait_change_offer", "kan_offer_no_draw", "kan_offer_over_cap", "call_offer_last_discard",
    "call_offer_kan4_abort", "call_offer_in_chankan", "call_offer_no_legal_discard",
    "yaku_oracle_mismatch", "furiten_mismatch", "win_reason_conflict", "win_mask_mismatch", "win_offer_without_yaku",
    "ron_offer_incomplete", "tsumo_offer_incomplete",
    "kyuushu_offer_not_first_draw", "kyuushu_offer_after_call", "kyuushu_offer_few_kinds", "kyuushu_label_mismatch",
    "state_xml_dora", "pov_xml_mismatch", "dora_draw_to_discard", "dora_vs_agari",
    "round_wind_xml",
]
OPTION_RULES = ["riichi_offer_low_points", "riichi_offer_few_draws", "riichi_offer_after_kan4", "riichi_okurikan_offer",
                "riichi_ankan_wait_change_offer", "kan_offer_no_draw", "kan_offer_over_cap", "call_offer_last_discard",
                "call_offer_kan4_abort", "call_offer_in_chankan", "call_offer_no_legal_discard",
                "kyuushu_offer_not_first_draw", "kyuushu_offer_after_call", "kyuushu_offer_few_kinds"]
Snap = collections.namedtuple("Snap", ["hands", "melds", "draws_left", "scores", "reaches", "last_draw", "actor", "tile",
                                       "window", "own_draws", "called", "doras", "seen", "wall_out"])
FURITEN = ("discard", "temp", "riichi")  # the parser's furiten block reasons; "no_yaku" is the only other one
KAKAN_START = 37  # a kakan token's low byte is 37 + kind (Naki.get_during_turn_filter_idx)
PON_IDX = range(83, 120)  # post-head pon indices; chi is 2-82 (the low 8 bits of a call token)
CONTEXT_GATE = {"plain": "discard_mask_unrestricted_mismatch", "declaration": "riichi_mask_mismatch",
                "chi": "kuikae_mask_mismatch", "pon": "kuikae_mask_mismatch"}
Legality = collections.namedtuple("Legality", ["context", "legal", "forbidden", "called"])
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
    honba/8, rounds_remaining/8, then the round wind one-hot E, S, W (from so.round_wind)."""
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
    x3 = torch.cat([x3, torch.eye(3)[so.round_wind - EAST]])
    return x1, x2, x3


def answered_token(halfturns, i, tokens):
    """The exact token PostTurn halfturns[i] answers, from the halfturn before it: the preceding DiscardTurn's own
    discard token (same seat, same 37-slot tile, right before the PostTurn's history end; only an open kan's
    indicator, flipped after the discard decision, may sit between the two), or the token of the kakan the
    preceding DuringTurn executed. None when the PostTurn lines up with neither."""
    t, prev = halfturns[i], halfturns[i - 1] if i else None
    if isinstance(prev, DiscardTurn) and prev.player == t.player and t.encoding_idx > prev.encoding_idx:
        gap = tokens[prev.encoding_idx: t.encoding_idx - 1]
        tok = tokens[t.encoding_idx - 1]
        if (len(gap) <= 1 and all(_event(g) == EventStateTypes.NEW_DORA for g in gap)
                and _is_discard(tok) and _who(tok) == t.player and _tile(tok) == TILE2IDX[prev.discarded_tile][-1]):
            return tok
    if isinstance(prev, DuringTurn) and prev.player == t.player and t.encoding_idx > 0:
        kakan = [d for d in prev.decisions[DECISION_NAKI_IDX] if d.executed and d.naki.is_kakan()]
        if len(kakan) == 1 and tokens[t.encoding_idx - 1] == naki2idx(t.player, kakan[0].naki):
            return tokens[t.encoding_idx - 1]
    return None


def expected_responders(t: PostTurn, calls=None):
    """Responders that must be post samples, in turn order, with the exact target of each: executors always;
    a non-executor only when no other responder executed something of higher priority than ANY action this
    responder was offered (ron 3 > pon/kan 2 > chi 1), because it may have chosen the weaker one and lost.
    This restates the rule of TransformerTensorProcessor._post_turn_responders, so it catches a regression
    of that code, not a wrong rule; the XML-count gates (tensor_ron == xml_ron, ...) are the independent check.
    calls: {seat: set of post call indices} in place of the parser's call offers (the executed call is kept)."""
    lowest, done, target = {}, {}, {}
    for p in range(4):
        decs = t.decisions[p]
        kinds = [(3, d.executed, 1) for d in decs[DECISION_AGARI_IDX]]
        kinds += [(1 if d.naki.is_chi() else 2, d.executed, d.naki.get_post_turn_filter_idx()) for d in decs[DECISION_NAKI_IDX]]
        if calls is not None:
            executed = {i for _, x, i in kinds[len(decs[DECISION_AGARI_IDX]):] if x}
            kinds = kinds[:len(decs[DECISION_AGARI_IDX])] + [(1 if i < PON_IDX.start else 2, i in executed, i)
                                                             for i in sorted(calls.get(p, set()) | executed)]
        if not kinds:
            continue
        lowest[p] = min(k for k, _, _ in kinds)
        done[p] = max((k for k, x, _ in kinds if x), default=0)
        target[p] = next((idx for _, x, idx in kinds if x), 0)
    top = max(done.values(), default=0)
    return [(p, target[p]) for p in ((t.player + k) % 4 for k in (1, 2, 3)) if p in lowest and (done[p] or top <= lowest[p])]


def expected_during_target(t: DuringTurn) -> int:
    """The executed action of a DuringTurn in the 72-way layout: 1 tsumo, 2 riichi, 3-70 ankan / kakan, 71 kyuushu
    abort, else 0."""
    if any(d.executed for d in t.decisions[DECISION_ABORT_IDX]):
        return legal_oracles.KYUUSHU_IDX
    if any(d.executed for d in t.decisions[DECISION_AGARI_IDX]):
        return 1
    if any(d.executed for d in t.decisions[DECISION_REACH_IDX]):
        return 2
    return next((d.naki.get_during_turn_filter_idx() for d in t.decisions[DECISION_NAKI_IDX] if d.executed), 0)


def _attr(attrs, name):
    return re.search(rf'\b{name}="([^"]*)"', attrs).group(1)


def xml_rounds(raw: str):
    """Per round, replayed from the XML alone, in log order: end ('agari' | 'exhaustive' | 'abortive'), the INIT
    dora indicator, the six INIT seed ints (seed: round index, honba, kyotaku, dice, dice, indicator), the INIT
    dealer (oya), the seats of the riichi declarations (REACH step 1) in order, and the history tokens the
    parser must emit: the INIT indicator, then DORA -> new dora, REACH step 1 -> reach, N -> call, D/E/F/G ->
    discard (tsumogiri iff it is the tile just drawn); nothing after the round's first AGARI / RYUUKYOKU. wins: per
    AGARI event, (who, fromWho, Tenhou yaku ids). yao9: (seat, that seat's draw count) of a kyuushu abort (the
    seat whose hai<N> the RYUUKYOKU shows), else None."""
    rounds, acquired = [], None
    for m in XML_EVENT.finditer(raw):
        tag, attrs = m.group(1), m.group(2)
        if tag == "INIT":
            seed = [int(x) for x in _attr(attrs, "seed").split(",")]
            dora = seed[5]
            rounds.append({"end": None, "dora": dora, "seed": seed, "oya": int(_attr(attrs, "oya")), "reach": [],
                           "tokens": [new_dora2idx(dora // 4)], "wins": [], "yao9": None, "draws": [0] * 4,
                           "win_dora": []})
            acquired = None
            continue
        r = rounds[-1]
        if tag == "AGARI" and r["end"] in (None, "agari"):  # a double ron is two AGARI events
            r["wins"].append((int(_attr(attrs, "who")), int(_attr(attrs, "fromWho")), win_oracles.xml_yaku_ids(attrs)))
            r["win_dora"].append([int(x) for x in _attr(attrs, "doraHai").split(",")])
            r["end"] = "agari"
            continue
        if r["end"] is not None:
            continue
        if tag == "RYUUKYOKU":
            r["end"] = "abortive" if "type=" in attrs else "exhaustive"
            if 'type="yao9"' in attrs:
                shown = [q for q in range(4) if re.search(rf'\bhai{q}="', attrs)]
                r["yao9"] = (shown[0], r["draws"][shown[0]]) if len(shown) == 1 else ("hands shown", tuple(shown))
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
            r["draws"]["TUVW".index(tag[0])] += 1
        else:
            tile = int(tag[1:])
            r["tokens"].append(discard2idx("DEFG".index(tag[0]), TILE2IDX[tile][-1], tile == acquired))
            acquired = None
    return rounds


def held(so) -> set:
    return set(np.flatnonzero(so.hand_tensor[:34] > 0).tolist())


def presence_slots(hand) -> np.ndarray:
    """37-slot presence restated: a slot is on iff the hand holds that tile; a five's plain slot needs a non-red copy."""
    pres = (np.asarray(hand) > 0).astype(np.float32)
    for red, plain in RED_TO_PLAIN.items():
        pres[plain] = float(hand[plain] - hand[red] > 0)
    return pres


def expected_slots(pres, legal_kinds) -> np.ndarray:
    """pres with every kind outside legal_kinds cleared, both slots of a five."""
    exp = pres.copy()
    for slot in range(37):
        if RED_TO_PLAIN.get(slot, slot) not in legal_kinds:
            exp[slot] = 0.0
    return exp


def discard_legality(halfturns, i, tokens) -> Legality:
    """Legal kinds of DiscardTurn halfturns[i], rebuilt from tokens and states alone (legal_oracles):
      declaration  the token right before is the seat's own reach: held kinds that keep tenpai (Agari oracle)
      chi / pon    the token right before is the seat's own chi / pon: held kinds minus the forbidden ones, from the
                   latest discard token before the call (the called kind) and the two kinds that left the hand
                   between the answering PostTurn's responder state and this state
      plain        anything else: every held kind
      broken       a call context that cannot be rebuilt (counted as kuikae_mask_mismatch)"""
    t = halfturns[i]
    so, p, eidx = t.stateObj, t.player, t.encoding_idx
    prev = tokens[eidx - 1] if eidx else 0
    if _is_reach(prev) and _who(prev) == p:
        return Legality("declaration", held(so) & legal_oracles.tenpai_kinds_by_agari(so.hand_tensor, so.melds[p]), None, None)
    if not (_is_call(prev) and _event(prev) == EventStateTypes.POST and _who(prev) == p and (prev & 0xFF) < PON_IDX.stop):
        return Legality("plain", held(so), None, None)
    broken = Legality("broken", None, None, None)
    disc = next((tokens[j] for j in range(eidx - 2, -1, -1) if _is_discard(tokens[j])), None)
    before = halfturns[i - 1] if i else None
    if disc is None or not isinstance(before, PostTurn) or p not in before.responder_states:
        return broken
    diff = (before.responder_states[p].hand_tensor[:34] - so.hand_tensor[:34]).astype(int)
    if (diff < 0).any():
        return broken
    called = RED_TO_PLAIN.get(_tile(disc), _tile(disc))
    used = np.repeat(np.arange(34), diff).tolist()
    forbidden = legal_oracles.kuikae_forbidden_by_ranks(called, used)
    context = "pon" if sorted(used) == [called, called] else "chi"
    if forbidden is None or ((prev & 0xFF) in PON_IDX) != (context == "pon"):
        return broken
    return Legality(context, held(so) - forbidden, forbidden, called)


def xml_snapshots(raw: str):
    """Per round, replayed from the XML alone: the state at every decision point the parser builds a halfturn for,
    in log order. 'during': after each draw (T/U/V/W); 'discard': each discard, the tile still in hand; 'post':
    after each discard, and after each kakan (window 'kakan', the chankan window). A Snap holds the 37-slot hands
    (INIT hai0-3, draws, discards, the tiles each N takes from its caller), every seat's meld codes (a kakan
    replaces its pon), the live draws left (70 minus the draws so far), the scores (INIT ten minus 10 per own
    REACH step 2), the riichi seats (REACH step 1), the latest draw (seat, tile; None after a call), the actor
    and the tile (drawn / discarded / added by the kakan), each seat's draw count (T/U/V/W, rinshan included) and
    whether any N (chi / pon / kan, ankan included) came yet; the public indicators (136-ids in flip order), the
    37-slot counts of every tile in the open (discards, meld tiles taken from a hand) and of every tile dealt or
    drawn. A DORA whose last non-DORA event is a draw that directly follows a daiminkan / kakan N is that kan's
    indicator: it turns public after the drawer's discard snapshot, before the responders' one."""
    rounds, st = [], None

    def snap(actor, tile, window):
        return Snap(st["hands"].copy(), [list(x) for x in st["melds"]], st["draws"], list(st["scores"]),
                    list(st["reaches"]), st["last_draw"], actor, tile, window, tuple(st["own_draws"]), st["called"],
                    list(st["doras"]), st["seen"].copy(), st["wall_out"].copy())

    def count(key, tile):
        for slot in TILE2IDX[tile]:
            st[key][slot] += 1

    def move(seat, tile, sign):
        for slot in TILE2IDX[tile]:
            st["hands"][seat, slot] += sign

    for m in XML_EVENT.finditer(raw):
        tag, attrs = m.group(1), m.group(2)
        if tag == "INIT":
            st = {"hands": np.zeros((4, 37), dtype=int), "melds": [[] for _ in range(4)], "draws": 70,
                  "scores": [int(x) for x in _attr(attrs, "ten").split(",")], "reaches": [0] * 4, "last_draw": None,
                  "own_draws": [0] * 4, "called": False,
                  "doras": [int(_attr(attrs, "seed").split(",")[5])], "hidden": None, "events": [],
                  "seen": np.zeros(37, dtype=int), "wall_out": np.zeros(37, dtype=int)}
            for q in range(4):
                for tile in map(int, _attr(attrs, f"hai{q}").split(",")):
                    move(q, tile, 1)
                    count("wall_out", tile)
            rounds.append({"during": [], "discard": [], "post": [], "ended": False})
            continue
        r = rounds[-1]
        if r["ended"]:
            continue
        if tag in ("AGARI", "RYUUKYOKU"):
            r["ended"] = True
        elif tag == "DORA":
            tile, ev = int(_attr(attrs, "hai")), st["events"]
            if len(ev) >= 2 and ev[-1][0][0] in "TUVW" and ev[-2][0] == "N" and kind_of(Naki(int(_attr(ev[-2][1], "m")))) in ("minkan", "kakan"):
                assert st["hidden"] is None, "two open-kan indicators face down"
                st["hidden"] = tile
            else:
                st["doras"].append(tile)
            continue  # not an event of its own for the lookback
        elif tag == "REACH":
            who = int(_attr(attrs, "who"))
            if _attr(attrs, "step") == "1":
                st["reaches"][who] = 1
            else:
                st["scores"][who] -= 10
        elif tag == "N":
            who, code = int(_attr(attrs, "who")), int(_attr(attrs, "m"))
            nk = Naki(code)
            exposed, _ = nk.get_exposed()
            for tile in exposed:
                move(who, tile, -1)
                count("seen", tile)
            if nk.is_kakan():
                pon = next(c for c in st["melds"][who] if Naki(c).is_pon() and Naki(c).get_exposed()[0][0] // 4 == exposed[0] // 4)
                st["melds"][who].remove(pon)
            st["melds"][who].append(code)
            st["last_draw"] = None
            st["called"] = True
            if nk.is_kakan():
                r["post"].append(snap(who, exposed[0], "kakan"))
        elif tag[0] in "TUVW":
            who, tile = "TUVW".index(tag[0]), int(tag[1:])
            move(who, tile, 1)
            count("wall_out", tile)
            st["draws"] -= 1
            st["own_draws"][who] += 1
            st["last_draw"] = (who, tile)
            r["during"].append(snap(who, tile, "draw"))
        elif tag[0] in "DEFG":
            who, tile = "DEFG".index(tag[0]), int(tag[1:])
            r["discard"].append(snap(who, tile, "discard"))
            if st["hidden"] is not None:  # the open kan's indicator flips once its discard is made
                st["doras"].append(st["hidden"])
                st["hidden"] = None
            move(who, tile, -1)
            count("seen", tile)
            r["post"].append(snap(who, tile, "discard"))
        st["events"].append((tag, attrs))
    return rounds


def snap_kans(snap):
    """(kan count, distinct kan owners) of a Snap."""
    owners = [q for q in range(4) for c in snap.melds[q] if kind_of(Naki(c)) in ("ankan", "minkan", "kakan")]
    return len(owners), len(set(owners))


def snap_menzen(snap, q) -> bool:
    return all(Naki(c).is_ankan() for c in snap.melds[q])


def check_state_xml(so, snap, seat, C):
    """so (the state of `seat`) == the XML replay: hand, melds, drawn tile, draws left, scores, riichi, menzen."""
    drawn = list(TILE2IDX[snap.last_draw[1]]) if snap.last_draw is not None and snap.last_draw[0] == seat else None
    fields = {
        "hand": np.array_equal(np.asarray(so.hand_tensor), snap.hands[seat]),
        "melds": all(sorted(m.naki_code for m in so.melds[q]) == sorted(snap.melds[q]) for q in range(4)),
        "drawn": (None if so.drawn_tile is None else list(so.drawn_tile)) == drawn,
        "draws": so.remaining_tsumo == snap.draws_left,
        "scores": list(so.scores) == snap.scores,
        "reaches": list(so.reaches) == snap.reaches,
        "menzen": list(so.is_menzen) == [int(snap_menzen(snap, q)) for q in range(4)],
    }
    bad = [name for name, ok in fields.items() if not ok]
    if bad:
        C["state_xml_mismatch"] += 1
        for name in bad:
            C["state_xml_" + name] += 1  # which field drifted (named in the failure)
    if list(so.dora) != snap.doras:
        C["state_xml_dora"] += 1
    public = np.zeros(37, dtype=int)
    for tile in snap.doras:
        for slot in TILE2IDX[tile]:
            public[slot] += 1
    if not (np.array_equal(np.asarray(so.remaining_tiles_pov), FULL_WALL - snap.hands[seat] - snap.seen - public)
            and np.array_equal(np.asarray(so.remaining_tiles), FULL_WALL - snap.wall_out - public)):
        C["pov_xml_mismatch"] += 1


def during_options(snap):
    """(oracle bits, tile-only bits) of the during head (2 riichi, 3-36 ankan, 37-70 kakan, 71 kyuushu) from the
    replay; the tile-only bits are what the parser offered before T-032 (diagnostics; kyuushu: >= 9 kinds on the
    seat's first draw, so removed_kyuushu_offer_after_call counts first draws that a call made ineligible)."""
    p, c = snap.actor, snap.hands[snap.actor][:34]
    n_kans, n_owners = snap_kans(snap)
    menzen, in_riichi = snap_menzen(snap, p), bool(snap.reaches[p])
    pons = [Naki(x).get_exposed()[0][0] // 4 for x in snap.melds[p] if Naki(x).is_pon()]
    oracle = {legal_oracles.RIICHI_IDX} if legal_oracles.riichi_by_rules(
        c, menzen, in_riichi, snap.scores[p], snap.draws_left, n_kans, n_owners) else set()
    oracle |= legal_oracles.kan_by_rules(c, pons, in_riichi, snap.tile // 4, snap.draws_left, n_kans)
    base = {legal_oracles.RIICHI_IDX} if menzen and not in_riichi and legal_oracles.riichi_by_rules(
        c, True, False, 10 ** 6, 70, 0, 0) else set()
    base |= {legal_oracles.ANKAN_IDX + k for k in np.flatnonzero(c == 4).tolist()}
    base |= {legal_oracles.KAKAN_IDX + k for k in pons if c[k] >= 1}
    if legal_oracles.kyuushu_by_rules(c, snap.own_draws[p], snap.called):
        oracle.add(legal_oracles.KYUUSHU_IDX)
    if legal_oracles.kyuushu_by_rules(c, snap.own_draws[p], False):
        base.add(legal_oracles.KYUUSHU_IDX)
    return oracle, base


def during_rule_hits(idx, snap):
    """The named option rules that forbid during-head bit idx in this state."""
    p = snap.actor
    n_kans, n_owners = snap_kans(snap)
    if idx == legal_oracles.KYUUSHU_IDX:
        return [g for g, bad in (("kyuushu_offer_not_first_draw", snap.own_draws[p] != 1),
                                 ("kyuushu_offer_after_call", snap.called),
                                 ("kyuushu_offer_few_kinds", not legal_oracles.kyuushu_by_rules(snap.hands[p][:34], 1, False)))
                if bad]
    if idx == legal_oracles.RIICHI_IDX:
        return ([g for g, bad in (("riichi_offer_low_points", snap.scores[p] < legal_oracles.MIN_RIICHI_SCORE),
                                  ("riichi_offer_few_draws", snap.draws_left < legal_oracles.MIN_RIICHI_DRAWS),
                                  ("riichi_offer_after_kan4", legal_oracles.suukaikan(n_kans, n_owners))) if bad])
    hits = [g for g, bad in (("kan_offer_no_draw", snap.draws_left == 0), ("kan_offer_over_cap", n_kans >= legal_oracles.KAN_CAP)) if bad]
    if snap.reaches[p]:
        kind = idx - legal_oracles.ANKAN_IDX if idx < legal_oracles.KAKAN_IDX else None
        if kind != snap.tile // 4:
            hits.append("riichi_okurikan_offer")
        elif snap.hands[p][kind] == 4 and not legal_oracles.riichi_ankan_keeps_waits(snap.hands[p][:34], kind):
            hits.append("riichi_ankan_wait_change_offer")
    return hits


def post_options(snap, p, unrestricted):
    """(oracle bits, tile-only bits) of responder p's call bits (2-153); unrestricted = decision_mask's offers."""
    n_kans, n_owners = snap_kans(snap)
    chankan = snap.window == "kakan"
    base = set() if chankan or snap.reaches[p] else set(unrestricted)
    oracle = legal_oracles.calls_by_rules(base, snap.hands[p][:34], snap.tile // 4, bool(snap.reaches[p]), chankan,
                                          snap.draws_left, n_kans, n_owners)
    return oracle, base


def post_rule_hits(idx, snap, p):
    n_kans, n_owners = snap_kans(snap)
    hits = [g for g, bad in (("call_offer_last_discard", snap.draws_left == 0),
                             ("call_offer_kan4_abort", legal_oracles.suukaikan(n_kans, n_owners)),
                             ("call_offer_in_chankan", snap.window == "kakan")) if bad]
    if idx >= legal_oracles.MINKAN_IDX:
        if n_kans >= legal_oracles.KAN_CAP:
            hits.append("kan_offer_over_cap")
    else:
        used = legal_oracles.call_used_kinds(idx)
        held_used = all(snap.hands[p][k] >= used.count(k) for k in set(used))
        if held_used and snap.window == "discard" and legal_oracles.call_dead_end(snap.hands[p][:34], idx, snap.tile // 4):
            hits.append("call_offer_no_legal_discard")
    return hits


def check_options(bits, oracle, hits, gate, C):
    """bits (the built mask, or the parser's options of a non-sample) == oracle; every rule that forbids an
    offered bit is counted under its own name."""
    if bits != oracle:
        C[gate] += 1
    for idx in bits:
        for rule in hits(idx):
            C[rule] += 1


def unrestricted_calls(snap):
    """decision_mask (tile-only chi / pon / daiminkan) on the replayed hands: {seat: set of post indices}."""
    kind = TILE2IDX[snap.tile]
    offers = decision_mask(snap.actor, snap.hands.astype(np.float32), kind[0], len(kind) == 2)
    return {q: {d.naki.get_post_turn_filter_idx() for d in offers[q]} for q in range(4)}


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
    expected: per sample, in build order, (target, state, history length, actor, answered token or None,
    Legality or None, win offered or None). The win slot is index 1 of the during and the post head."""
    enc, (x1, x2, x3), mask, y, pad = tt
    for i, (target, so, eidx, actor, answer, lg, win) in enumerate(expected):
        if y[i].item() != target:
            C[TARGET_GATE[head]] += 1
        if win is not None and mask[i, 1].item() != float(win):
            C["win_mask_mismatch"] += 1
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
        if head == "discard" and lg.context != "broken":
            m = mask[i].numpy()
            pres = presence_slots(so.hand_tensor)
            want = expected_slots(pres, lg.legal)
            if m[target] != 1:
                C["discard_target_illegal"] += 1
            if (m > pres).any():
                C["discard_mask_superset"] += 1
            if not np.array_equal(m, want):
                C[CONTEXT_GATE[lg.context]] += 1
            for red, plain in RED_TO_PLAIN.items():
                if m[plain] != want[plain] or m[red] != want[red]:
                    C["discard_mask_red_rule"] += 1
            if lg.context != "plain":  # diagnostics, not gates
                C[f"legal_{lg.context}_samples"] += 1
                C[f"legal_{'declaration' if lg.context == 'declaration' else 'kuikae'}_slots_removed"] += int(pres.sum() - m.sum())
        if head == "post" and (answer is None or eidx == 0 or hist[eidx - 1] != rotate([answer], actor)[0]):
            C["post_history_last_token"] += 1


def answered_kind(tok):
    """The tile kind a PostTurn answers, from its last history token: a discard's kind, or the kind of a kakan."""
    if _is_discard(tok):
        return RED_TO_PLAIN.get(_tile(tok), _tile(tok))
    if _is_call(tok) and _event(tok) == EventStateTypes.DURING and (tok & 0xFF) >= KAKAN_START:
        return (tok & 0xFF) - KAKAN_START
    return None


def _executed_win(halfturns, who, from_who):
    """The executed AgariDecision of an <AGARI> (who, fromWho): it sits on the round's final halfturn."""
    last = halfturns[-1] if halfturns else None
    if who == from_who and isinstance(last, DuringTurn):
        agari = last.decisions[DECISION_AGARI_IDX]
    elif who != from_who and isinstance(last, PostTurn) and last.player == from_who:
        agari = last.decisions[who][DECISION_AGARI_IDX]
    else:
        return None
    return next((d for d in agari if d.executed), None)


def _win_state(halfturns, who, from_who):
    """The winner's StateObject at the executed win decision of an <AGARI> (the round's final halfturn), or None."""
    if _executed_win(halfturns, who, from_who) is None:
        return None
    last = halfturns[-1]
    return last.stateObj if who == from_who else last.responder_states[who]


def check_wins(path, ri, xr, halfturns, tokens, C):
    """WIN gates of one round (win_oracles, never utils/agari.py):
      yaku_oracle_mismatch  an executed win's yaku (Tenhou ids) != its <AGARI> yaku / yakuman ids
      furiten_mismatch      for a PostTurn responder whose concealed hand the answered kind completes: the parser's
                            furiten reason (ron_blocked in FURITEN, else none) != the one rebuilt from the tokens and
                            the earlier PostTurns (a pass = an earlier PostTurn whose answered kind completed the
                            seat's hand then; the current PostTurn is not one of its own passes)
      win_reason_conflict   a complete hand with both or neither of (offer, block reason); an incomplete hand with a
                            block reason
      ron/tsumo_offer_incomplete  an offer on an incomplete hand
      win_offer_without_yaku      an offer whose yaku is empty
    Diagnostics (not gates): offers taken / declined, block reasons."""
    for who, from_who, xml_ids in xr["wins"]:
        C["xml_agari"] += 1
        d = _executed_win(halfturns, who, from_who)
        if d is None:
            continue  # lab_tsumo / lab_ron fail
        C["yaku_checked"] += 1
        ids = win_oracles.parser_yaku_ids(d.yaku)
        if ids != xml_ids:
            C["yaku_oracle_mismatch"] += 1
            if C["yaku_oracle_mismatch"] <= 5:
                print(f"yaku mismatch in {os.path.basename(path)} round {ri} who {who} fromWho {from_who}: "
                      f"parser {d.yaku} -> {sorted(ids) if ids is not None else None}, xml {sorted(xml_ids)}")

    for (who, from_who, _), dora_hai in zip(xr["wins"], xr["win_dora"]):
        so = _win_state(halfturns, who, from_who)
        if so is None:
            continue  # dora_vs_agari_coverage fails
        C["dora_vs_agari_checked"] += 1
        if list(so.dora) != dora_hai:
            C["dora_vs_agari"] += 1

    passes = collections.defaultdict(list)  # seat -> [(answered token index, the answered kind completed the hand)]
    for t in halfturns:
        if isinstance(t, DuringTurn):
            offers = t.decisions[DECISION_AGARI_IDX]
            complete = win_oracles.complete_hand(t.stateObj.hand_tensor[:34])
            if not complete and offers:
                C["tsumo_offer_incomplete"] += 1
            if (complete and bool(offers) == bool(t.tsumo_blocked)) or (not complete and t.tsumo_blocked):
                C["win_reason_conflict"] += 1
            C["win_offer_without_yaku"] += sum(1 for d in offers if not d.yaku)
            C["diag_tsumo_offered"] += len(offers)
            C["diag_tsumo_declined"] += sum(1 for d in offers if not d.executed)
            if t.tsumo_blocked:
                C[f"diag_tsumo_blocked_{t.tsumo_blocked}"] += 1
            continue
        if not isinstance(t, PostTurn) or t.encoding_idx == 0:
            continue
        ans = t.encoding_idx - 1
        kind = answered_kind(tokens[ans])
        if kind is None:
            continue  # post_answered_event_hidden fails
        for p, rs in t.responder_states.items():
            offers = t.decisions[p][DECISION_AGARI_IDX]
            blocked = t.ron_blocked.get(p)
            concealed = rs.hand_tensor[:34]
            complete = win_oracles.complete_on(concealed, kind)
            C["win_offer_without_yaku"] += sum(1 for d in offers if not d.yaku)
            C["diag_ron_offered"] += len(offers)
            C["diag_ron_declined"] += sum(1 for d in offers if not d.executed)
            if blocked:
                C[f"diag_ron_blocked_{blocked}"] += 1
            if not complete:
                if offers:
                    C["ron_offer_incomplete"] += 1
                if blocked:
                    C["win_reason_conflict"] += 1
            else:
                if bool(offers) == bool(blocked):
                    C["win_reason_conflict"] += 1
                own = [i for i in range(ans) if _is_discard(tokens[i]) and _who(tokens[i]) == p]
                reach = next((i for i in range(ans) if _is_reach(tokens[i]) and _who(tokens[i]) == p), None)
                last_own = own[-1] if own else -1
                verdict = win_oracles.furiten_reason(
                    concealed,
                    {answered_kind(tokens[i]) for i in own},
                    any(c for j, c in passes[p] if j > last_own),
                    reach is not None and any(c for j, c in passes[p] if j > reach),
                )
                C["furiten_checked"] += 1
                if verdict != (blocked if blocked in FURITEN else None):
                    C["furiten_mismatch"] += 1
                    if C["furiten_mismatch"] <= 5:
                        print(f"furiten mismatch in {os.path.basename(path)} round {ri} token {ans} seat {p} kind "
                              f"{kind}: parser {blocked}, oracle {verdict}")
            passes[p].append((ans, complete))


def declined_offers(H):
    """(round, halfturn index, seat, 'tsumo' | 'ron', yaku, scores) of every win offer that was not taken."""
    for ri, halfturns in enumerate(H):
        for i, t in enumerate(halfturns):
            if isinstance(t, DuringTurn):
                for d in t.decisions[DECISION_AGARI_IDX]:
                    if not d.executed:
                        yield ri, i, t.player, "tsumo", d.yaku, t.stateObj.scores
            elif isinstance(t, PostTurn):
                for p, rs in t.responder_states.items():
                    for d in t.decisions[p][DECISION_AGARI_IDX]:
                        if not d.executed:
                            yield ri, i, p, "ron", d.yaku, rs.scores


def check_kyuushu(halfturns, xr, C):
    """The executed kyuushu aborts of one round, as (seat, that seat's draw count): one DuringTurn per draw, so the
    rank of a DuringTurn among its seat's DuringTurns is the seat's draw count. == the XML yao9, none without one."""
    draws, executed = [0] * 4, []
    for t in halfturns:
        if not isinstance(t, DuringTurn):
            continue
        draws[t.player] += 1
        offered = t.decisions[DECISION_ABORT_IDX]
        C["kyuushu_offers"] += len(offered)
        if any(d.executed for d in offered):
            executed.append((t.player, draws[t.player]))
        elif offered:
            C["kyuushu_declined"] += 1
    C["lab_kyuushu"] += len(executed)
    if executed != ([xr["yao9"]] if xr["yao9"] is not None else []):
        C["kyuushu_label_mismatch"] += 1


def check_round_wind(head, x3, xr, tonpuu, C):
    """Bakaze slots 4-6 of EVERY built sample row of one head == the one-hot of the round wind read from the XML
    INIT seed (seed[0] // 4: 0 East, 1 South, 2 West), never from agari.round_wind or the StateObject. A North round
    or an x3 without the 3 slots fails every row. head: the head name (the self-test skips one head)."""
    n, wind = x3.shape[0], xr["seed"][0] // 4
    C["round_wind_checked"] += n
    if tonpuu:
        C["round_wind_tonpuu_samples"] += n
    if wind > 2 or x3.shape[1] < 7:
        C["round_wind_xml"] += n
        return
    C["round_wind_samples_" + "ESW"[wind]] += n
    C["round_wind_xml"] += int((x3[:, 4:7] != torch.eye(3)[wind]).any(dim=1).sum().item())


def count_game(path, H, T, C, tp):
    """Add every counter of one parsed game (H = halfturns per round, T = tokens per round) to C."""
    raw = open(path, "rb").read().decode("utf-8", "ignore")
    ag = re.findall(r'<AGARI [^>]*?who="(\d)"[^>]*?fromWho="(\d)"', raw)
    C["xml_tsumo"] += sum(1 for w, fw in ag if w == fw)
    C["xml_ron"] += sum(1 for w, fw in ag if w != fw)
    C["xml_kyuushu"] += len(re.findall(YAO9, raw))
    ron3 = len(re.findall(r'<RYUUKYOKU [^>]*type="ron3"', raw))
    C["xml_ron3"] += ron3
    C["xml_ron"] += 3 * ron3  # triple ron: three rons, no AGARI
    for m in re.findall(r'<N who="\d" m="(\d+)"', raw):
        C["xml_" + kind_of(Naki(int(m)))] += 1
    C["xml_open_kan_dora"] += len(re.findall(OPEN_KAN_DORA_RE, raw))
    go = re.search(r'<GO type="(\d+)"', raw)
    tonpuu = go is not None and not int(go.group(1)) & 0x8  # GO bit 0x8 = hanchan (East + South)
    rounds = xml_rounds(raw)
    assert len(rounds) == len(H), f"{path}: {len(rounds)} rounds in XML, {len(H)} parsed"
    replays = xml_snapshots(raw)

    for ri, (xr, xs, halfturns, tokens) in enumerate(zip(rounds, replays, H, T)):
        C["rounds"] += 1
        check_wins(path, ri, xr, halfturns, tokens, C)
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
        check_kyuushu(halfturns, xr, C)
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
        opt = {"during": [], "post": []}  # per sample, in build order: (oracle bits, hits(idx))
        kinds = {DuringTurn: "during", DiscardTurn: "discard", PostTurn: "post"}
        aligned = all(len(xs[k]) == sum(isinstance(t, cls) for t in halfturns) for cls, k in kinds.items())
        if not aligned:
            C["state_xml_mismatch"] += 1  # the parser built other decision points than the XML has
            C["state_xml_alignment"] += 1
        snaps = {k: iter(xs[k]) for k in kinds.values()}
        for i, t in enumerate(halfturns):
            so = t.stateObj
            snap = next(snaps[kinds[type(t)]]) if aligned else None
            if isinstance(t, DuringTurn):
                if snap is not None:
                    check_state_xml(so, snap, t.player, C)
                    oracle, base = during_options(snap)
                    hits = lambda idx, snap=snap: during_rule_hits(idx, snap)
                    for idx in base:
                        for rule in hits(idx):
                            C["removed_" + rule] += 1
                    if not sum(map(len, t.decisions)):  # no sample: the parser offered nothing
                        check_options(set(), oracle, hits, "during_mask_mismatch", C)
                if sum(map(len, t.decisions)):
                    exp["during"].append((expected_during_target(t), so, t.encoding_idx, t.player, None, None,
                                          bool(t.decisions[DECISION_AGARI_IDX])))
                    opt["during"].append((oracle, hits) if snap is not None else None)
                if t.decisions[DECISION_REACH_IDX]:
                    C["riichi_offers"] += 1
                    if not legal_oracles.tenpai_kinds_by_agari(so.hand_tensor, so.melds[t.player]):
                        C["riichi_offer_without_legal_discard"] += 1
                if t.decisions[DECISION_AGARI_IDX] and t.decisions[DECISION_AGARI_IDX][0].executed:
                    C["lab_tsumo"] += 1
                for d in t.decisions[DECISION_NAKI_IDX]:
                    if d.executed:
                        C["lab_" + kind_of(d.naki)] += 1
                check_state(so, C, tp)

            elif isinstance(t, DiscardTurn):
                if snap is not None:
                    check_state_xml(so, snap, t.player, C)
                if t.encoding_idx < len(tokens) and _event(tokens[t.encoding_idx]) == EventStateTypes.NEW_DORA:
                    C["open_kan_dora_deferred"] += 1  # an indicator flipped between this decision and its token
                drew = halfturns[i - 1] if i else None
                if isinstance(drew, DuringTurn) and drew.player == t.player:  # its own draw: only a riichi can follow
                    C["dora_draw_to_discard_checked"] += 1
                    between = tokens[drew.encoding_idx: t.encoding_idx]
                    if list(so.dora) != list(drew.stateObj.dora) or any(_event(x) == EventStateTypes.NEW_DORA for x in between):
                        C["dora_draw_to_discard"] += 1
                # a declaration discard is a sample, a later riichi discard is not: decided from the tokens (the
                # seat's own reach right before), never from is_riichi_declaration, which is checked against it
                prev = tokens[t.encoding_idx - 1] if t.encoding_idx else 0
                declared = _is_reach(prev) and _who(prev) == t.player
                if not (so.reaches[t.player] and not declared):
                    lg = discard_legality(halfturns, i, tokens)
                    if lg.context == "broken":
                        C["kuikae_mask_mismatch"] += 1
                    exp["discard"].append((TILE2IDX[t.discarded_tile][-1], so, t.encoding_idx, t.player, None, lg, None))
                if t.is_riichi_declaration != declared or (declared and so.reaches[t.player] != 1):
                    C["riichi_declaration_unflagged"] += 1  # the flag disagrees with the tokens, either way
                elif so.reaches[t.player] and not declared:
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
                samples = expected_responders(t)
                post_opt = {}
                if snap is not None:
                    unrestricted = unrestricted_calls(snap)
                    bases = {}
                    for p in (q for q in range(4) if q != snap.actor):  # every seat but the actor, from the replay
                        rs = t.responder_states.get(p)
                        if rs is None:  # a missing state would hide this seat's options from every gate
                            C["state_xml_mismatch"] += 1
                            C["state_xml_responder_missing"] += 1
                        else:
                            check_state_xml(rs, snap, p, C)
                        oracle, base = post_options(snap, p, unrestricted[p])
                        bases[p] = base
                        hits = lambda idx, snap=snap, p=p: post_rule_hits(idx, snap, p)
                        for idx in base:
                            for rule in hits(idx):
                                C["removed_" + rule] += 1
                        post_opt[p] = (oracle, hits)
                        if p not in {q for q, _ in samples}:  # not a sample: its parser options are what it saw
                            offered = {d.naki.get_post_turn_filter_idx() for d in t.decisions[p][DECISION_NAKI_IDX]}
                            check_options(offered, oracle, hits, "post_mask_mismatch", C)
                    before = {q for q, _ in expected_responders(t, bases)}
                    after = {q for q, _ in samples}
                    C["post_samples_gained"] += len(after - before)
                    C["post_samples_lost"] += len(before - after)
                for p, target in samples:
                    exp["post"].append((target, t.responder_states[p], t.encoding_idx, p, answer, None,
                                        bool(t.decisions[p][DECISION_AGARI_IDX])))
                    opt["post"].append(post_opt.get(p))

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
        C["discard_mask_empty"] = tp.skipped["discard_mask_empty"]
        for name, tt in zip(HEADS, (td, tdisc, tpost)):
            check_round_wind(name, tt[1][2], xr, tonpuu, C)
            if len(tt[3]):
                C[f"{name}_label_outside_mask"] += int((tt[2][np.arange(len(tt[3])), tt[3]] != 1).sum().item())
                C[f"{name}_samples"] += len(tt[3])
            if len(tt[3]) != len(exp[name]):
                C[f"{name}_sample_count_mismatch"] += 1
            else:
                check_samples(name, tt, exp[name], tokens, C)
                for row, o in zip(tt[2], opt.get(name, [])):  # the BUILT masks the loss uses
                    if o is not None:
                        oracle, hits = o
                        check_options(set((np.flatnonzero(row[2:].numpy()) + 2).tolist()), oracle, hits, f"{name}_mask_mismatch", C)
        tensor_counts(td[3].tolist(), tpost[3].tolist(), C)


def tensor_counts(y_during, y_post, C):
    """Count the built targets per action kind (the head layouts: during 72, post 154)."""
    C["tensor_tsumo"] += sum(1 for v in y_during if v == 1)
    C["tensor_riichi"] += sum(1 for v in y_during if v == 2)
    C["tensor_ankan"] += sum(1 for v in y_during if 3 <= v <= 36)
    C["tensor_kakan"] += sum(1 for v in y_during if 37 <= v <= 70)
    C["tensor_kyuushu"] += sum(1 for v in y_during if v == legal_oracles.KYUUSHU_IDX)
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
            if e.msg == "North round":  # 0 in 2.2M games: name the file, never hide it in skipped_invalid
                C["skipped_invalid_north"] += 1
                print(f"North round rejected: {f}")
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
    if "state_xml_mismatch" in bad:  # name the fields that drifted
        bad["state_xml_mismatch"] = f"{C['state_xml_mismatch']} " + str(
            {k[len("state_xml_"):]: v for k, v in C.items() if k.startswith("state_xml_") and k != "state_xml_mismatch" and v})
    # every riichi declaration and every chi / pon has its discard sample, and the oracle found its context
    if C["legal_declaration_samples"] != C["lab_riichi"]:
        bad["legal_declaration_coverage"] = f"{C['legal_declaration_samples']} != lab_riichi={C['lab_riichi']}"
    if C["yaku_checked"] != C["xml_agari"]:  # every <AGARI> had its executed decision checked against Tenhou's yaku
        bad["yaku_coverage"] = f"{C['yaku_checked']} != xml_agari={C['xml_agari']}"
    # ... and xml_rounds() saw every <AGARI>: tie its count to the raw regex counts (a triple ron has no AGARI)
    if C["xml_agari"] != C["xml_tsumo"] + C["xml_ron"] - 3 * C["xml_ron3"]:
        bad["agari_coverage"] = (f"xml_agari={C['xml_agari']} != xml_tsumo + xml_ron - 3 * xml_ron3="
                                 f"{C['xml_tsumo'] + C['xml_ron'] - 3 * C['xml_ron3']}")
    if C["dora_vs_agari_checked"] != C["xml_agari"]:  # every <AGARI>'s doraHai was compared
        bad["dora_vs_agari_coverage"] = f"{C['dora_vs_agari_checked']} != xml_agari={C['xml_agari']}"
    if C["open_kan_dora_deferred"] != C["xml_open_kan_dora"]:
        bad["open_kan_dora_deferred"] = f"{C['open_kan_dora_deferred']} != xml `T DORA D`={C['xml_open_kan_dora']}"
    if C["round_wind_checked"] != sum(C[f"{h}_samples"] for h in HEADS):  # every built row had its bakaze checked
        bad["round_wind_coverage"] = (f"{C['round_wind_checked']} != during + discard + post samples="
                                      f"{sum(C[f'{h}_samples'] for h in HEADS)}")
    if C["dora_draw_to_discard_checked"] == 0:
        bad["dora_draw_to_discard_checked"] = 0
    if C["legal_chi_samples"] + C["legal_pon_samples"] != C["lab_chi"] + C["lab_pon"]:
        bad["legal_call_coverage"] = (f"{C['legal_chi_samples']} + {C['legal_pon_samples']} != "
                                      f"lab_chi + lab_pon={C['lab_chi'] + C['lab_pon']}")
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
        f"samples " + " ".join(f"{h} {C[h + '_samples']}" for h in HEADS) + " | tensors == parser == xml | "
        f"legal mask: declarations {C['legal_declaration_samples']} (-{C['legal_declaration_slots_removed']} slots) "
        f"after chi {C['legal_chi_samples']} pon {C['legal_pon_samples']} (-{C['legal_kuikae_slots_removed']} slots) "
        f"riichi offers {C['riichi_offers']} | kyuushu {C['lab_kyuushu']}/{C['xml_kyuushu']} (offers "
        f"{C['kyuushu_offers']}, declined {C['kyuushu_declined']}) | options removed: "
        + " ".join(f"{r} {C['removed_' + r]}" for r in OPTION_RULES if r != "call_offer_in_chankan")
        + f" | post samples +{C['post_samples_gained']} -{C['post_samples_lost']} | "
        f"dora: open kan deferred {C['open_kan_dora_deferred']}/{C['xml_open_kan_dora']} draw->discard checked "
        f"{C['dora_draw_to_discard_checked']} == doraHai {C['dora_vs_agari_checked']} | "
        f"win: yaku == tenhou {C['yaku_checked']} furiten checked {C['furiten_checked']}; offers tsumo "
        f"{C['diag_tsumo_offered']} (declined {C['diag_tsumo_declined']}) ron {C['diag_ron_offered']} (declined "
        f"{C['diag_ron_declined']}); blocked tsumo no_yaku {C['diag_tsumo_blocked_no_yaku']}, ron "
        + " ".join(f"{r} {C['diag_ron_blocked_' + r]}" for r in FURITEN + ("no_yaku",))
        + f" | bakaze E/S/W {C['round_wind_samples_E']}/{C['round_wind_samples_S']}/{C['round_wind_samples_W']} "
        f"(tonpuu {C['round_wind_tonpuu_samples']}) == checked {C['round_wind_checked']}"
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
    found = glob.glob(os.path.join(DATA_ROOT, day[:4], day, "*", name))
    assert found, f"self-test needs the game {name} under {DATA_ROOT}/{day[:4]}/{day}"
    raw = open(found[0], "rb").read().decode("utf-8", "ignore")
    assert re.search(pattern, raw), f"{found[0]} does not match {pattern}"
    return found[0]


def _parsed_fixture(fixture, pred):
    """A named fixture game (day, file, round) whose parsed round (every round if None) satisfies pred(H, T)."""
    day, name, rnd = fixture
    found = glob.glob(os.path.join(DATA_ROOT, "2021", day, "*", name))
    assert found, f"self-test needs the game {name} under {DATA_ROOT}/2021/{day}"
    H, T = process(found[0])
    sel = (H, T) if rnd is None else ([H[rnd]], [T[rnd]])
    assert pred(*sel), f"{found[0]} round {rnd} no longer has the case its self-test needs"
    return found[0]


def _durings(H):
    return (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn))


def _base_riichi(t) -> bool:
    """Menzen, not in riichi, shanten <= 0: the tile conditions of a riichi."""
    so, p = t.stateObj, t.player
    return bool(so.is_menzen[p]) and not so.reaches[p] and legal_oracles.riichi_by_rules(
        so.hand_tensor, True, False, 10 ** 6, 70, 0, 0)


def _executed_kinds(H) -> set:
    return {kind_of(d.naki) for halfturns in H for t in halfturns
            for decs in ([t.decisions] if isinstance(t, DuringTurn) else t.decisions if isinstance(t, PostTurn) else [])
            for d in (decs[DECISION_NAKI_IDX] if decs else []) if d.executed}


def _houtei_call_candidate(H) -> bool:
    """A responder not in riichi who holds a pon / chi shape for the last discard."""
    for halfturns in H:
        for i, t in enumerate(halfturns):
            if not isinstance(t, PostTurn) or not isinstance(halfturns[i - 1], DiscardTurn):
                continue
            for p, rs in t.responder_states.items():
                if rs.remaining_tsumo or rs.reaches[p]:
                    continue
                hands = np.zeros((4, 37), dtype=np.float32)
                hands[p] = rs.hand_tensor
                tile = TILE2IDX[halfturns[i - 1].discarded_tile]
                if decision_mask(t.player, hands, tile[0], len(tile) == 2)[p]:
                    return True
    return False


def _quiet_draw(H, kind_idx):
    """(round, remaining_tsumo) of the first DuringTurn offered an option of decisions[kind_idx] that executed
    nothing: switching that draw's options off cannot hit the parser's label asserts. None if there is none."""
    for r, halfturns in enumerate(H):
        for t in halfturns:
            if isinstance(t, DuringTurn) and t.decisions[kind_idx] and not any(d.executed for ds in t.decisions for d in ds):
                return r, t.stateObj.remaining_tsumo
    return None


def _option_fixtures() -> dict:
    """Games for the T-032 corruptions: the 2021-01-01 pool where it has the case, else a named fixture."""
    def riichi_at(draws):  # an offer at exactly `draws`, and no EXECUTED riichi there (a stricter rule then removes
        # only unexecuted offers: the re-parse cannot trip the L10 assert)
        def pred(H, T):
            at = [t for t in _durings(H) if t.decisions[DECISION_REACH_IDX] and t.stateObj.remaining_tsumo == draws]
            return bool(at) and not any(d.executed for t in at for d in t.decisions[DECISION_REACH_IDX])
        return pred

    def kan_offer_unexecuted(H, T):
        return _quiet_draw(H, DECISION_NAKI_IDX) is not None

    def riichi_offer_unexecuted(H, T):
        return _quiet_draw(H, DECISION_REACH_IDX) is not None

    def okurikan(H, T):
        return any(t.stateObj.reaches[t.player] and any(t.stateObj.hand_tensor[k] == 4 and k != t.stateObj.drawn_tile[0]
                                                         for k in range(34)) for t in _durings(H))

    def haitei_kan(H, T):
        def kan_shape(t):
            so, p = t.stateObj, t.player
            pons = [m.get_exposed()[0][0] // 4 for m in so.melds[p] if m.is_pon()]
            return (so.hand_tensor[:34] == 4).any() or any(so.hand_tensor[k] >= 1 for k in pons)
        return any(t.stateObj.remaining_tsumo == 0 and kan_shape(t) for t in _durings(H))

    def chankan(H, T):
        return "kakan" in _executed_kinds(H)

    def call_choice(H, T):
        return any(len(t.decisions[p][DECISION_NAKI_IDX]) >= 2 for halfturns in H for t in halfturns
                   if isinstance(t, PostTurn) for p in range(4))

    def named(fixture, pred):
        return _parsed_fixture(fixture, pred)

    def riichi_low_points(H, T):
        return any(_base_riichi(t) and t.stateObj.scores[t.player] < 10 for t in _durings(H))

    def wait_change(H, T):
        return any(t.stateObj.reaches[t.player] and t.stateObj.hand_tensor[t.stateObj.drawn_tile[0]] == 4
                   and not legal_oracles.riichi_ankan_keeps_waits(t.stateObj.hand_tensor, t.stateObj.drawn_tile[0])
                   for t in _durings(H))

    def kan4(H, T):
        return any(isinstance(t, PostTurn) and any(legal_oracles.suukaikan(*legal.kans_on_table(rs.melds)) and rs.remaining_tsumo
                                                    for rs in t.responder_states.values()) for halfturns in H for t in halfturns)

    def riichi_ankan(H, T):
        return any(t.stateObj.reaches[t.player] and any(d.executed for d in t.decisions[DECISION_NAKI_IDX]) for t in _durings(H))

    def dead_end(H, T):
        for halfturns in H:
            for i, t in ((i, t) for i, t in enumerate(halfturns) if isinstance(t, PostTurn)):
                for p, rs in t.responder_states.items():
                    hands = np.zeros((4, 37), dtype=np.float32)
                    hands[p] = rs.hand_tensor
                    disc = halfturns[i - 1] if isinstance(halfturns[i - 1], DiscardTurn) else None
                    if disc is None or rs.reaches[p] or not rs.remaining_tsumo:
                        continue
                    tile = TILE2IDX[disc.discarded_tile]
                    for d in decision_mask(t.player, hands, tile[0], len(tile) == 2)[p]:
                        if not d.naki.is_minkan() and not legal.call_leaves_legal_discard(rs.hand_tensor, d.naki):
                            return True
        return False

    pool = {
        "few_draws": lambda H, T: any(_base_riichi(t) and t.stateObj.remaining_tsumo < 4 for t in _durings(H)),
        "riichi_at_4": riichi_at(4),
        "okurikan": okurikan,
        "haitei_kan": haitei_kan,
        "houtei_call": lambda H, T: _houtei_call_candidate(H),
        "kan_offer_unexecuted": kan_offer_unexecuted,
        "riichi_offer_unexecuted": riichi_offer_unexecuted,
        "chankan": chankan,
        "call_choice": call_choice,
    }
    games = {name: _game_with(pred) for name, pred in pool.items()}
    games["low_points"] = named(LOW_POINTS_RIICHI, riichi_low_points)
    games["wait_change"] = named(WAIT_CHANGE_ANKAN, wait_change)
    games["kan4"] = named(KAN4_ROUND, kan4)
    games["dead_end"] = named(DEAD_END_CALL, dead_end)
    games["riichi_ankan"] = named(RIICHI_ANKAN, riichi_ankan)
    return games


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


def _patch_mask(tp, edit):
    """Corrupt the discard masks: edit(turn, mask) runs on a copy of each sample's mask and returns the mask used."""
    orig = tp._discard_mask
    tp._discard_mask = lambda turn: edit(turn, orig(turn).clone())


def legal_samples(H, T):
    """(DiscardTurn, Legality) for every discard sample (the forced discards of a player in riichi excluded)."""
    for halfturns, tokens in zip(H, T):
        for i, t in enumerate(halfturns):
            if isinstance(t, DiscardTurn) and not (t.stateObj.reaches[t.player] and not t.is_riichi_declaration):
                yield t, discard_legality(halfturns, i, tokens)


def self_test():
    """Clean games pass every gate; each corruption, applied to a game that passes every gate, must fail its NAMED
    gate (another gate failing as well is allowed). Also: a one-loader cluster rolls over at the epoch boundary."""
    f = _first_valid_game()
    ron3 = _named_game(RON3_LOG)
    red_kan_dora = _named_game(RED_KAN_DORA_LOG)
    chankan_ippatsu = _named_game(CHANKAN_IPPATSU_LOG)
    early_ankan = _named_game(EARLY_ANKAN_LOG)
    open_kan, open_kan_ron = _named_game(OPEN_KAN_LOG), _named_game(OPEN_KAN_RON_LOG)
    kan_games = [_named_game(g) for g in (PENDING_FLIP_TSUMO_LOG, OPEN_KAN_THEN_ANKAN_LOG)]
    option_games = _option_fixtures()
    kyuushu_games = [_named_game(YAO9_LOG), _named_game(KYUUSHU_AFTER_CALL_LOG)]  # an abort, a decline, 9 kinds after a call
    tonpuu, west = _named_game(TONPUU_SOUTH_LOG), _named_game(HANCHAN_WEST_LOG)
    clean = run([f, ron3, red_kan_dora, chankan_ippatsu, early_ankan, open_kan, open_kan_ron] + kan_games
                + sorted(set(option_games.values())) + kyuushu_games + [tonpuu, west], TransformerTensorProcessor())
    check(clean)
    for k in ("round_wind_samples_E", "round_wind_samples_S", "round_wind_samples_W", "round_wind_tonpuu_samples"):
        if not clean[k]:  # the bakaze gate saw every wind and a tonpuu game, or it proves nothing about them
            raise SystemExit(f"self-test FAILED: the clean pool has no {k}")

    def expect_failure(name, gate, corrupt, game=f, absent=()):
        """corrupt(H, T, tp) edits the parse in place; it may return a context manager that stays active while the
        game is counted. gate: one gate name or a tuple of names, every one of which must fail. absent: gates that
        must NOT fail (the corruption is invisible to them)."""
        gates = (gate,) if isinstance(gate, str) else gate
        baseline = failed_gates(_count(game))
        if baseline:
            raise SystemExit(f"self-test FAILED: {name}: the clean game {game} already fails {baseline}")
        tp = TransformerTensorProcessor()
        H, T = process(game)
        ret = corrupt(H, T, tp)  # a context manager kept active while the game is counted, or run()'s counters
        if hasattr(ret, "__enter__"):
            with ret:
                bad = failed_gates(_count(game, H, T, tp))
        else:
            bad = failed_gates(ret if ret is not None else _count(game, H, T, tp))
        missed = [g for g in gates if g not in bad]
        if missed:
            raise SystemExit(f"self-test FAILED: {name} did not fail gate(s) {missed} (failed: {bad or 'none'})")
        caught = [g for g in absent if g in bad]
        if caught:
            raise SystemExit(f"self-test FAILED: {name} also failed {caught}, which must not see it (failed: {bad})")
        print(f"self-test OK: {name} -> " + ", ".join(f"{g}={bad[g]}" for g in gates)
              + "".join(f", {g} passes" for g in absent))

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

    def open_kan_discards(H, T):  # (DiscardTurn, the indicator flipped right after it) of every open-kan discard
        for halfturns, tokens in zip(H, T):
            for i, t in enumerate(halfturns):
                if isinstance(t, DiscardTurn) and t.encoding_idx < len(tokens) and _event(tokens[t.encoding_idx]) == EventStateTypes.NEW_DORA:
                    yield t, next(iter(halfturns[i + 1].responder_states.values())).dora[-1]

    def open_kan_dora_seen(with_dora):  # the old XML order: the kan player's discard state already has the indicator
        def corrupt(H, T, tp):
            n = 0
            for t, tile in open_kan_discards(H, T):
                so = t.stateObj
                if with_dora:
                    so.dora = so.dora + [tile]
                so.remaining_tiles_pov = so.remaining_tiles_pov.copy()
                so.remaining_tiles_pov[TILE2IDX[tile]] -= 1
                n += 1
            found(n, "an open-kan discard")
        return corrupt

    def open_kan_token_seen(H, T, tp):  # the old history cut: the discard sample's prefix holds the indicator token
        n = 0
        for t, _ in open_kan_discards(H, T):
            t.encoding_idx += 1
            n += 1
        found(n, "an open-kan discard")

    def ron_misses_kan_dora(H, T, tp):  # the ron winner's state lacks the indicator flipped after the kan discard
        for halfturns, tokens in zip(H, T):
            disc, last = halfturns[-2:] if len(halfturns) > 1 else (None, None)
            if not (isinstance(disc, DiscardTurn) and isinstance(last, PostTurn) and disc in [t for t, _ in open_kan_discards([halfturns], [tokens])]):
                continue
            for p, rs in last.responder_states.items():
                if any(d.executed for d in last.decisions[p][DECISION_AGARI_IDX]):
                    rs.dora = rs.dora[:-1]
                    return
        found(0, "a ron on an open-kan discard")

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

    def zero_scalars(H, T, tp):  # the 4 scalar slots only: the bakaze slots must not be what trips the gate
        _patch_inputs(tp, lambda x1, x2, x3: x3[:, :4].zero_())

    # ---- round wind (T-038)
    def rolled_bakaze(H, T, tp):  # (A) the converter's one-hot rolled by one slot: E->S, S->W, W->E
        _patch_inputs(tp, lambda x1, x2, x3: x3[:, 4:7].copy_(x3[:, 4:7].roll(1, dims=1)))

    def parser_wind_east(H, T, tp):  # (B) the parser stores East for every round: input == StateObject still holds
        for so in _states(H):
            so.round_wind = EAST

    def hanchan_wind_formula(H, T, tp):  # (C) the wind derived as if every game were hanchan (tonpuu E1-E4 -> South)
        for so in _states(H):
            so.round_wind = EAST + (8 - so.rounds_remaining) // 4

    def post_wind_unchecked(H, T, tp):  # (D) check_round_wind skipped for the post head
        orig = check_round_wind
        return mock.patch(f"{__name__}.check_round_wind", lambda head, *a: None if head == "post" else orig(head, *a))

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

    # ---- option legality (T-032): parser rules switched off / made too strict, then re-parsed
    def reparse(game, **patches):  # corrupt: re-parse with legal.<name> replaced
        def corrupt(H, T, tp):
            with contextlib.ExitStack() as stack:
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(legal, name, value))
                H[:], T[:] = process(game)
        return corrupt

    riichi_allowed, ankan_allowed, kans_on_table = legal.riichi_allowed, legal.riichi_ankan_allowed, legal.kans_on_table
    g = option_games
    expect_failure("riichi points rule off", ("riichi_offer_low_points", "during_mask_mismatch"),
                   reparse(g["low_points"], RIICHI_MIN_SCORE=0), game=g["low_points"])
    expect_failure("riichi wall rule off", "riichi_offer_few_draws", reparse(g["few_draws"], RIICHI_MIN_DRAWS_LEFT=0),
                   game=g["few_draws"])
    expect_failure("riichi wall rule too strict (5 draws)", "during_mask_mismatch",
                   reparse(g["riichi_at_4"], RIICHI_MIN_DRAWS_LEFT=5), game=g["riichi_at_4"])
    expect_failure("riichi okurikan allowed", ("riichi_okurikan_offer", "during_mask_mismatch"),
                   reparse(g["okurikan"], riichi_ankan_allowed=lambda c, d, k, m: k != d or ankan_allowed(c, d, k, m)),
                   game=g["okurikan"])
    expect_failure("riichi ankan wait check off", ("riichi_ankan_wait_change_offer", "during_mask_mismatch"),
                   reparse(g["wait_change"], riichi_ankan_allowed=lambda c, d, k, m: k == d), game=g["wait_change"])
    expect_failure("kan on the haitei draw allowed", ("kan_offer_no_draw", "during_mask_mismatch"),
                   reparse(g["haitei_kan"], kan_allowed=lambda draws, n: n < legal.MAX_KANS), game=g["haitei_kan"])
    expect_failure("calls on the last discard allowed", ("call_offer_last_discard", "post_mask_mismatch"),
                   reparse(g["houtei_call"], calls_allowed=lambda draws, n, o: not legal.suukaikan_pending(n, o)),
                   game=g["houtei_call"])
    expect_failure("calls after a 4th kan by two players allowed", ("call_offer_kan4_abort", "post_mask_mismatch"),
                   reparse(g["kan4"], calls_allowed=lambda draws, n, o: draws > 0), game=g["kan4"])
    expect_failure("kuikae dead-end call allowed", ("call_offer_no_legal_discard", "post_mask_mismatch"),
                   reparse(g["dead_end"], call_leaves_legal_discard=lambda hand, call: True), game=g["dead_end"])
    expect_failure("parser riichi rule reads the draw counter off by one", "during_mask_mismatch",
                   reparse(g["few_draws"], riichi_allowed=lambda sc, d, n, o: riichi_allowed(sc, d + 1, n, o)),
                   game=g["few_draws"])

    def at_draw(game, kind_idx, fake):  # kans_on_table answers `fake` on ONE quiet draw of the parser, else the truth
        target = _quiet_draw(process(game)[0], kind_idx)

        def patched(melds):
            loc = sys._getframe(1).f_locals
            ev = loc.get("eventtype", "")
            here = ev and ev[0] in "TUVW" and (len(loc["all_halfturns"]), loc["remaining_tsumo"]) == target
            return fake if here else kans_on_table(melds)
        return patched

    kan_game, riichi_offer_game = g["kan_offer_unexecuted"], g["riichi_offer_unexecuted"]
    expect_failure("parser sees 4 kans (one owner) at a draw: its kan bits vanish", "during_mask_mismatch",
                   reparse(kan_game, kans_on_table=at_draw(kan_game, DECISION_NAKI_IDX, (4, 1))), game=kan_game)
    expect_failure("parser sees suukaikan pending at a draw: its riichi bit vanishes", "during_mask_mismatch",
                   reparse(riichi_offer_game, kans_on_table=at_draw(riichi_offer_game, DECISION_REACH_IDX, (4, 2))),
                   game=riichi_offer_game)

    def replay_sees_kans(fake):  # the gate's replay believes `fake` kans are down: the counters must see real offers
        return lambda H, T, tp: mock.patch.object(sys.modules[__name__], "snap_kans", lambda snap: fake)

    expect_failure("replay sees 4 kans (one owner): real kan offers counted", "kan_offer_over_cap",
                   replay_sees_kans((4, 1)), game=g["kan_offer_unexecuted"])
    expect_failure("replay sees suukaikan pending: real riichi offers counted", "riichi_offer_after_kan4",
                   replay_sees_kans((4, 2)), game=g["riichi_offer_unexecuted"])

    def drop_executed_riichi_ankan(H, T, tp):  # after parsing: a legal, executed riichi ankan removed
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn) and t.stateObj.reaches[t.player]):
            kans = [d for d in t.decisions[DECISION_NAKI_IDX] if d.executed]
            if kans:
                t.decisions[DECISION_NAKI_IDX].remove(kans[0])
                return
        found(0, "an executed riichi ankan")

    expect_failure("executed riichi ankan removed", ("lab_ankan", "during_mask_mismatch"), drop_executed_riichi_ankan,
                   game=g["riichi_ankan"])

    def drop_riichi_offers(H, T, tp):  # what the parser would emit without the REACH-step-1 assert
        n = 0
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn)):
            n += sum(d.executed for d in t.decisions[DECISION_REACH_IDX])
            t.decisions[DECISION_REACH_IDX].clear()
        found(n, "an executed riichi")

    riichi_game = f if has_riichi(*process(f)) else _game_with(has_riichi)
    expect_failure("riichi offers dropped", "riichi_label_mismatch", drop_riichi_offers, game=riichi_game)
    try:  # L10: an executed riichi that was not offered stops the parse
        with mock.patch.object(legal, "riichi_allowed", lambda *a: False):
            process(riichi_game)
        raise SystemExit("self-test FAILED: an executed, un-offered riichi parsed without the REACH assert")
    except AssertionError as e:
        assert "riichi" in str(e), e
        print("self-test OK: an executed, un-offered riichi raises at REACH step 1")

    def wrong_riichi_drawn_tile(H, T, tp):
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn) and t.stateObj.reaches[t.player]):
            so = t.stateObj
            so.drawn_tile = [(so.drawn_tile[0] + 1) % 34]
            return
        found(0, "a draw by a player in riichi")

    expect_failure("riichi-turn drawn tile corrupted", "state_xml_mismatch", wrong_riichi_drawn_tile, game=riichi_game)

    def extra_during_bit(H, T, tp):
        def edit(td, tdisc, tpost):
            if len(td[3]):
                off = [j for j in range(2, DURING_TURN_ACTION_DIM) if td[2][0, j] == 0]
                td[2][0, off[0]] = 1.0
        _patch_tensors(tp, edit)

    def cleared_post_call_bit(H, T, tp):
        def edit(td, tdisc, tpost):
            for i in range(len(tpost[3])):
                on = [j for j in range(2, 154) if tpost[2][i, j] == 1 and j != tpost[3][i].item()]
                if on:
                    tpost[2][i, on[0]] = 0.0
                    return
        _patch_tensors(tp, edit)

    expect_failure("built during mask gains a bit", "during_mask_mismatch", extra_during_bit)
    expect_failure("built post mask loses a call bit", "post_mask_mismatch", cleared_post_call_bit,
                   game=g["call_choice"])

    def chi_in_chankan_window(H, T, tp):
        for halfturns in H:
            for i, t in enumerate(halfturns):
                prev = halfturns[i - 1] if i else None
                if isinstance(t, PostTurn) and isinstance(prev, DuringTurn) and any(
                        d.executed and d.naki.is_kakan() for d in prev.decisions[DECISION_NAKI_IDX]):
                    p = (t.player + 1) % 4
                    t.decisions[p][DECISION_NAKI_IDX].append(NakiDecision(p, Naki.from_chi_info(0, 0, 0, False), False))
                    return
        found(0, "a kakan")

    def drop_call_and_responder(H, T, tp):  # a legal, unexecuted call removed together with its responder state
        for t in (t for halfturns in H for t in halfturns if isinstance(t, PostTurn)):
            for p in range(4):
                calls = [d for d in t.decisions[p][DECISION_NAKI_IDX] if not d.executed]
                if calls and not any(d.executed for ds in t.decisions[p] for d in ds):
                    t.decisions[p][DECISION_NAKI_IDX].remove(calls[0])
                    del t.responder_states[p]
                    return
        found(0, "an unexecuted call")

    expect_failure("legal call removed with its responder state", ("post_mask_mismatch", "state_xml_mismatch"),
                   drop_call_and_responder)
    expect_failure("chi offered in a chankan window", ("call_offer_in_chankan", "post_mask_mismatch"),
                   chi_in_chankan_window, game=g["chankan"])

    # ---- legal discard mask (T-031); every fixture predicate uses the oracle context, never after_call
    def declaration_restricted(H, T):
        return any(lg.context == "declaration" and lg.legal < held(t.stateObj) for t, lg in legal_samples(H, T))

    def chi_holds_suji(H, T):
        return any(lg.context == "chi" and (lg.forbidden - {lg.called}) & held(t.stateObj) for t, lg in legal_samples(H, T))

    def call_holds_forbidden(H, T):
        return any(lg.context in ("chi", "pon") and lg.forbidden & held(t.stateObj) for t, lg in legal_samples(H, T))

    def red_only_legal_five(H, T):
        return any(lg.context == "plain" and t.stateObj.hand_tensor[red] > 0
                   and t.stateObj.hand_tensor[plain] == t.stateObj.hand_tensor[red]
                   for t, lg in legal_samples(H, T) for red, plain in RED_TO_PLAIN.items())

    def declaration_presence_mask(H, T, tp):  # the old mask on a declaration: every held tile
        _patch_mask(tp, lambda turn, m: torch.from_numpy(legal.presence_mask(turn.stateObj.hand_tensor))
                    if turn.is_riichi_declaration else m)

    def drop_suji_ban(H, T, tp):  # after a chi, only the called kind is forbidden
        def edit(turn, m):
            call = turn.after_call
            if call is None or not call.is_chi():
                return m
            color, number, which, *_ = call.pattern_chi()
            called = 9 * color + number + which
            m = torch.from_numpy(legal.presence_mask(turn.stateObj.hand_tensor))
            m[called] = 0.0
            if called in PLAIN_TO_RED:
                m[PLAIN_TO_RED[called]] = 0.0
            return m
        _patch_mask(tp, edit)

    def clear_after_call(H, T, tp):  # the parser forgets the call: no kuikae at all
        n = 0
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DiscardTurn) and t.after_call is not None):
            t.after_call = None
            n += 1
        found(n, "a discard after a chi / pon")

    def first_sample_only(edit):  # edit(turn, m) -> m or None (skip this sample); applied to one sample
        done = []
        def once(turn, m):
            if done:
                return m
            out = edit(turn, m)
            if out is None:
                return m
            done.append(True)
            return out
        return once

    def zero_legal_plain_slot(H, T, tp):  # an ordinary sample loses a legal, non-target tile
        def edit(turn, m):
            if turn.is_riichi_declaration or turn.after_call is not None:
                return None
            target = TILE2IDX[turn.discarded_tile][-1]
            others = [j for j in np.flatnonzero(m.numpy()).tolist() if j != target]
            if not others:
                return None
            m[others[0]] = 0.0
            return m
        _patch_mask(tp, first_sample_only(edit))

    def zero_target_slot(H, T, tp):
        def edit(turn, m):
            if m.sum() < 2:
                return None
            m[TILE2IDX[turn.discarded_tile][-1]] = 0.0
            return m
        _patch_mask(tp, first_sample_only(edit))

    def enable_unheld_slot(H, T, tp):
        def edit(turn, m):
            m[int(np.flatnonzero(turn.stateObj.hand_tensor == 0)[0])] = 1.0
            return m
        _patch_mask(tp, first_sample_only(edit))

    def empty_mask(H, T, tp):
        _patch_mask(tp, first_sample_only(lambda turn, m: m.zero_()))

    def offer_riichi_off_tenpai(H, T, tp):  # a riichi offered on a hand with no tenpai-keeping discard
        # the turn is picked with the gate's own oracle: this tests the gate's wiring, not the oracle (that is
        # covered by riichi_mask_mismatch, which compares it with an independent shanten-based mask)
        for t in (t for halfturns in H for t in halfturns if isinstance(t, DuringTurn)):
            so = t.stateObj
            if not t.decisions[DECISION_REACH_IDX] and not legal_oracles.tenpai_kinds_by_agari(so.hand_tensor, so.melds[t.player]):
                t.decisions[DECISION_REACH_IDX].append(ReachDecision(t.player, executed=False))
                return
        found(0, "a DuringTurn whose hand has no tenpai-keeping discard")

    def tsumogiri_declaration(H, T):  # a declaration whose discard is the drawn tile: the flag alone decides its fate
        return any(t.is_riichi_declaration and (tokens[t.encoding_idx] >> 7) & 1
                   for halfturns, tokens in zip(H, T) for t in halfturns if isinstance(t, DiscardTurn))

    def unflag_tsumogiri_declaration(H, T, tp):  # the sample would vanish from the encoder AND the old gate set
        for halfturns, tokens in zip(H, T):
            for t in (t for t in halfturns if isinstance(t, DiscardTurn)):
                if t.is_riichi_declaration and (tokens[t.encoding_idx] >> 7) & 1:
                    t.is_riichi_declaration = False
                    return
        found(0, "a tsumogiri riichi declaration")

    def zero_red_only_five(H, T, tp):  # a red-only five (plain slot 0) loses its red slot too
        def edit(turn, m):
            hand = turn.stateObj.hand_tensor
            reds = [red for red, plain in RED_TO_PLAIN.items() if hand[red] > 0 and hand[plain] == hand[red] and m[red] == 1]
            if not reds:
                return None
            m[reds[0]] = 0.0
            return m
        _patch_mask(tp, first_sample_only(edit))

    # ---- win options (T-033)
    def rerun(game, *patchers):
        """Re-run `game` under mock patchers and return run()'s counters: a parser exception (an unoffered real win,
        a library error) counts as parse_error, as on real data."""
        def corrupt(H, T, tp):
            with contextlib.ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                return run([game], tp)
        return corrupt

    def win_context_with(**forced):  # every WinContext the parser builds, with some flags forced
        return mock.patch.object(process_xml, "WinContext", lambda **kw: agari.WinContext(**{**kw, **forced}))

    def tracker(cls):
        return mock.patch.object(process_xml, "FuritenTracker", cls)

    def spy_rons(seen):  # evaluate_ron that records the offered seats and the chankan flag of the latest window
        orig = process_xml.evaluate_ron

        def spy(discarder, kind, hands, melds, furiten, contexts):
            rons, blocked = orig(discarder, kind, hands, melds, furiten, contexts)
            seen["offered"] = {p for p, d in enumerate(rons) if d is not None}
            seen["chankan"] = any(c.is_chankan for c in contexts.values())
            return rons, blocked
        return mock.patch.object(process_xml, "evaluate_ron", spy)

    class NoTempFuriten(agari.FuritenTracker):  # a pass makes only a riichi player furiten
        def on_offer_passed(self, kind, responders, reaches):
            super().on_offer_passed(kind, [q for q in responders if reaches[q]], reaches)

    class TempNeverCleared(agari.FuritenTracker):
        def set_hand(self, p, concealed34):
            temp = self._temp[p]
            super().set_hand(p, concealed34)
            self._temp[p] = temp

    class NoRiichiFuriten(agari.FuritenTracker):
        def on_offer_passed(self, kind, responders, reaches):
            super().on_offer_passed(kind, responders, [0] * 4)

    def pass_filter(seen, keep):  # a tracker that counts a pass only when keep(seen, responder)
        class Filtered(agari.FuritenTracker):
            def on_offer_passed(self, kind, responders, reaches):
                super().on_offer_passed(kind, [q for q in responders if keep(seen, q)], reaches)
        return Filtered

    def offered_kind_only():  # the old rule: discard furiten only when the OFFERED kind is among the own discards
        orig = process_xml.evaluate_ron

        def old(discarder, kind, hands, melds, furiten, contexts):
            class Old:
                def reason(self, p):
                    if kind in furiten._discards[p]:
                        return agari.FURITEN_DISCARD
                    return agari.FURITEN_RIICHI if furiten._riichi[p] else agari.FURITEN_TEMP if furiten._temp[p] else None
            return orig(discarder, kind, hands, melds, Old(), contexts)
        return mock.patch.object(process_xml, "evaluate_ron", old)

    def partial_discard_furiten(H, T):  # a block on 'discard' whose answered kind is NOT among the seat's own discards
        return any(r == agari.FURITEN_DISCARD and not t.responder_states[p].sutehai_tensors[p, answered_kind(tokens[t.encoding_idx - 1])]
                   for halfturns, tokens in zip(H, T) for t in halfturns if isinstance(t, PostTurn)
                   for p, r in t.ron_blocked.items())

    def ron_kept_while_blocked(H, T, tp):
        for t in (t for halfturns in H for t in halfturns if isinstance(t, PostTurn)):
            for p in t.ron_blocked:
                t.decisions[p][DECISION_AGARI_IDX].append(AgariDecision(p, False, yaku=("Riichi",)))
                return
        found(0, "a PostTurn with a blocked responder")

    def win_slot_without_offer(head):
        def corrupt(H, T, tp):
            _patch_tensors(tp, lambda td, tdisc, tpost: {"during": td, "post": tpost}[head][2][:, 1].fill_(1.0))
        return corrupt

    seen = {}
    seat_wind_game = _named_game(SEAT_WIND_HOUTEI_LOG)
    rinshan_game = _named_game(RINSHAN_LOG)
    expect_failure("seat wind shared by all four seats (old player_wind)", "yaku_oracle_mismatch",
                   rerun(seat_wind_game, mock.patch.object(process_xml, "seat_wind", lambda p, oya: agari.EAST + oya)),
                   game=seat_wind_game)
    expect_failure("rinshan flag off", "yaku_oracle_mismatch", rerun(rinshan_game, win_context_with(is_rinshan=False)),
                   game=rinshan_game)
    last_rinshan = _named_game(LAST_RINSHAN_TSUMO_LOG)
    expect_failure("haitei on a last-tile rinshan draw (old rule)", "parse_error",
                   rerun(last_rinshan, mock.patch.object(process_xml, "_is_haitei", lambda r, rinshan: r == 0)),
                   game=last_rinshan)
    expect_failure("houtei off on rons", "yaku_oracle_mismatch",
                   rerun(seat_wind_game, mock.patch.object(process_xml, "_is_houtei", lambda r, chankan, after: False)),
                   game=seat_wind_game)
    houtei_after_rinshan = _named_game(LAST_RINSHAN_HOUTEI_LOG)
    expect_failure("houtei off after a last-tile rinshan", "parse_error",
                   rerun(houtei_after_rinshan, mock.patch.object(
                       process_xml, "_is_houtei", lambda r, chankan, after: r == 0 and not chankan and not after)),
                   game=houtei_after_rinshan)
    for name, fixture, flag in (("chiihou", CHIIHOU_LOG, "is_chiihou"), ("tenhou", TENHOU_LOG, "is_tenhou")):
        game = _named_game(fixture)
        expect_failure(f"{name} flag off", "yaku_oracle_mismatch", rerun(game, win_context_with(**{flag: False})), game=game)
    suuankou = _named_game(SUUANKOU_LOG)

    def suuankou_ids_swapped(H, T, tp):  # the library's own ids: Suuankou 41, SuuankouTanki 40
        with mock.patch.dict(win_oracles.TENHOU_YAKU_ID, {"Suuankou": 41, "SuuankouTanki": 40}):
            return _count(suuankou, H, T, tp)
    expect_failure("suuankou id override removed", "yaku_oracle_mismatch", suuankou_ids_swapped, game=suuankou)
    expect_failure("yaku check off (shape only)", "win_offer_without_yaku",
                   rerun(f, mock.patch.object(agari, "win_yaku", lambda *a: ())))
    partial = _game_with(partial_discard_furiten)
    expect_failure("furiten on the offered kind only (old rule)", "furiten_mismatch", rerun(partial, offered_kind_only()),
                   game=partial)
    temp_game = _named_game(TEMP_FURITEN_LOG)
    expect_failure("same-turn furiten off", "furiten_mismatch", rerun(temp_game, tracker(NoTempFuriten)), game=temp_game)
    expect_failure("same-turn furiten only after an offered ron (no-yaku pass ignored)", "furiten_mismatch",
                   rerun(rinshan_game, spy_rons(seen), tracker(pass_filter(seen, lambda s, q: q in s["offered"]))),
                   game=rinshan_game)
    expect_failure("same-turn furiten never cleared", "furiten_mismatch", rerun(rinshan_game, tracker(TempNeverCleared)),
                   game=rinshan_game)
    riichi_furiten = _named_game(RIICHI_FURITEN_LOG)
    expect_failure("riichi furiten off", "furiten_mismatch", rerun(riichi_furiten, tracker(NoRiichiFuriten)),
                   game=riichi_furiten)
    expect_failure("ron offer kept while blocked", "win_reason_conflict", ron_kept_while_blocked,
                   game=_game_with(lambda H, T: any(t.ron_blocked for hs in H for t in hs if isinstance(t, PostTurn))))
    for head in ("during", "post"):
        expect_failure(f"{head} win slot enabled with no offer", "win_mask_mismatch", win_slot_without_offer(head))

    def wins_dropped(H, T, tp):  # xml_rounds() misses every <AGARI>: yaku_coverage shrinks on both sides and passes
        orig = xml_rounds
        with mock.patch(f"{__name__}.xml_rounds", lambda raw: [dict(r, wins=[]) for r in orig(raw)]):
            return _count(f, H, T, tp)
    expect_failure("xml_rounds drops the wins", "agari_coverage", wins_dropped)
    expect_failure("passed chankan ignored", "furiten_mismatch",
                   rerun(CHANKAN_PASS_GAME, spy_rons(seen), tracker(pass_filter(seen, lambda s, q: not s["chankan"]))),
                   game=CHANKAN_PASS_GAME)

    # ---- kyuushu kyuuhai (T-034)
    yao9_game, after_call_game = kyuushu_games
    drawn_abort_game = _named_game(NINTH_KIND_DRAWN_ABORT_LOG)
    has_declined = lambda H, T: any(t.decisions[DECISION_ABORT_IDX] and not t.decisions[DECISION_ABORT_IDX][0].executed
                                    for t in _durings(H))
    drawn_declined_game = _parsed_fixture(NINTH_KIND_DRAWN_DECLINED, has_declined)
    declined_game = _parsed_fixture(KYUUSHU_DECLINED, has_declined)
    kyuushu_legal = legal.kyuushu_legal

    def ignore_calls(hand34, first_draw):  # the seat's own first draw is enough, a call before it is ignored
        loc = sys._getframe(1).f_locals
        return kyuushu_legal(hand34, loc["own_discards"][loc["player"]] == 0)

    def thirteen_tiles(hand34, first_draw):  # the kinds counted before the draw (13 tiles)
        h = np.array(hand34, copy=True)
        h[sys._getframe(1).f_locals["tile_idx"][0]] -= 1
        return kyuushu_legal(h, first_draw)

    def skip_yao9_marking(H, T, tp):
        with mock.patch.object(process_xml, "_mark_yao9_executed", lambda halfturns, attr: None):
            H[:], T[:] = process(yao9_game)

    def built_during(pick):  # pick(mask_row, target) -> True for the ONE built during sample to edit; edits once
        done = []
        def patch(tp, edit):
            def wrapped(td, tdisc, tpost):
                for i in range(len(td[3])):
                    if not done and pick(td[2][i], td[3][i].item()):
                        edit(td, i)
                        done.append(i)
            _patch_tensors(tp, wrapped)
        return patch

    K = legal_oracles.KYUUSHU_IDX

    def zero_kyuushu_targets(H, T, tp):
        _patch_tensors(tp, lambda td, tdisc, tpost: td[3].masked_fill_(td[3] == K, 0))

    def clear_bit(pick):
        return lambda H, T, tp: built_during(pick)(tp, lambda td, i: td[2][i].__setitem__(K, 0.0))

    def set_bit(H, T, tp):
        built_during(lambda m, y: m[K] == 0)(tp, lambda td, i: td[2][i].__setitem__(K, 1.0))

    def abort_moved_to_declined(H, T, tp):
        offers = [t.decisions[DECISION_ABORT_IDX][0] for t in _durings(H) if t.decisions[DECISION_ABORT_IDX]]
        executed, declined = [d for d in offers if d.executed], [d for d in offers if not d.executed]
        found(executed and declined, "an executed and a declined kyuushu offer in one game")
        executed[0].executed, declined[0].executed = False, True

    expect_failure("yao9 abort left unlabelled", ("lab_kyuushu", "kyuushu_label_mismatch"), skip_yao9_marking,
                   game=yao9_game)
    expect_failure("kyuushu rule ignores calls", ("kyuushu_offer_after_call", "during_mask_mismatch"),
                   reparse(after_call_game, kyuushu_legal=ignore_calls), game=after_call_game)
    expect_failure("kyuushu threshold 8 kinds", ("kyuushu_offer_few_kinds", "during_mask_mismatch"),
                   reparse(f, KYUUSHU_MIN_KINDS=8))
    expect_failure("kyuushu kinds counted on 13 tiles (declined offer lost)", "during_mask_mismatch",
                   reparse(drawn_declined_game, kyuushu_legal=thirteen_tiles), game=drawn_declined_game)
    expect_failure("kyuushu kinds counted on 13 tiles (abort not offered)", "parse_error",
                   rerun(drawn_abort_game, mock.patch.object(legal, "kyuushu_legal", thirteen_tiles)),
                   game=drawn_abort_game)
    expect_failure("zeroed kyuushu targets", ("tensor_kyuushu", "during_target_mismatch"), zero_kyuushu_targets,
                   game=yao9_game)
    expect_failure("abort slot cleared on the executed abort", ("during_label_outside_mask", "during_mask_mismatch"),
                   clear_bit(lambda m, y: y == K), game=yao9_game)
    expect_failure("abort slot cleared on a declined offer", "during_mask_mismatch",
                   clear_bit(lambda m, y: m[K] == 1 and y != K), game=declined_game)
    expect_failure("abort slot set on a sample with no offer", "during_mask_mismatch", set_bit, game=yao9_game)
    expect_failure("executed abort moved to a declined offer", "kyuushu_label_mismatch", abort_moved_to_declined,
                   game=after_call_game)

    expect_failure("declaration mask = every held tile", "riichi_mask_mismatch", declaration_presence_mask,
                   game=_game_with(declaration_restricted))
    expect_failure("suji kuikae ban dropped after a chi", "kuikae_mask_mismatch", drop_suji_ban,
                   game=_game_with(chi_holds_suji))
    expect_failure("after_call cleared on post-call discards", "kuikae_mask_mismatch", clear_after_call,
                   game=_game_with(call_holds_forbidden))
    expect_failure("legal tile removed from an ordinary sample", "discard_mask_unrestricted_mismatch", zero_legal_plain_slot)
    expect_failure("executed discard masked out", "discard_target_illegal", zero_target_slot)
    expect_failure("unheld tile enabled", "discard_mask_superset", enable_unheld_slot)
    expect_failure("all-zero discard mask", "discard_mask_empty", empty_mask)
    expect_failure("riichi offered with no tenpai-keeping discard", "riichi_offer_without_legal_discard", offer_riichi_off_tenpai)
    expect_failure("red-only five's red slot cleared", "discard_mask_red_rule", zero_red_only_five,
                   game=_game_with(red_only_legal_five))
    unflag_game = _game_with(tsumogiri_declaration)
    expect_failure("declaration flag cleared (tsumogiri)", "riichi_declaration_unflagged", unflag_tsumogiri_declaration,
                   game=unflag_game)
    expect_failure("declaration flag cleared (tsumogiri): sample lost", "discard_sample_count_mismatch",
                   unflag_tsumogiri_declaration, game=unflag_game)

    expect_failure("un-labelled ron", "lab_ron", unlabel_ron)
    expect_failure("triple ron left unlabelled", "lab_ron", skip_ron3_marking, game=ron3)
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
    expect_failure("open-kan indicator in the kan player's discard state", ("state_xml_dora", "pov_xml_mismatch",
                   "dora_draw_to_discard"), open_kan_dora_seen(with_dora=True), game=open_kan)
    expect_failure("open-kan indicator in the discard state's unseen counts only", "pov_xml_mismatch",
                   open_kan_dora_seen(with_dora=False), game=open_kan)
    expect_failure("open-kan indicator token in the discard history", "dora_draw_to_discard", open_kan_token_seen,
                   game=open_kan)
    expect_failure("ron winner misses the open-kan indicator", "dora_vs_agari", ron_misses_kan_dora, game=open_kan_ron)
    expect_failure("a win left out of the doraHai check", "dora_vs_agari_coverage", unlabel_ron, game=open_kan_ron)
    expect_failure("zeroed ron targets", "tensor_ron", zero_ron_targets)
    expect_failure("zeroed riichi targets", "tensor_riichi", zero_riichi_targets, game=riichi_game)
    expect_failure("chi target moved to another chi", "post_target_mismatch", other_chi_target, game=_game_with(has_chi_choice))
    expect_failure("pre-empted responder kept", "post_sample_count_mismatch", keep_preempted, game=_game_with(has_preempted))
    expect_failure("shifted hand plane", "hand_plane_mismatch", shifted_hand_plane)
    expect_failure("zeroed unseen-count plane", "x1_plane_mismatch", zero_unseen_plane)
    expect_failure("zeroed score row", "x2_row_mismatch", zero_score_row)
    expect_failure("zeroed x3 scalars", "x3_scalar_mismatch", zero_scalars)
    expect_failure("bakaze one-hot rolled by one slot", ("round_wind_xml", "x3_scalar_mismatch"), rolled_bakaze,
                   game=tonpuu)
    expect_failure("parser stores East in every round", "round_wind_xml", parser_wind_east, game=west,
                   absent=("x3_scalar_mismatch",))
    expect_failure("tonpuu wind by the hanchan formula", "round_wind_xml", hanchan_wind_formula, game=tonpuu)
    expect_failure("bakaze of the post head unchecked", "round_wind_coverage", post_wind_unchecked, game=tonpuu)
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
    ap.add_argument("--list-declined", action="store_true", help="print every win offer that was not taken")
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
    if args.list_declined:
        for f in files:
            try:
                H, _ = process(f)
            except Exception:
                continue
            for ri, i, seat, kind, yaku, scores in declined_offers(H):
                print(f"declined {kind}: {os.path.basename(f)} round {ri} halfturn {i} seat {seat} yaku {yaku} scores {scores}")
    check(C)
