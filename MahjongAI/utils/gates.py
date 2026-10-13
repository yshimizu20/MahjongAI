"""Pre-training gate checks: the parsed data must agree with the XML ground truth and with itself.

    python -m MahjongAI.utils.gates --games 150      # random games from DEFAULT_DAYS; every gate is an assert
    python -m MahjongAI.utils.gates --test           # self-test: passes on a clean game, raises on corrupted ones
Run from the directory that contains MahjongAI/ (python -I would drop it from sys.path).

A clean run prints ONE line. A failing gate raises AssertionError with the offending counts.
Gate groups (counted over every game, round, halfturn and built tensor):
  WINS    executed tsumo / ron labels == <AGARI> events in the XML (who == fromWho is a tsumo), plus 3 rons
          per triple-ron abort (<RYUUKYOKU type="ron3">)
  CALLS   executed chi / pon / minkan / ankan / kakan labels == <N> events by kind
  MASK    every label index is inside its action mask (during 71, discard 37, post 154); chi/pon indices injective
  POV     a PostTurn sample's state belongs to the responder; the answered discard / kakan token is the last
          history token; unseen counts + own hand <= 135; no negative counts; no state shares a buffer
  TOKENS  token 0 of every round is the INIT dora indicator; a riichi declaration discard follows its own reach
          token; a discard by a player in riichi is the drawn tile; seat bits rotate so that the actor is seat 0
  SCALE   remaining_tsumo in [0, 70] and == 0 at an exhaustive draw; |x2|, |x3| <= 5 (normalised inputs;
          the whole 100,000-point pool on one seat is exactly 4.0)
  PLANES  the safe-against-riichi planes equal the set rebuilt from the token history alone
  IPPATSU a riichi declarer's next DuringTurn has ippatsu = 1 (0 if a call came between); his next discard
          and his DuringTurn after that have 0
  RED     every state's red-five slots 34-36 of remaining_tiles_pov == 1 - (that red visible to the player:
          own hand, any discard, any meld, a dora indicator)
  TENSORS the BUILT tensors agree with the parser and the XML: win / call targets per head == XML events and
          every during / post target == the executed action of its sample (exact index, not only the kind);
          discard targets == TILE2IDX (red five -> slot 34-36) and the plain-five slot is enabled only
          with a plain copy; the hand plane of every sample is its decider's hand; the post sample set
          follows the priority rule (executors always, pre-empted responders never); the answered event
          is the last history token and nothing after it is visible; the tensor builder skipped nothing
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
from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.turn import DuringTurn, DiscardTurn, PostTurn
from MahjongAI.draw import Naki
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_NAKI_IDX, DECISION_REACH_IDX, EventStateTypes, EventTypes, TILE2IDX
from MahjongAI.agents.encoder_conv import TransformerTensorProcessor, N_X1_PLANES

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PACKAGE_DIR, "data", "processed")
SAMPLE_LOG = os.path.join(PACKAGE_DIR, "data", "sample", "sample_haifu.xml")
RON3_LOG = "2021010521gm-00a9-0000-e76b4550.log"  # 2021-01-05, round 5 is a triple-ron abort
DEFAULT_DAYS = ["20210103", "20210214", "20210322", "20210430", "20210518", "20210626",
                "20210709", "20210815", "20210921", "20211030", "20211112", "20211225"]
KINDS = ["chi", "pon", "minkan", "ankan", "kakan"]
HEADS = ["during", "discard", "post"]
ZERO_GATES = [
    "parse_error", "tensor_build_error",
    "post_wrong_pov", "post_answered_event_hidden", "pov_negative", "pov_sum_too_big", "pov_shares_memory_rounds",
    "index_collisions", "index_spill_into_pon",
    "first_token_not_dora", "riichi_declaration_unflagged", "riichi_discard_not_tsumogiri", "rotation_mismatch",
    "tsumo_negative", "tsumo_over_70", "exhaustive_draw_tsumo_not_zero", "x_scale_over_5", "x2_not_actor_first",
    "x1_plane_count", "safe_plane_mismatch", "discard_sample_count_mismatch",
    "tensor_skipped", "discard_label_mismatch", "discard_mask_red_rule", "hand_plane_mismatch",
    "post_sample_count_mismatch", "post_history_last_token",
    "during_sample_count_mismatch", "during_target_mismatch", "post_target_mismatch",
    "ippatsu_window_wrong", "red_pov_mismatch",
]
RED_TO_PLAIN = {34: 4, 35: 13, 36: 22}


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


def safe_from_history(tokens, q):
    """34-index tiles safe against q, rebuilt from the token history alone. Empty set if q is not in riichi."""
    reach_pos = next((i for i, t in enumerate(tokens) if _is_reach(t) and _who(t) == q), None)
    if reach_pos is None:
        return set()
    safe = {_tile(t) for t in tokens if _is_discard(t) and _who(t) == q}
    safe |= {_tile(t) for i, t in enumerate(tokens) if i > reach_pos and _is_discard(t)}
    return {RED_TO_PLAIN.get(t, t) for t in safe}  # a red-five discard token carries slot 34-36


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


def thermometer(counts) -> torch.Tensor:
    """(37,) counts -> (37, 4) plane with column j = 1 iff count > j; the layout of every x1 plane."""
    return (torch.arange(4) < torch.as_tensor(np.asarray(counts), dtype=torch.long).unsqueeze(1)).float()


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


def xml_round_ends(raw: str):
    """Per round: ('agari' | 'exhaustive' | 'abortive', dora_indicator_tile) in log order."""
    ends = []
    for m in re.finditer(r"<(INIT|AGARI|RYUUKYOKU)\b([^>]*)>", raw):
        tag, attrs = m.group(1), m.group(2)
        if tag == "INIT":
            dora = int(re.search(r'seed="([^"]*)"', attrs).group(1).split(",")[5])
            ends.append([None, dora])
        elif tag == "AGARI":
            ends[-1][0] = "agari"
        else:
            ends[-1][0] = "abortive" if "type=" in attrs else "exhaustive"
    return ends


def check_state(so, hist, C, tp):
    """Built-tensor invariants of one StateObject against its own history prefix."""
    x1, x2, x3 = tp._convert_state_obj_to_tensors(so)
    me = so.player
    if x1.shape[1] != N_X1_PLANES:
        C["x1_plane_count"] += 1
    if x2[0, 4, 0].item() != so.reaches[me] or x2[0, 1, 0].item() != so.parent_tensor[me]:
        C["x2_not_actor_first"] += 1
    if max(x2.abs().max().item(), x3.abs().max().item()) > 5:
        C["x_scale_over_5"] += 1
    for d in (1, 2, 3):
        q = (me + d) % 4
        got = set(np.flatnonzero(x1[0, 6 + d, :34, 0].numpy()).tolist())
        if got != safe_from_history(hist, q):
            C["safe_plane_mismatch"] += 1


def count_game(path, H, T, C, tp):
    """Add every counter of one parsed game (H = halfturns per round, T = tokens per round) to C."""
    raw = open(path, "rb").read().decode("utf-8", "ignore")
    ag = re.findall(r'<AGARI [^>]*?who="(\d)"[^>]*?fromWho="(\d)"', raw)
    C["xml_tsumo"] += sum(1 for w, fw in ag if w == fw)
    C["xml_ron"] += sum(1 for w, fw in ag if w != fw)
    C["xml_ron"] += 3 * len(re.findall(r'<RYUUKYOKU [^>]*type="ron3"', raw))  # triple ron: three rons, no AGARI
    for m in re.findall(r'<N who="\d" m="(\d+)"', raw):
        C["xml_" + kind_of(Naki(int(m)))] += 1
    ends = xml_round_ends(raw)
    assert len(ends) == len(H), f"{path}: {len(ends)} rounds in XML, {len(H)} parsed"

    for (end, dora), halfturns, tokens in zip(ends, H, T):
        C["rounds"] += 1
        if not (_event(tokens[0]) == EventStateTypes.NEW_DORA and _tile(tokens[0]) == dora // 4):
            C["first_token_not_dora"] += 1
        durings = [t for t in halfturns if isinstance(t, DuringTurn)]
        if end == "exhaustive" and durings and durings[-1].stateObj.remaining_tsumo != 0:
            C["exhaustive_draw_tsumo_not_zero"] += 1
        if len(durings) > 1 and np.shares_memory(durings[0].stateObj.remaining_tiles_pov, durings[-1].stateObj.remaining_tiles_pov):
            C["pov_shares_memory_rounds"] += 1
        tok_t = torch.tensor(tokens, dtype=torch.int64)
        for actor in range(4):
            rot = tp._relative_seats(tok_t, actor).tolist()
            for t, r in zip(tokens, rot):
                want = (t & ~(3 << 8)) | (((_who(t) - actor) % 4) << 8) if _has_seat(t) else t
                if r != want:
                    C["rotation_mismatch"] += 1

        exp_during = []  # (target, hand) per during sample (a draw with at least one decision), in order
        exp_disc = []  # (target, hand) per free discard, in order
        exp_post = []  # (encoding_idx, responder, discarder, hand, target) per expected post sample, in order
        for t in halfturns:
            so = t.stateObj
            if (so.remaining_tiles_pov < 0).any():
                C["pov_negative"] += 1
            if so.remaining_tsumo < 0:
                C["tsumo_negative"] += 1
            if so.remaining_tsumo > 70:
                C["tsumo_over_70"] += 1
            if so.remaining_tiles_pov[:34].sum() + so.hand_tensor[:34].sum() > 135:
                C["pov_sum_too_big"] += 1
            check_red_pov(so, C)

            if isinstance(t, DuringTurn):
                if sum(map(len, t.decisions)):
                    exp_during.append((expected_during_target(t), so.hand_tensor))
                if t.decisions[DECISION_AGARI_IDX] and t.decisions[DECISION_AGARI_IDX][0].executed:
                    C["lab_tsumo"] += 1
                for d in t.decisions[DECISION_NAKI_IDX]:
                    if d.executed:
                        C["lab_" + kind_of(d.naki)] += 1
                check_state(so, tokens[: t.encoding_idx], C, tp)

            elif isinstance(t, DiscardTurn):
                if not (so.reaches[t.player] and not t.is_riichi_declaration):
                    exp_disc.append((TILE2IDX[t.discarded_tile][-1], so.hand_tensor))
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
                check_state(so, tokens[: t.encoding_idx], C, tp)

            elif isinstance(t, PostTurn):
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
                    check_state(rs, tokens[: t.encoding_idx], C, tp)
                    check_red_pov(rs, C)
                for p, target in expected_responders(t):
                    exp_post.append((t.encoding_idx, p, t.player, t.responder_states[p].hand_tensor, target))

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
            if len(tt[3]) == 0:
                continue
            C[f"{name}_label_outside_mask"] += int((tt[2][np.arange(len(tt[3])), tt[3]] != 1).sum().item())
            C[f"{name}_samples"] += len(tt[3])
        tensor_counts(td[3].tolist(), tpost[3].tolist(), C)

        if len(td[3]) != len(exp_during):
            C["during_sample_count_mismatch"] += 1
        else:
            for i, (target, hand) in enumerate(exp_during):
                if td[3][i].item() != target:
                    C["during_target_mismatch"] += 1
                if not torch.equal(td[1][0][i, 0], thermometer(hand)):
                    C["hand_plane_mismatch"] += 1

        if len(tdisc[3]) != len(exp_disc):
            C["discard_sample_count_mismatch"] += 1
        else:
            for i, (target, hand) in enumerate(exp_disc):
                if tdisc[3][i].item() != target:
                    C["discard_label_mismatch"] += 1
                m = tdisc[2][i]
                for red, plain in RED_TO_PLAIN.items():
                    if m[plain].item() != float(hand[plain] - hand[red] > 0) or m[red].item() != float(hand[red] > 0):
                        C["discard_mask_red_rule"] += 1
                if not torch.equal(tdisc[1][0][i, 0], thermometer(hand)):
                    C["hand_plane_mismatch"] += 1

        if len(tpost[3]) != len(exp_post):
            C["post_sample_count_mismatch"] += 1
        else:
            for i, (eidx, p, discarder, hand, target) in enumerate(exp_post):
                if tpost[3][i].item() != target:
                    C["post_target_mismatch"] += 1
                if not torch.equal(tpost[1][0][i, 0], thermometer(hand)):
                    C["hand_plane_mismatch"] += 1
                hist = tpost[0][i]
                tok = int(hist[eidx - 1].item()) if eidx > 0 else 0
                answered = _is_discard(tok) or (_event(tok) == EventStateTypes.DURING and _etype(tok) == EventTypes.NAKI)
                if not (answered and _who(tok) == (discarder - p) % 4 and int(hist[eidx:].abs().sum().item()) == 0):
                    C["post_history_last_token"] += 1


def tensor_counts(y_during, y_post, C):
    """Count the built targets per action kind (the head layouts: during 71, post 154)."""
    C["tensor_tsumo"] += sum(1 for v in y_during if v == 1)
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


def check(C):
    """Raise AssertionError naming the first failed gate; print one line when every gate passes."""
    assert C["games"] > 0, "no game parsed"
    for k in ["tsumo", "ron"] + KINDS:
        assert C[f"lab_{k}"] == C[f"xml_{k}"], f"lab_{k}={C[f'lab_{k}']} != xml_{k}={C[f'xml_{k}']}"
        assert C[f"tensor_{k}"] == C[f"xml_{k}"], f"tensor_{k}={C[f'tensor_{k}']} != xml_{k}={C[f'xml_{k}']}"
    for h in HEADS:
        assert C[f"{h}_samples"] > 0, f"no {h} samples"
        assert C[f"{h}_label_outside_mask"] == 0, f"{h}_label_outside_mask={C[f'{h}_label_outside_mask']}"
    bad = {k: C[k] for k in ZERO_GATES if C[k] != 0}
    assert not bad, f"gates failed: {bad}"
    print(
        f"GATES OK | games {C['games']} (skipped {C['skipped_invalid']}) rounds {C['rounds']} | "
        f"wins tsumo {C['lab_tsumo']}/{C['xml_tsumo']} ron {C['lab_ron']}/{C['xml_ron']} | "
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


def _ron3_game():
    """A known triple-ron game (round 5 ends in <RYUUKYOKU type="ron3">): the 2021-01-01 self-test pool and the
    default --games 150 sample hold none, so the self-test names one explicitly."""
    found = glob.glob(os.path.join(DATA_ROOT, "2021", "20210105", "*", RON3_LOG))
    assert found, f"self-test needs the triple-ron game {RON3_LOG} under {DATA_ROOT}/2021/20210105"
    raw = open(found[0], "rb").read().decode("utf-8", "ignore")
    assert re.search(r'<RYUUKYOKU [^>]*type="ron3"', raw), f"{found[0]} holds no ron3"
    return found[0]


def self_test():
    """Clean games (one with a triple ron) pass; ten corruptions each raise: un-labelled win, wrong responder POV,
    stale safe plane, triple ron left unlabelled, ippatsu cleared before the first draw after riichi, red dora
    indicator left in the red slot (parser level); zeroed win targets, shifted hand plane, pre-empted responder
    kept, chi target moved to another allowed chi (tensor level). Also: a one-loader cluster rolls over at the
    epoch boundary."""
    f = _first_valid_game()
    ron3 = _ron3_game()
    check(run([f, ron3], TransformerTensorProcessor()))

    def expect_failure(name, corrupt, game=f):
        tp = TransformerTensorProcessor()
        H, T = process(game)
        corrupt(H, T, tp)
        C = collections.Counter()
        C["games"] += 1
        count_game(game, H, T, C, tp)
        index_checks(C)
        try:
            check(C)
        except AssertionError as e:
            print(f"self-test OK: {name} -> {e}")
            return
        raise SystemExit(f"self-test FAILED: gate did not fire on {name}")

    def unlabel_win(H, T, tp):
        for halfturns in H:
            for t in halfturns:
                if isinstance(t, PostTurn):
                    for p in range(4):
                        a = t.decisions[p][DECISION_AGARI_IDX]
                        if a and a[0].executed:
                            a[0].executed = False
                            return
                if isinstance(t, DuringTurn):
                    a = t.decisions[DECISION_AGARI_IDX]
                    if a and a[0].executed:
                        a[0].executed = False
                        return
        raise SystemExit("self-test needs a game with a win")

    def wrong_pov(H, T, tp):
        for halfturns in H:
            for t in halfturns:
                if isinstance(t, PostTurn):
                    for p in range(4):
                        if sum(map(len, t.decisions[p])):
                            t.responder_states[p].player = (p + 1) % 4
                            return

    def other_chi_target(H, T, tp):  # an executed chi target moved to ANOTHER enabled chi slot: same kind count
        orig = tp.build_tensors

        def bad(halfturns, tokens):
            td, tdisc, tpost = orig(halfturns, tokens)
            for i in range(len(tpost[3])):
                y = tpost[3][i].item()
                others = [j for j in range(2, 83) if tpost[2][i, j] == 1 and j != y]
                if 2 <= y <= 82 and others:
                    tpost[3][i] = others[0]
                    break
            return td, tdisc, tpost

        tp.build_tensors = bad

    def has_chi_choice(H, T):  # a round where an executed chi had a second chi option
        tp = TransformerTensorProcessor()
        for halfturns, tokens in zip(H, T):
            _, _, tpost = tp.build_tensors(halfturns, tokens)
            for i in range(len(tpost[3])):
                y = tpost[3][i].item()
                if 2 <= y <= 82 and int(tpost[2][i, 2:83].sum().item()) >= 2:
                    return True
        return False

    def stale_safe_plane(H, T, tp):
        for halfturns in H:
            for t in halfturns:
                if isinstance(t, DiscardTurn):
                    t.stateObj.safe_tiles[(t.player + 1) % 4, 0] = 1.0
                    return

    def zero_win_targets(H, T, tp):  # parser labels intact, built targets lose every win
        orig = tp.build_tensors

        def bad(halfturns, tokens):
            td, tdisc, tpost = orig(halfturns, tokens)
            td[3][td[3] == 1] = 0
            tpost[3][tpost[3] == 1] = 0
            return td, tdisc, tpost

        tp.build_tensors = bad

    def shifted_hand_plane(H, T, tp):  # every sample's hand plane is rolled by one tile
        orig = tp._convert_state_obj_to_tensors

        def bad(so):
            x1, x2, x3 = orig(so)
            x1 = x1.clone()
            x1[:, 0] = x1[:, 0].roll(1, dims=1)
            return x1, x2, x3

        tp._convert_state_obj_to_tensors = bad

    def keep_preempted(H, T, tp):  # the old rule: every responder with a decision is a sample
        tp._post_turn_responders = lambda turn: [
            p for p in ((turn.player + k) % 4 for k in (1, 2, 3)) if sum(map(len, turn.decisions[p]))
        ]

    def has_preempted(H, T):
        return any(
            len(expected_responders(t)) < sum(1 for p in range(4) if sum(map(len, t.decisions[p])))
            for halfturns in H for t in halfturns if isinstance(t, PostTurn)
        )

    def skip_ron3_marking(H, T, tp):  # re-parse with the triple-ron labelling switched off
        with mock.patch.object(process_xml, "_mark_ron3_executed", lambda halfturns: None):
            H[:], T[:] = process(ron3)

    def early_ippatsu_clear(H, T, tp):  # the old order: the first draw after riichi cleared ippatsu before its snapshot
        for halfturns, tokens in zip(H, T):
            for p, called, first, _, _ in ippatsu_windows(halfturns, tokens):
                if not called:
                    first.stateObj.ippatsu = [0 if q == p else v for q, v in enumerate(first.stateObj.ippatsu)]
                    return
        raise SystemExit("self-test needs a riichi followed by the declarer's draw")

    def has_ippatsu_window(H, T):
        return any(not called for halfturns, tokens in zip(H, T) for _, called, *_ in ippatsu_windows(halfturns, tokens))

    def red_indicator_kept(H, T, tp):  # the old DORA bug: a red indicator decremented the kind slot only
        for halfturns in H:
            for t in halfturns:
                for s in range(3):
                    if 16 + 36 * s in t.stateObj.dora:
                        t.stateObj.remaining_tiles_pov = t.stateObj.remaining_tiles_pov.copy()
                        t.stateObj.remaining_tiles_pov[34 + s] += 1
                        return
        raise SystemExit("self-test needs a red-five dora indicator")

    def has_red_indicator(H, T):
        return any(16 + 36 * s in t.stateObj.dora for halfturns in H for t in halfturns for s in range(3))

    expect_failure("un-labelled win", unlabel_win)
    expect_failure("wrong responder POV", wrong_pov)
    expect_failure("stale safe plane", stale_safe_plane)
    expect_failure("triple ron left unlabelled", skip_ron3_marking, game=ron3)
    expect_failure("ippatsu cleared before the first draw after riichi", early_ippatsu_clear, game=_game_with(has_ippatsu_window))
    expect_failure("red dora indicator left in the red slot", red_indicator_kept, game=_game_with(has_red_indicator))
    expect_failure("zeroed win targets", zero_win_targets)
    expect_failure("shifted hand plane", shifted_hand_plane)
    expect_failure("pre-empted responder kept", keep_preempted, game=_game_with(has_preempted))
    expect_failure("chi target moved to another chi", other_chi_target, game=_game_with(has_chi_choice))

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
