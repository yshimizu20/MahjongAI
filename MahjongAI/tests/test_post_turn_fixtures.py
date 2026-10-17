"""T-026: named post-turn cases on committed fixture games (tests/data/gate_games), each found by its content with
exactly one match, so a change to round handling cannot move a case silently.

- mixed priority + exact target: an own-red pon pre-empts another responder's chi-only offer; the chi responder is
  no sample (its pass was not observed), and the pon's built tensor row has target and mask slot 110, not the plain 87.
- ron on a riichi declaration discard: the declaration is public at once, but the stick is never paid (step 2 never
  comes), so the score and the kyotaku stay at the round's INIT values.
"""
import glob
import os
import re

import torch

from MahjongAI.agents.encoder_conv import TransformerTensorProcessor
from MahjongAI.draw import Naki
from MahjongAI.process_xml import process
from MahjongAI.turn import DiscardTurn, PostTurn
from MahjongAI.utils import legal_oracles
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_NAKI_IDX

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
MIXED_PRIORITY_LOG = "2021010100gm-00a9-0000-05d6c581.log"  # round 9: seat 3's own-red pon beats seat 1's chi
RIICHI_RON_LOG = "2021010313gm-00a9-0000-5120e1eb.log"  # round 5: seat 1 declares with E134, seat 0 rons it


def _parsed(name):
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
    assert len(found) == 1, (name, found)
    raw = open(found[0], "rb").read().decode("utf-8", "ignore")
    H, T = process(found[0])
    rounds = raw.split("<INIT ")[1:]
    assert len(rounds) == len(H), (name, len(rounds), len(H))
    return rounds, H, T


def _naki_offers(turn, p):
    return [(d.naki.get_post_turn_filter_idx(), d.executed) for d in turn.decisions[p][DECISION_NAKI_IDX]]


def test_mixed_priority_exact_post_target():
    rounds, H, T = _parsed(MIXED_PRIORITY_LOG)
    matches = [(r, t) for r, halfturns in enumerate(H) for t in halfturns if isinstance(t, PostTurn)
               and (110, True) in _naki_offers(t, 3)
               and _naki_offers(t, 1) and all(i < 83 and not x for i, x in _naki_offers(t, 1))
               and not t.decisions[1][DECISION_AGARI_IDX]]
    assert len(matches) == 1, [(r, t.encoding_idx) for r, t in matches]
    r, turn = matches[0]
    assert TransformerTensorProcessor._post_turn_responders(turn) == [3]  # seat 1 is dropped, not a "pass"

    # the log's pon is the own-red pon: same index from the real code
    pons = [int(m) for m in re.findall(r'<N who="3" m="(\d+)"', rounds[r])
            if legal_oracles.decode_meld(int(m)).kind == "pon" and legal_oracles.meld_kind(
                legal_oracles.decode_meld(int(m))) % 9 == 4]
    assert len(pons) == 1, pons
    assert Naki(pons[0]).get_post_turn_filter_idx() == 110

    # the built post tensor: this PostTurn gives exactly one row (its padding length is its encoding_idx), seat 3's
    tp = TransformerTensorProcessor()
    _, _, (tokens, (x1, _, _), mask, y, padding) = tp.build_tensors(H[r], T[r])
    rows = (padding.sum(dim=1) == turn.encoding_idx).nonzero().flatten().tolist()
    assert len(rows) == 1, rows
    row = rows[0]
    assert y[row].item() == 110 and mask[row, 110].item() == 1.0 and mask[row, 87].item() == 0.0
    seat3 = tp._convert_state_obj_to_tensors(turn.responder_states[3])[0][0]
    seat1 = tp._convert_state_obj_to_tensors(turn.responder_states[1])[0][0]
    assert torch.equal(x1[row], seat3) and not torch.equal(x1[row], seat1)


def test_ron_on_riichi_declaration_discard():
    rounds, H, T = _parsed(RIICHI_RON_LOG)
    sequence = re.compile(r'<REACH who="1" step="1"\s*/><E134\s*/><AGARI [^>]*who="0" fromWho="1"')
    matches = [(r, t) for r, halfturns in enumerate(H) for i, t in enumerate(halfturns)
               if isinstance(t, PostTurn) and i > 0 and isinstance(halfturns[i - 1], DiscardTurn)
               and halfturns[i - 1].player == 1 and halfturns[i - 1].discarded_tile == 134
               and halfturns[i - 1].is_riichi_declaration and sequence.search(rounds[r])]
    assert len(matches) == 1, [r for r, _ in matches]
    r, turn = matches[0]
    assert any(d.executed for d in turn.decisions[0][DECISION_AGARI_IDX])  # seat 0 rons the declaration discard
    seen = turn.responder_states[0]
    init = re.match(r'[^>]*', rounds[r]).group(0)
    ten = [int(x) for x in re.search(r'ten="([^"]*)"', init).group(1).split(",")]
    kyotaku = int(re.search(r'seed="([^"]*)"', init).group(1).split(",")[2])
    assert seen.reaches[1] == 1  # declared: public at once
    assert seen.scores[1] == ten[1] and seen.kyotaku == kyotaku  # no stick: step 2 never came
