"""Unit tests for MahjongAI.utils.baselines (the no-state references of plan T-005) and the run_report self-test."""
import math

import torch

from MahjongAI.utils import run_report
from MahjongAI.utils.baselines import fit_bias_prior, score_bias, shanten_after, shanten_greedy, uniform_legal


def test_uniform_legal_is_log_k():
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    y = torch.tensor([0, 1, 4])
    nll, p = uniform_legal(mask, y)
    assert torch.allclose(nll, torch.tensor([0.0, math.log(2), math.log(5)], dtype=torch.float64))
    assert torch.allclose(p, torch.tensor([1.0, 0.5, 0.2], dtype=torch.float64))


def test_bias_fit_all_legal_recovers_log_odds():
    y = torch.tensor([0] * 60 + [1] * 30 + [2] * 10)
    mask = torch.ones(len(y), 3, dtype=torch.bool)
    b = fit_bias_prior(mask, y, l2=0.0)
    assert abs((b[1] - b[0]).item() - math.log(30 / 60)) < 1e-5
    assert abs((b[2] - b[0]).item() - math.log(10 / 60)) < 1e-5
    nll, hit = score_bias(b, mask, y)
    expected = -(60 * math.log(0.6) + 30 * math.log(0.3) + 10 * math.log(0.1)) / 100
    assert abs(nll.mean().item() - expected) < 1e-6
    assert hit.double().mean().item() == 0.6  # always predicts the majority action


def test_bias_fit_ron_legal_means_ron():
    """Pass (0) vs ron (1) offers are always a ron; pass vs chi (2) offers are always a pass. The bias orders
    ron > pass > chi, so top-1 is 1.0; chi is never taken and its bias stays finite (the L2)."""
    mask = torch.zeros(100, 3, dtype=torch.bool)
    mask[:, 0] = True
    mask[:40, 1] = True
    mask[40:, 2] = True
    y = torch.tensor([1] * 40 + [0] * 60)
    b = fit_bias_prior(mask, y)
    assert torch.isfinite(b).all() and b[1] > b[0] > b[2]
    nll, hit = score_bias(b, mask, y)
    assert hit.all() and nll.max().item() < 0.05


def _hand(slots) -> torch.Tensor:
    h = torch.zeros(37)
    for s in slots:
        h[s] += 1
    return h


def test_shanten_greedy_unique_best_with_a_red_five():
    # 123m 4m red-5m 6m 789m 11p 34p + East: only the East discard leaves tenpai (1p pair, 34p wait)
    hand = _hand([0, 1, 2, 3, 4, 34, 5, 6, 7, 8, 9, 9, 11, 12, 27])  # the red 5m is in its kind (4) and red slot (34)
    assert hand[:34].sum() == 14
    mask = hand > 0
    mask[4] = False  # no plain 5m held: only the red slot is a legal 5m discard
    assert shanten_after(hand.long().tolist(), 27) == 0 and shanten_after(hand.long().tolist(), 34) == 1
    hands = hand.repeat(3, 1)
    masks = mask.repeat(3, 1)
    y = torch.tensor([27, 9, 34])
    hits = shanten_greedy(hands, masks, y, torch.zeros(37))
    assert hits.tolist() == [True, False, False]


def test_shanten_greedy_ties_broken_by_bias():
    # 123m 456m 789m 11p + East South West: each honor discard leaves 1-shanten; the bias picks among them
    hand = _hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 27, 28, 29])
    mask = hand > 0
    y = torch.tensor([28])
    south_first = torch.zeros(37)
    south_first[28] = 1.0
    west_first = torch.zeros(37)
    west_first[29] = 1.0
    assert shanten_greedy(hand[None], mask[None], y, south_first).tolist() == [True]
    assert shanten_greedy(hand[None], mask[None], y, west_first).tolist() == [False]
    # a bias that prefers a non-minimal tile (1m) does not override the shanten rule
    one_m_first = torch.zeros(37)
    one_m_first[0] = 5.0
    one_m_first[28] = 1.0
    assert shanten_greedy(hand[None], mask[None], y, one_m_first).tolist() == [True]


def test_run_report_self_test():
    run_report.self_test()
