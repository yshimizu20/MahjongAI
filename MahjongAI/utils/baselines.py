"""Stateless reference policies for the three heads, scored on the same masked samples as the model (plan T-005).

Inputs are the tensors TransformerTensorProcessor builds: `mask` (N, A) of legal actions (bool or 0/1), `y` (N,)
the human action. Every function returns per-sample values; the caller sums them per game.

  uniform_legal   a uniform pick among the legal actions: the init model, NOT a bar a model must beat.
  fit_bias_prior  the no-state prior: one learned logit per action and no input, under the same masked softmax CE
                  as the model. It equals encoder_conv with a constant state vector (heads[h](ln_f(const)) is a
                  fixed logit vector), so a model that does not beat it has not shown that it reads the state.
  score_bias      the prior's per-sample NLL and top-1 hit.
  shanten_greedy  discard head only: discard a legal tile that leaves the lowest shanten, ties broken by a bias
                  prior. A hand rule to compare imitation against, not a measure of play quality.
"""
from typing import Tuple

import torch
from mahjong.shanten import Shanten

from MahjongAI.utils.constants import RED_FIVE_SLOTS

_shanten = Shanten()
SLOT_KIND = list(range(34)) + [plain for plain, _ in RED_FIVE_SLOTS]  # 37-slot action -> tile kind 0-33


def _legal(mask: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    assert mask.dim() == 2 and y.shape == (mask.shape[0],), f"mask {tuple(mask.shape)} vs y {tuple(y.shape)}"
    assert mask.any(dim=1).all(), "a sample without a legal action"
    assert mask[torch.arange(len(y)), y].all(), "a label outside the legal mask"
    return mask


def uniform_legal(mask: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per sample: NLL = log #legal, P(correct) = 1 / #legal (the expected top-1 of a uniform legal pick)."""
    k = _legal(mask, y).sum(dim=1).double()
    return torch.log(k), 1.0 / k


def _masked_log_softmax(bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(bias.double().expand(mask.shape).masked_fill(~mask, float("-inf")), dim=1)


def fit_bias_prior(mask: torch.Tensor, y: torch.Tensor, l2: float = 1e-4) -> torch.Tensor:
    """Max-likelihood bias b (A,) of the masked softmax, + l2 * ||b||^2.

    Loss = mean_i [logsumexp_{a legal in i} b_a - b_{y_i}] + l2 * ||b||^2, by LBFGS from b = 0 in float64.
    Identical (mask, y) rows are merged with their counts first: the same loss, a smaller problem, and no
    dependence on the sample order. The L2 keeps a slot never taken in the fit sample finite. Deterministic.
    """
    mask = _legal(mask, y)
    rows, counts = torch.unique(torch.cat([mask.long(), y.long().unsqueeze(1)], dim=1), dim=0, return_counts=True)
    m, t = rows[:, :-1].bool(), rows[:, -1]
    w = counts.double() / counts.sum()
    b = torch.zeros(mask.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = -_masked_log_softmax(b, m)[torch.arange(len(t)), t]
        loss = (w * nll).sum() + l2 * (b * b).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return b.detach()


def score_bias(bias: torch.Tensor, mask: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per sample: NLL of y under the masked softmax of `bias`, and whether its argmax (first max) is y."""
    mask = _legal(mask, y)
    logp = _masked_log_softmax(bias, mask)
    return -logp[torch.arange(len(y)), y], logp.argmax(dim=1) == y


def shanten_after(hand37, slot: int) -> int:
    """Shanten of the concealed hand after discarding action slot `slot` (37-slot, red five = its kind).
    Chiitoi / kokushi count only for a closed 13-tile hand; otherwise the regular form only."""
    counts = [int(c) for c in hand37[:34]]  # a red five is also counted in its kind's slot
    counts[SLOT_KIND[slot]] -= 1
    assert min(counts) >= 0, f"discard of slot {slot} not held"
    if sum(counts) == 13:
        return _shanten.calculate_shanten(counts)
    return _shanten.calculate_shanten_for_regular_hand(counts)


def shanten_greedy(hand37: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per discard sample: does "discard a legal tile leaving the lowest shanten, ties by the highest bias (then
    the lowest slot)" pick the human discard? hand37 (N, 37): concealed counts after the draw, red fives in both
    their kind and their red slot (x1 plane 0 of the discard tensor)."""
    mask = _legal(mask, y)
    assert hand37.shape == mask.shape, f"hand {tuple(hand37.shape)} vs mask {tuple(mask.shape)}"
    bias = bias.tolist()
    hits = torch.zeros(len(y), dtype=torch.bool)
    for i in range(len(y)):
        hand = hand37[i].round().long().tolist()
        assert sum(hand[:34]) % 3 == 2, f"sample {i}: {sum(hand[:34])} concealed tiles at a discard"
        legal = mask[i].nonzero().flatten().tolist()
        sh = {slot: shanten_after(hand, slot) for slot in legal}
        best = min(sh.values())
        pick = max((slot for slot in legal if sh[slot] == best), key=lambda slot: (bias[slot], -slot))
        hits[i] = pick == int(y[i])
    return hits
