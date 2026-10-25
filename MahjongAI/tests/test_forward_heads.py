"""T-055: forward_heads (positions mode) is the network of forward (games mode). The same two fixture games, no cap,
dropout on, one seed: forward on the games and forward_heads on the tensors prepare_batches builds from them give
equal labels, logits and losses, equal gradients and the same final RNG state (dropout draws in the same order).
An absent head is skipped; a mask of the wrong width raises."""
import glob
import os

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.process_xml import process
from MahjongAI.tests.test_legacy_forward_ref import GAMES, GATE_GAMES


@pytest.fixture(scope="module")
def games():
    return [process(glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", g))[0]) for g in GAMES]


def _run(games, use_heads, drop=()):
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False).train()
    halfturns, tokens = [g[0] for g in games], [g[1] for g in games]
    tensors = dict(zip(SL.HEADS, model.tensor_processor.prepare_batches(halfturns, tokens)))
    torch.manual_seed(5)
    if use_heads:
        results = model.forward_heads({h: t for h, t in tensors.items() if h not in drop})
    else:
        results = model(halfturns, tokens, sample=False)
    sum(r.loss for r in results.values()).backward()
    return results, [p.grad for p in model.parameters()], torch.get_rng_state()


def test_forward_heads_equals_forward_bitwise(games):
    a, grads_a, rng_a = _run(games, use_heads=False)
    b, grads_b, rng_b = _run(games, use_heads=True)
    assert list(a) == list(b) == list(SL.HEADS)
    for h in a:
        assert torch.equal(a[h].y, b[h].y) and torch.equal(a[h].logits, b[h].logits), h
        assert torch.equal(a[h].loss, b[h].loss), h
    assert all(torch.equal(x, y) for x, y in zip(grads_a, grads_b))
    assert torch.equal(rng_a, rng_b)


def test_forward_heads_skips_an_absent_head(games):
    results, grads, _ = _run(games, use_heads=True, drop=("during",))
    assert list(results) == ["discard", "post"]
    torch.manual_seed(0)
    names = [n for n, _ in EncoderConvModel(verbose=False).named_parameters()]
    assert {n for n, g in zip(names, grads) if g is None} == {"during_head.weight", "during_head.bias"}


def test_forward_heads_rejects_a_mask_of_another_width(games):
    model = EncoderConvModel(verbose=False)
    t = dict(zip(SL.HEADS, model.tensor_processor.prepare_batches([games[0][0]], [games[0][1]])))["discard"]
    with pytest.raises(AssertionError, match="discard: logits"):
        model.forward_heads({"discard": (t[0], t[1], t[2][:, :36], t[3], t[4])})
