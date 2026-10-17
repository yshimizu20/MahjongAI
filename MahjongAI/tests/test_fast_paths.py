"""T-011 speed levers: the fast paths compute exactly what the full paths compute (same weights, same inputs)."""
import contextlib
import glob
import os

import pytest
import torch

from MahjongAI.agents.encoder_conv import (DROPOUT_RATIO, EMBD_SIZE, N_HEADS, EncoderBlock, EncoderConvModel,
                                           MultiHeadAttention)
from MahjongAI.process_xml import process


def _padded_batch(B=6, T=151, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, EMBD_SIZE, generator=g)
    valid = torch.randint(1, T, (B,), generator=g)
    mask = torch.arange(T).unsqueeze(0) < valid.unsqueeze(1)
    mask[:, -1] = True  # the state token is always valid
    return x, mask.unsqueeze(1)


def _no_dropout(block):
    for m in block.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    block.self_attention.attn_dropout_p = 0.0


@pytest.mark.parametrize("mode", ["eval", "train_p0"])
def test_state_row_only_equals_full_last_row(mode):
    torch.manual_seed(0)
    block = EncoderBlock(EMBD_SIZE, N_HEADS)
    if mode == "eval":
        block.eval()
    else:  # train mode takes the math attention kernel with the mask; dropout off so both paths are deterministic
        block.train()
        _no_dropout(block)
    x, mask = _padded_batch()

    full = block(x, mask)[:, -1:]
    full.sum().backward()
    grads_full = {n: p.grad.clone() for n, p in block.named_parameters()}
    block.zero_grad()

    fast = block(x, mask, state_row_only=True)
    fast.sum().backward()

    assert fast.shape == (x.shape[0], 1, EMBD_SIZE)
    torch.testing.assert_close(fast, full, atol=1e-6, rtol=1e-5)
    for n, p in block.named_parameters():
        torch.testing.assert_close(p.grad, grads_full[n], atol=1e-6, rtol=1e-5, msg=f"grad {n}")


# ---------------------------------------------------------------- full model, real inputs (T-011 L1t)
GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
FIXTURE_LOGS = ("2021010100gm-00a9-0000-05d6c581.log", "2021010313gm-00a9-0000-5120e1eb.log")


@pytest.fixture(scope="module")
def games():
    """Two real fixture games, parsed: ([halfturns per game], [tokens per game]), the model's input."""
    halfturns, tokens = [], []
    for name in FIXTURE_LOGS:
        found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
        assert len(found) == 1, (name, found)
        h, t = process(found[0])
        halfturns.append(h)
        tokens.append(t)
    return halfturns, tokens


def _set_dropout(model, p, attn_p=None):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
        if isinstance(m, MultiHeadAttention):
            m.attn_dropout_p = p if attn_p is None else attn_p


@contextlib.contextmanager
def _every_block_full():
    """Reference path: every block runs on all rows; a block asked for the state row only returns the last row of
    its full output."""
    fast = EncoderBlock.forward

    def full(self, x, attn_mask, state_row_only=False):
        out = fast(self, x, attn_mask)
        return out[:, -1:] if state_row_only else out

    EncoderBlock.forward = full
    try:
        yield
    finally:
        EncoderBlock.forward = fast


def _run(model, games):
    """Forward + backward of the summed head losses. Returns the per-head logits and losses, the gradients of the
    history and state embeddings (the encoder's inputs, one per head call) and of every parameter."""
    embeddings = []

    def keep(module, inputs, output):
        output.retain_grad()
        embeddings.append(output)

    hooks = [model.token_embedder.register_forward_hook(keep), model.state_net.register_forward_hook(keep)]
    try:
        results = model(*games, sample=False)
    finally:
        for h in hooks:
            h.remove()
    model.zero_grad(set_to_none=True)
    sum(r.loss for r in results.values()).backward()
    return ({h: (r.logits.detach(), r.loss.detach()) for h, r in results.items()},
            [e.grad for e in embeddings],
            {n: p.grad for n, p in model.named_parameters()})


def _assert_same(a, b):
    out_a, emb_a, par_a = a
    out_b, emb_b, par_b = b
    assert out_a.keys() == out_b.keys() == {"during", "discard", "post"}
    for h in out_a:
        torch.testing.assert_close(out_a[h], out_b[h], atol=1e-6, rtol=1e-5, msg=f"{h} logits / loss")
    assert len(emb_a) == len(emb_b) == 6  # history and state embedding of each of the 3 heads
    for i, (ga, gb) in enumerate(zip(emb_a, emb_b)):
        torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-5, msg=f"embedding grad {i}")
    assert par_a.keys() == par_b.keys()
    for n in par_a:
        assert par_a[n] is not None, f"no grad for {n}"
        torch.testing.assert_close(par_a[n], par_b[n], atol=1e-6, rtol=1e-5, msg=f"grad {n}")


@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("mode", ["eval", "train_p0"])
def test_model_fast_path_equals_full_blocks(games, n_layers, mode):
    """The whole model on real games: outputs, input-embedding gradients and every parameter gradient equal the
    reference that runs every block on all rows. With 2 layers only the last block takes the state-row path."""
    torch.manual_seed(0)
    model = EncoderConvModel(n_layers=n_layers, verbose=False)
    if mode == "eval":
        model.eval()
    else:
        model.train()
        _set_dropout(model, 0.0)

    shapes = []
    hooks = [b.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape[1:]))) for b in model.blocks]
    fast = _run(model, games)
    for h in hooks:
        h.remove()
    with _every_block_full():
        reference = _run(model, games)

    # per head call: blocks 0 .. n-2 return every row (T + 1 = 151), the last one the state row only
    assert shapes == ([(151, EMBD_SIZE)] * (n_layers - 1) + [(1, EMBD_SIZE)]) * 3, shapes
    _assert_same(fast, reference)


def test_dropout_active_on_state_row(games):
    """Train mode, default dropout: the last block draws its masks on the state row. Two seeds give different
    outputs (the same seed the same), with the module dropouts alone and with the attention dropout alone; the
    share the module dropouts zero over many draws is DROPOUT_RATIO +/- 0.02."""
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False).train()

    def logits(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            return torch.cat([r.logits.flatten() for r in model(*games, sample=False).values()])

    for module_p, attn_p in ((DROPOUT_RATIO, 0.0), (0.0, DROPOUT_RATIO)):
        _set_dropout(model, module_p, attn_p)
        a, b = logits(1), logits(2)
        assert not torch.equal(a, b), f"same output under two seeds (module p {module_p}, attention p {attn_p})"
        assert torch.equal(a, logits(1)), "same seed, different output"

    _set_dropout(model, DROPOUT_RATIO)
    block = model.blocks[-1]
    counts = {"kept": 0, "dropped": 0}
    shapes = set()

    def count(module, inputs, output):
        shapes.add(tuple(output.shape[1:]))
        live = inputs[0] != 0
        dropped = (output == 0) & live
        counts["dropped"] += int(dropped.sum())
        counts["kept"] += int((live & ~dropped).sum())

    hooks = [block.self_attention.dropout.register_forward_hook(count),
             block.ffwd.net[-1].register_forward_hook(count)]
    assert isinstance(block.ffwd.net[-1], torch.nn.Dropout)
    try:
        for seed in range(10):
            logits(seed)
    finally:
        for h in hooks:
            h.remove()
    assert shapes == {(1, EMBD_SIZE)}, f"last-block dropout not on the state row only: {shapes}"
    share = counts["dropped"] / (counts["dropped"] + counts["kept"])
    print(f"last-block dropped share {share:.4f} over {counts['dropped'] + counts['kept']} state-row elements")
    assert counts["dropped"] + counts["kept"] > 100_000
    assert abs(share - DROPOUT_RATIO) <= 0.02, f"dropped share {share:.4f}"
