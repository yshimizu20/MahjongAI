"""T-047 M1: the one-query attention as broadcast multiply + sum computes the 468d71c SDPA block's function: same
outputs and every gradient with dropout off (eval, train p=0; float64), same outputs, gradients and final RNG state
with dropout on under the same seed, a fully masked row included, and a 2-layer model whose first block keeps SDPA."""
import contextlib
import copy
import glob
import os

import pytest
import torch
import torch.nn.functional as F

from MahjongAI.agents import encoder_conv as EC
from MahjongAI.agents.encoder_conv import DROPOUT_RATIO, EMBD_SIZE, N_HEADS, EncoderConvModel, MultiHeadAttention
from MahjongAI.process_xml import process

ATOL, RTOL = 1e-6, 1e-5


class ReferenceMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention.forward of 468d71c, verbatim apart from comments: every query count goes through SDPA."""

    def forward(self, k, v, q, attn_mask):
        B, T, _ = k.shape
        Q = q.shape[1]
        H, HS = self.num_heads, self.head_size
        k = self.key_proj(k).view(B, T, H, HS).transpose(1, 2)
        v = self.value_proj(v).view(B, T, H, HS).transpose(1, 2)
        q = self.query_proj(q).view(B, Q, H, HS).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        dropout_p = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Q, H * HS)
        out = self.proj(out)
        out = self.dropout(out)
        return out


def _reference_of(module):
    """A deep copy of `module` whose attention modules run the 468d71c forward (same parameters)."""
    ref = copy.deepcopy(module)
    for m in ref.modules():
        if type(m) is MultiHeadAttention:
            m.__class__ = ReferenceMultiHeadAttention
    assert [n for n, _ in ref.named_parameters()] == [n for n, _ in module.named_parameters()]
    return ref


@contextlib.contextmanager
def _count_fast_path():
    """Counts the calls of the production one-query path (the activation check)."""
    calls = []
    original = MultiHeadAttention._single_query_attention

    def spy(*a, **k):
        calls.append(a[0].shape)
        return original(*a, **k)

    MultiHeadAttention._single_query_attention = staticmethod(spy)
    try:
        yield calls
    finally:
        MultiHeadAttention._single_query_attention = staticmethod(original)


def _set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
        if isinstance(m, MultiHeadAttention):
            m.attn_dropout_p = p


def _assert_grads(a, b):
    """Same parameter names, every gradient present and finite in both, equal within the tolerances."""
    assert a.keys() == b.keys()
    for n in a:
        assert a[n] is not None and b[n] is not None, f"no grad for {n}"
        assert torch.isfinite(a[n]).all() and torch.isfinite(b[n]).all(), f"non-finite grad {n}"
        torch.testing.assert_close(a[n], b[n], atol=ATOL, rtol=RTOL, msg=f"grad {n}")


# ---------------------------------------------------------------- the attention module on constructed inputs
def _inputs(B=7, T=151, seed=0, empty_rows=(), embd_size=EMBD_SIZE):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(B, T, embd_size, generator=g, dtype=torch.float64)
    q = torch.randn(B, 1, embd_size, generator=g, dtype=torch.float64)
    valid = torch.randint(1, T, (B,), generator=g)
    mask = torch.arange(T).unsqueeze(0) < valid.unsqueeze(1)
    mask[:, -1] = True
    for r in empty_rows:
        mask[r] = False  # a row with no key to attend
    return kv, q, mask.unsqueeze(1)


def _attention_run(module, kv, q, mask, seed):
    kv, q = kv.clone().requires_grad_(), q.clone().requires_grad_()
    module.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    out = module(kv, kv, q, mask)
    (out * torch.linspace(-1, 1, out.numel(), dtype=out.dtype).view_as(out)).sum().backward()
    grads = {n: p.grad for n, p in module.named_parameters()}
    grads["input kv"], grads["input q"] = kv.grad, q.grad
    return out.detach(), grads, torch.get_rng_state()


@pytest.mark.parametrize("embd_size", [EMBD_SIZE, 128])
@pytest.mark.parametrize("mode", ["eval", "train_p0", "train_dropout"])
@pytest.mark.parametrize("empty_rows", [(), (2, 5)], ids=["masked", "fully_masked_rows"])
def test_attention_module_equals_sdpa_reference(mode, empty_rows, embd_size):
    torch.manual_seed(0)
    module = MultiHeadAttention(N_HEADS, embd_size // N_HEADS, embd_size).double()
    module.train(mode != "eval")
    if mode == "train_p0":
        _set_dropout(module, 0.0)
    ref = _reference_of(module)
    kv, q, mask = _inputs(empty_rows=empty_rows, embd_size=embd_size)
    with _count_fast_path() as calls:
        out, grads, rng = _attention_run(module, kv, q, mask, seed=3)
    assert len(calls) == 1, "the one-query path did not run"
    ref_out, ref_grads, ref_rng = _attention_run(ref, kv, q, mask, seed=3)
    assert torch.isfinite(out).all() and torch.isfinite(ref_out).all()
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    _assert_grads(grads, ref_grads)
    assert torch.equal(rng, ref_rng), "final RNG state differs"


def test_fully_masked_row_gives_zero_weights():
    """A row with no valid key: the attention output before the projection is zero (the projection's bias alone)."""
    kv, q, mask = _inputs(empty_rows=(1,))
    H, HS = N_HEADS, EMBD_SIZE // N_HEADS
    k = kv.view(7, 151, H, HS).transpose(1, 2)
    qh = q.view(7, 1, H, HS).transpose(1, 2)
    out = MultiHeadAttention._single_query_attention(qh, k, k, mask.unsqueeze(1), 0.0)
    assert torch.isfinite(out).all() and not out[1].any() and out[0].abs().sum() > 0


# ---------------------------------------------------------------- the whole model on real games (float64)
GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
FIXTURE_LOGS = ("2021010100gm-00a9-0000-05d6c581.log", "2021010313gm-00a9-0000-5120e1eb.log")


@pytest.fixture(scope="module")
def games():
    halfturns, tokens = [], []
    for name in FIXTURE_LOGS:
        found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
        assert len(found) == 1, (name, found)
        h, t = process(found[0])
        halfturns.append(h)
        tokens.append(t)
    return halfturns, tokens


def _as_double(processor):
    """Wrap processor.prepare_batches so the float tensors reach the float64 model as float64."""
    prepare = processor.prepare_batches

    def cast(t):
        if isinstance(t, tuple):
            return tuple(cast(x) for x in t)
        return t.double() if t.is_floating_point() else t

    processor.prepare_batches = lambda *a, **k: tuple(cast(h) for h in prepare(*a, **k))


def _model_run(model, games, seed, sample_size):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    results = model(*games, sample=True, sample_size=sample_size)
    sum(r.loss for r in results.values()).backward()
    return ({h: (r.y, r.logits.detach(), r.loss.detach()) for h, r in results.items()},
            {n: p.grad for n, p in model.named_parameters()}, torch.get_rng_state())


@pytest.mark.parametrize("embd_size", [EMBD_SIZE, 128])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("mode", ["eval", "train_p0", "train_dropout"])
def test_model_equals_sdpa_reference(games, n_layers, mode, embd_size):
    """Real games, a forced small sample cap (every head above it): selected targets equal; logits, losses and every
    parameter gradient equal; final RNG state equal. With 2 layers the first block keeps SDPA (Q = T) and only the
    last block takes the one-query path."""
    torch.manual_seed(0)
    model = EncoderConvModel(n_layers=n_layers, embd_size=embd_size, verbose=False).double()
    model.train(mode != "eval")
    if mode == "train_p0":
        _set_dropout(model, 0.0)
    ref = _reference_of(model)
    _as_double(model.tensor_processor)
    _as_double(ref.tensor_processor)
    sdpa_queries = []
    sdpa = F.scaled_dot_product_attention

    def sdpa_spy(q, *a, **k):
        sdpa_queries.append(q.shape[2])
        return sdpa(q, *a, **k)

    EC.F.scaled_dot_product_attention = sdpa_spy
    try:
        with _count_fast_path() as calls:
            out, grads, rng = _model_run(model, games, seed=11, sample_size=40)
    finally:
        EC.F.scaled_dot_product_attention = sdpa
    assert len(calls) == 3, f"one-query path ran {len(calls)} times for 3 heads"
    assert sdpa_queries == [151] * 3 * (n_layers - 1), sdpa_queries
    ref_out, ref_grads, ref_rng = _model_run(ref, games, seed=11, sample_size=40)
    assert out.keys() == ref_out.keys() == {"during", "discard", "post"}
    for h in out:
        assert len(out[h][0]) == 40 and torch.equal(out[h][0], ref_out[h][0]), f"{h} selected targets"
        torch.testing.assert_close(out[h][1], ref_out[h][1], atol=ATOL, rtol=RTOL, msg=f"{h} logits")
        torch.testing.assert_close(out[h][2], ref_out[h][2], atol=ATOL, rtol=RTOL, msg=f"{h} loss")
    _assert_grads(grads, ref_grads)
    assert torch.equal(rng, ref_rng), "final RNG state differs"


def test_dropout_on_changes_with_the_seed(games):
    """The seed-exact comparison is not vacuous: with dropout on, another seed gives other logits."""
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False).double().train()
    _set_dropout(model, 0.0)
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.attn_dropout_p = DROPOUT_RATIO  # the attention-weight dropout alone
    _as_double(model.tensor_processor)
    a, _, _ = _model_run(model, games, seed=11, sample_size=40)
    b, _, _ = _model_run(model, games, seed=12, sample_size=40)
    assert any(not torch.equal(a[h][1], b[h][1]) for h in a)
