"""T-011 speed levers: the fast paths compute exactly what the full paths compute (same weights, same inputs)."""
import contextlib
import glob
import io
import os

import pytest
import torch

from MahjongAI.agents.encoder_conv import (DROPOUT_RATIO, EMBD_SIZE, N_HEADS, N_X1_PLANES, N_X3, EncoderBlock,
                                           EncoderConvModel, MultiHeadAttention, StateNet, conv1d_k3_rank_major)
from MahjongAI.process_xml import process


def _padded_batch(B=6, T=151, seed=0, embd_size=EMBD_SIZE):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, embd_size, generator=g)
    valid = torch.randint(1, T, (B,), generator=g)
    mask = torch.arange(T).unsqueeze(0) < valid.unsqueeze(1)
    mask[:, -1] = True  # the state token is always valid
    return x, mask.unsqueeze(1)


def _no_dropout(block):
    for m in block.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    block.self_attention.attn_dropout_p = 0.0


@pytest.mark.parametrize("embd_size", [EMBD_SIZE, 128])
@pytest.mark.parametrize("mode", ["eval", "train_p0"])
def test_state_row_only_equals_full_last_row(mode, embd_size):
    torch.manual_seed(0)
    block = EncoderBlock(embd_size, N_HEADS)
    if mode == "eval":
        block.eval()
    else:  # train mode takes the math attention kernel with the mask; dropout off so both paths are deterministic
        block.train()
        _no_dropout(block)
    x, mask = _padded_batch(embd_size=embd_size)

    full = block(x, mask)[:, -1:]
    full.sum().backward()
    grads_full = {n: p.grad.clone() for n, p in block.named_parameters()}
    block.zero_grad()

    fast = block(x, mask, state_row_only=True)
    fast.sum().backward()

    assert fast.shape == (x.shape[0], 1, embd_size)
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


@pytest.mark.parametrize("embd_size", [EMBD_SIZE, 128])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("mode", ["eval", "train_p0"])
def test_model_fast_path_equals_full_blocks(games, n_layers, mode, embd_size):
    """The whole model on real games: outputs, input-embedding gradients and every parameter gradient equal the
    reference that runs every block on all rows. With 2 layers only the last block takes the state-row path."""
    torch.manual_seed(0)
    model = EncoderConvModel(n_layers=n_layers, embd_size=embd_size, verbose=False)
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
    assert shapes == ([(151, embd_size)] * (n_layers - 1) + [(1, embd_size)]) * 3, shapes
    _assert_same(fast, reference)


def test_dropout_active_on_state_row(games):
    """Train mode, default dropout: the last block draws its masks on the state row. Two seeds give different
    outputs (the same seed the same), with the module dropouts alone and with the attention dropout alone. Each module
    dropout, counted on its own, runs once per block forward, on the state row only, and zeroes DROPOUT_RATIO +/- 0.02
    of its inputs over many draws."""
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
    modules = {"attention": block.self_attention.dropout, "ffwd": block.ffwd.net[-1]}
    assert isinstance(modules["ffwd"], torch.nn.Dropout)
    stats = {name: {"calls": 0, "kept": 0, "dropped": 0, "shapes": set()} for name in modules}

    def counter(name):
        def count(module, inputs, output):
            s = stats[name]
            s["calls"] += 1
            s["shapes"].add(tuple(output.shape[1:]))
            live = inputs[0] != 0
            dropped = (output == 0) & live
            s["dropped"] += int(dropped.sum())
            s["kept"] += int((live & ~dropped).sum())
        return count

    hooks = [m.register_forward_hook(counter(name)) for name, m in modules.items()]
    forwards = 10
    try:
        for seed in range(forwards):
            logits(seed)
    finally:
        for h in hooks:
            h.remove()
    for name, s in stats.items():  # each module on its own: called, on the state row only, at its rate
        n = s["dropped"] + s["kept"]
        share = s["dropped"] / max(1, n)
        print(f"last-block {name} dropout: {s['calls']} calls, dropped share {share:.4f} over {n} state-row elements")
        assert s["calls"] >= forwards, f"{name} dropout called {s['calls']} times in {forwards} forwards"
        assert s["shapes"] == {(1, EMBD_SIZE)}, f"last-block {name} dropout not on the state row only: {s['shapes']}"
        assert n > 50_000, f"{name} dropout: {n} elements"
        assert abs(share - DROPOUT_RATIO) <= 0.02, f"{name} dropout: dropped share {share:.4f}"
    assert stats["attention"]["calls"] == stats["ffwd"]["calls"], {k: v["calls"] for k, v in stats.items()}


# ---------------------------------------------------------------- StateNet conv as one matmul (T-011 L2)
def reference_state_net_forward(self, state_obj):
    """StateNet.forward of main (b7c8d9c), verbatim apart from comments: the suits go through nn.Conv1d."""
    x1, x2, x3 = state_obj
    batch_size = x1.size(0)
    C, K = N_X1_PLANES, 4
    x1_suits = torch.stack([x1[:, :, 0:9, :], x1[:, :, 9:18, :], x1[:, :, 18:27, :]], dim=1)
    x1_suits = x1_suits.permute(0, 1, 2, 4, 3).contiguous()
    x1_suits = x1_suits.view(batch_size, 3, C * K, 9)
    x1_suits = x1_suits.view(batch_size * 3, C * K, 9)
    suits_feat = self.suit_conv1d_1(x1_suits)
    suits_feat = self.relu(suits_feat)
    suits_feat = self.suit_conv1d_2(suits_feat)
    suits_feat = self.relu(suits_feat)
    suits_feat = torch.mean(suits_feat, dim=2)
    suits_feat = suits_feat.view(batch_size, 3, 16)
    suits_feat = torch.mean(suits_feat, dim=1)
    x1_honors = x1[:, :, 27:34, :].reshape(batch_size, -1)
    honors_feat = self.relu(self.honors_fc2(self.relu(self.honors_fc1(x1_honors))))
    x1_reds = x1[:, :, 34:37, :].reshape(batch_size, -1)
    reds_feat = self.relu(self.reds_fc(x1_reds))
    raw_x1_flat = x1.view(batch_size, -1)
    x1_combined = torch.cat([suits_feat, honors_feat, reds_feat, raw_x1_flat], dim=1)
    x2_flat = x2.view(x2.size(0), -1)
    combined = torch.cat([x1_combined, x2_flat, x3], dim=1)
    x = self.relu_fc_main(self.fc_main_1(combined))
    return self.relu_fc_main(self.fc_main_2(x))


class ReferenceStateNet(StateNet):
    forward = reference_state_net_forward


@contextlib.contextmanager
def _state_net_conv1d():
    """Reference path: StateNet runs main's nn.Conv1d forward."""
    fast = StateNet.forward
    StateNet.forward = reference_state_net_forward
    try:
        yield
    finally:
        StateNet.forward = fast


# state_dict of main's StateNet (b7c8d9c): checkpoints hold these keys with these shapes
MAIN_STATE_NET_KEYS = {
    "suit_conv1d_1.weight": (16, 40, 3), "suit_conv1d_1.bias": (16,),
    "suit_conv1d_2.weight": (16, 16, 3), "suit_conv1d_2.bias": (16,),
    "honors_fc1.weight": (32, 280), "honors_fc1.bias": (32,),
    "honors_fc2.weight": (16, 32), "honors_fc2.bias": (16,),
    "reds_fc.weight": (8, 120), "reds_fc.bias": (8,),
    "fc_main_1.weight": (64, 1555), "fc_main_1.bias": (64,),
    "fc_main_2.weight": (64, 64), "fc_main_2.bias": (64,),
}


def _random_states(n=2250, seed=0):
    """(x1, x2, x3) with the real shapes; x1 thermometer-like 0/1 planes, so the rank convolutions see real
    sparsity. n = the training sample cap of a head."""
    g = torch.Generator().manual_seed(seed)
    x1 = (torch.rand(n, N_X1_PLANES, 37, 4, generator=g) < 0.15).float()
    return x1, torch.rand(n, 7, 4, generator=g), torch.rand(n, N_X3, generator=g)


@pytest.mark.parametrize("cin", [40, 16])
def test_conv1d_k3_rank_major_equals_conv1d(cin):
    """Same math as nn.Conv1d, in float64 (so a summation-order difference cannot hide a wrong index): outputs and
    the gradients of input, weight and bias under a random upstream gradient. In float32 the 60,750-term bias-grad
    sum alone differs by ~1e-3 between the two kernels (the stock one is the less accurate); the float32 model-level
    tests below hold the plan's tolerance."""
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(cin, 16, kernel_size=3, padding=1).double()
    x = torch.randn(6750, cin, 9, dtype=torch.float64, requires_grad=True)  # [B*3, C, R] for B = 2250
    ref = conv(x)
    gy = torch.randn(ref.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ref.backward(gy)
    grads_ref = [t.grad.clone() for t in (x, conv.weight, conv.bias)]
    for t in (x, conv.weight, conv.bias):
        t.grad = None
    out = conv1d_k3_rank_major(x.transpose(1, 2), conv).transpose(1, 2)
    out.backward(gy)
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    for name, t, g in zip(("input", "weight", "bias"), (x, conv.weight, conv.bias), grads_ref):
        torch.testing.assert_close(t.grad, g, atol=1e-9, rtol=1e-10, msg=f"grad {name}")


def test_state_net_state_dict_unchanged():
    sd = StateNet().state_dict()
    assert {k: tuple(v.shape) for k, v in sd.items()} == MAIN_STATE_NET_KEYS
    model_sd = EncoderConvModel(verbose=False).state_dict()
    assert {k[len("state_net."):]: tuple(v.shape) for k, v in model_sd.items()
            if k.startswith("state_net.")} == MAIN_STATE_NET_KEYS


def test_main_state_dict_loads_and_gives_same_outputs():
    """A state_dict saved from main's StateNet loads strictly into the new one: same outputs, and the same gradients
    of every parameter and of the state inputs. Compared in float64: the bias gradients sum ~60,750 terms, and in
    float32 the result depends on the BLAS summation order (stock Conv1d is ~2e-5 off the float64 truth, the matmul
    path ~2e-6; their difference passes on one CPU and fails on another). float64 removes that noise and keeps the
    1e-6 tolerance meaningful: a wrong tap or bias moves a gradient by O(1)."""
    torch.manual_seed(0)
    old = ReferenceStateNet()
    buf = io.BytesIO()
    torch.save(old.state_dict(), buf)
    buf.seek(0)
    new = StateNet()
    new.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    old, new = old.double(), new.double()

    def run(net):
        states = tuple(t.double().requires_grad_(True) for t in _random_states())
        out = net(states)
        net.zero_grad(set_to_none=True)
        out.square().sum().backward()
        return out.detach(), [t.grad for t in states], {n: p.grad for n, p in net.named_parameters()}

    (o_old, gi_old, gp_old), (o_new, gi_new, gp_new) = run(old), run(new)
    torch.testing.assert_close(o_new, o_old, atol=1e-6, rtol=1e-5)
    for i, (a, b) in enumerate(zip(gi_new, gi_old)):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5, msg=f"grad of state input x{i + 1}")
    for n in gp_old:
        torch.testing.assert_close(gp_new[n], gp_old[n], atol=1e-6, rtol=1e-5, msg=f"grad {n}")


@pytest.mark.parametrize("mode", ["eval", "train_p0", "train_dropout"])
def test_model_state_net_matmul_equals_conv1d(games, mode):
    """The whole model on real games, the new StateNet vs main's: function-exact with dropout off, seed-exact with
    dropout on (the conv draws no random numbers, so the same seed gives the same masks)."""
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False)
    if mode == "eval":
        model.eval()
    else:
        model.train()
        if mode == "train_p0":
            _set_dropout(model, 0.0)
    torch.manual_seed(7)
    fast = _run(model, games)
    after_fast = torch.get_rng_state()
    torch.manual_seed(7)
    with _state_net_conv1d():
        reference = _run(model, games)
    assert torch.equal(torch.get_rng_state(), after_fast), "the two paths drew a different number of random values"
    _assert_same(fast, reference)
