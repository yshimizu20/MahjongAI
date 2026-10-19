"""T-050: the compiled per-head network of training updates (supervised_learning.compile_head_core).

Bounded (one module-scoped compiled callable on two real fixture games, plus one small trainer run): it runs on the
Linux CI as well, mandatory, never skipped. Every check compares the arms on the same weights, optimizer state and
RNG state ("lockstep"): the frozen c7580b1 path (`reference_forward` below, verbatim), the eager path of this branch
(`model(...)`, the evaluation and --no-compile path), the compiled path's split uncompiled
(`model(..., head_core=model.head_core)`) and the compiled path (`model(..., head_core=compiled)`).

- eager and split vs reference: bitwise (losses, logits, gradients, AdamW moments and updates, sampled indices, RNG state).
- compiled vs reference: the [P8] rule per tensor, elementwise |a - b| <= atol + RTOL * |b| (b = reference), with the
  atol scaled to the tensor (ATOL_SCALE * max |b| + ATOL_FLOOR: a fixed atol of 1e-5 passed every element of the tiny
  AdamW second moments and small gradients, whatever their value; stage-5 review); sampled indices and RNG state equal
  (fatal, [R1]: inductor's fallback_random draws dropout from the eager RNG). The comparator itself is tested: a
  1e-3 relative error in any compared tensor fails it, and so does the valid-logit defect (the negative control).
- every parameter gets a gradient > COVERAGE_MIN in some fixture, except STRUCTURAL_ZERO [R2].
- inductor code runs in every compiled update (2 compiled graph calls per head: forward and backward), no eager
  fallback, no graph break [P5].
- evaluation stays eager [P11]; checkpoints keep the c7580b1 keys and resume in a fresh eager process [P6]; invalid
  inputs raise the c7580b1 exception on the compiled path [P1].
"""
import copy
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import (MAX_ACTION_LEN, EncoderConvModel, HeadOutput, MultiHeadAttention,
                                           TransformerTensorProcessor, build_game, device)
from MahjongAI.process_xml import process

# [P8], atol per tensor = ATOL_SCALE * max |reference| + ATOL_FLOOR. The floor only serves an all-zero reference: the
# AdamW second moments of the embeddings peak near 1e-10, so a floor of 1e-12 would pass any of their values.
ATOL_SCALE, ATOL_FLOOR, RTOL = 1e-4, 1e-30, 1e-4
# The AdamW update divides by sqrt(v): where an element's gradient is near zero, a round-off difference of the
# gradient becomes a relative difference of the update (measured on these fixtures: up to 6.1e-4 of the tensor's
# largest update; the other tensors need at most 3.1e-6). The moments it is computed from are compared at ATOL_SCALE.
ATOL_SCALE_UPDATE = 5e-3
COVERAGE_MIN = 1e-8  # a covered parameter's largest gradient: above the structural-zero round-off (about 1e-10)
# [R2] parameters whose loss gradient is zero by the math, so only float round-off reaches them:
STRUCTURAL_ZERO = {
    # adds the same constant to the scores of every key of a query (q . (k + b) = q . k + q . b), and softmax over the
    # keys cancels a per-row constant: d loss / d bias = 0 exactly (the probe measured ~1e-10 round-off in both arms)
    "blocks.0.self_attention.key_proj.bias",
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE_GAMES = REPO_ROOT / "MahjongAI" / "tests" / "data" / "gate_games"
FIXTURE_LOGS = ("2021010100gm-00a9-0000-05d6c581.log", "2021010313gm-00a9-0000-5120e1eb.log")

# c7580b1 parameter and buffer inventory (name, shape), in named_parameters / named_buffers order
C7580B1_PARAMS = (
    ("token_embedder.type_embedding.weight", (4, 64)),
    ("token_embedder.tile_embedding.weight", (37, 64)),
    ("token_embedder.tsumogiri_embedding.weight", (1, 64)),
    ("token_embedder.during_naki_embedding.weight", (72, 64)),
    ("token_embedder.post_naki_embedding.weight", (154, 64)),
    ("blocks.0.self_attention.key_proj.weight", (64, 64)),
    ("blocks.0.self_attention.key_proj.bias", (64,)),
    ("blocks.0.self_attention.query_proj.weight", (64, 64)),
    ("blocks.0.self_attention.query_proj.bias", (64,)),
    ("blocks.0.self_attention.value_proj.weight", (64, 64)),
    ("blocks.0.self_attention.value_proj.bias", (64,)),
    ("blocks.0.self_attention.proj.weight", (64, 64)),
    ("blocks.0.self_attention.proj.bias", (64,)),
    ("blocks.0.ffwd.net.0.weight", (256, 64)),
    ("blocks.0.ffwd.net.0.bias", (256,)),
    ("blocks.0.ffwd.net.2.weight", (64, 256)),
    ("blocks.0.ffwd.net.2.bias", (64,)),
    ("blocks.0.ln1.weight", (64,)),
    ("blocks.0.ln1.bias", (64,)),
    ("blocks.0.ln2.weight", (64,)),
    ("blocks.0.ln2.bias", (64,)),
    ("state_net.suit_conv1d_1.weight", (16, 40, 3)),
    ("state_net.suit_conv1d_1.bias", (16,)),
    ("state_net.suit_conv1d_2.weight", (16, 16, 3)),
    ("state_net.suit_conv1d_2.bias", (16,)),
    ("state_net.honors_fc1.weight", (32, 280)),
    ("state_net.honors_fc1.bias", (32,)),
    ("state_net.honors_fc2.weight", (16, 32)),
    ("state_net.honors_fc2.bias", (16,)),
    ("state_net.reds_fc.weight", (8, 120)),
    ("state_net.reds_fc.bias", (8,)),
    ("state_net.fc_main_1.weight", (64, 1555)),
    ("state_net.fc_main_1.bias", (64,)),
    ("state_net.fc_main_2.weight", (64, 64)),
    ("state_net.fc_main_2.bias", (64,)),
    ("ln_f.weight", (64,)),
    ("ln_f.bias", (64,)),
    ("discard_head.weight", (37, 64)),
    ("discard_head.bias", (37,)),
    ("during_head.weight", (72, 64)),
    ("during_head.bias", (72,)),
    ("post_head.weight", (154, 64)),
    ("post_head.bias", (154,)),
)
C7580B1_BUFFERS = (
    ("token_embedder.empty_embedding", (1, 1, 64)),
    ("token_embedder.position_coding_table", (150, 64)),
    ("token_embedder.player_coding_table", (4, 64)),
)
C7580B1_STATE_KEYS = [n for n, _ in C7580B1_BUFFERS] + [n for n, _ in C7580B1_PARAMS]  # state_dict order: the
# token embedder registers its buffers before its embeddings


# ----------------------------------------------------------------------------------------------- frozen c7580b1 path
def reference_token_embedder_forward(self, encoding_tokens_batch):
    """TokenEmbedder.forward of c7580b1, verbatim apart from comments."""
    T = encoding_tokens_batch.size(1)
    assert T <= MAX_ACTION_LEN
    positions = torch.arange(T, device=encoding_tokens_batch.device)
    codes = encoding_tokens_batch * MAX_ACTION_LEN + positions
    unique_codes, inverse = torch.unique(codes, return_inverse=True)
    tokens = torch.div(unique_codes, MAX_ACTION_LEN, rounding_mode="floor")
    rows = self._embed_codes(tokens, self.position_coding_table[unique_codes - tokens * MAX_ACTION_LEN])
    return F.embedding(inverse, rows)


def reference_forward(self, all_halfturns_list, all_encoding_tokens_list, sample=False, sample_size=2250):
    """EncoderConvModel.forward of c7580b1, verbatim apart from comments and the token embedder call (the frozen
    c7580b1 TokenEmbedder.forward on the same module)."""
    (
        tensors_during,
        tensors_discard,
        tensors_post,
    ) = self.tensor_processor.prepare_batches(all_halfturns_list, all_encoding_tokens_list)

    results = {}
    for tensors, head in zip(
        [tensors_during, tensors_discard, tensors_post],
        ["during", "discard", "post"],
    ):
        if len(tensors[0]) == 0:
            continue

        if sample and len(tensors[0]) > sample_size:
            idx = torch.randperm(len(tensors[0]))[:sample_size]

            tensors = (
                tensors[0][idx],
                (
                    tensors[1][0][idx],
                    tensors[1][1][idx],
                    tensors[1][2][idx],
                ),
                tensors[2][idx],
                tensors[3][idx],
                tensors[4][idx],
            )

        (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask_batch,
        ) = tensors
        encoding_tokens_batch = encoding_tokens_batch.to(device)
        state_obj_tensor_batch = tuple(tensor.to(device) for tensor in state_obj_tensor_batch)
        action_mask_batch = action_mask_batch.to(device)
        y_tensor = y_tensor.to(device)
        padding_mask_batch = padding_mask_batch.to(device)

        assert torch.all((action_mask_batch == 0) | (action_mask_batch == 1)), "action_mask_batch must be binary"
        assert torch.any(action_mask_batch, dim=1).all(), "Each sample must have at least one allowed action"

        history_emb = reference_token_embedder_forward(self.token_embedder, encoding_tokens_batch)

        state_emb = self.state_net(state_obj_tensor_batch)
        state_emb = state_emb.unsqueeze(1)

        fused = torch.cat([history_emb, state_emb], dim=1)

        B = padding_mask_batch.size(0)
        state_mask = torch.ones(B, 1, dtype=torch.bool, device=padding_mask_batch.device)
        fused_mask = torch.cat([padding_mask_batch, state_mask], dim=1)

        attn_mask = fused_mask.unsqueeze(1)
        x = fused
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask, state_row_only=(i == len(self.blocks) - 1))

        final_state = x[:, -1, :]
        final_state = self.ln_f(final_state)
        logits = self.heads[head](final_state)

        assert (
            logits.shape == action_mask_batch.shape
        ), f"Logits shape {logits.shape} and mask shape {action_mask_batch.shape} do not match."

        assert torch.all(
            action_mask_batch[torch.arange(y_tensor.size(0)), y_tensor] == 1
        ), "Target action is masked out as invalid!"

        logits = logits.masked_fill(action_mask_batch == 0, -1e9)

        loss = F.cross_entropy(logits, y_tensor, reduction="mean")
        results[head] = HeadOutput(logits, loss, y_tensor)

    return results


# ----------------------------------------------------------------------------------------------- spies
class InductorCalls:
    """Counts the calls of compiled inductor graphs (forward and backward) while active."""

    def __init__(self):
        from torch._inductor.codecache import CompiledFxGraph
        self.cls, self.n = CompiledFxGraph, 0
        self.orig = CompiledFxGraph.__call__

    def __enter__(self):
        spy = self

        def call(graph, inputs):
            spy.n += 1
            return spy.orig(graph, inputs)

        self.cls.__call__ = call
        return self

    def __exit__(self, *exc):
        self.cls.__call__ = self.orig


class RandpermLog:
    """The sampled-index draws (torch.randperm outside compilation) while active."""

    def __init__(self):
        self.orig, self.draws = torch.randperm, []

    def __enter__(self):
        def randperm(*a, **k):
            r = self.orig(*a, **k)
            if not torch._subclasses.fake_tensor.is_fake(r):
                self.draws.append(r.clone())
            return r

        torch.randperm = randperm
        return self

    def __exit__(self, *exc):
        torch.randperm = self.orig


# ----------------------------------------------------------------------------------------------- comparison
def tensor_atol(b, scale=ATOL_SCALE) -> float:
    """The [P8] atol of a reference tensor: `scale` of its largest magnitude, plus ATOL_FLOOR."""
    return scale * b.double().abs().max().item() + ATOL_FLOOR if b.numel() else ATOL_FLOOR


def atol_scale(key: str) -> float:
    return ATOL_SCALE_UPDATE if key == "delta" else ATOL_SCALE


def check_close(a, b, where, scale=ATOL_SCALE):
    """[P8]: |a - b| <= atol + RTOL * |b| elementwise, atol = tensor_atol(b, scale); on failure the max abs, max rel
    and the worst index."""
    assert a.shape == b.shape and a.dtype == b.dtype, f"{where}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
    a64, b64 = a.double(), b.double()
    diff = (a64 - b64).abs()
    atol = tensor_atol(b, scale)
    bad = diff > atol + RTOL * b64.abs()
    if bad.any():
        rel = diff / b64.abs().clamp_min(1e-30)
        worst = tuple(int(i) for i in torch.nonzero(bad)[diff[bad].argmax()]) if diff.dim() else ()
        raise AssertionError(f"{where}: {int(bad.sum())} of {diff.numel()} elements outside |a-b| <= {atol:.3e} + "
                             f"{RTOL}|b|;"
                             f" max abs {diff.max().item():.3e}, max rel {rel.max().item():.3e}, worst index {worst} "
                             f"(a {a64[worst].item():.6e}, b {b64[worst].item():.6e})")


def check_equal(a, b, where):
    assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), f"{where}: not bitwise equal"


MASKED_LOGIT = -1e9  # head_network's masked_fill value


def check_logits(a, b, where):
    """Masked logits: the same positions, bitwise MASKED_LOGIT; the valid logits by check_close, so their atol scales
    with the valid logits (over the whole tensor, the -1e9 fill would make it 1e5)."""
    assert a.shape == b.shape and a.dtype == b.dtype, f"{where}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
    masked = b == MASKED_LOGIT
    check_equal(a == MASKED_LOGIT, masked, f"{where}: masked positions")
    check_close(a[~masked], b[~masked], f"{where} (valid)")


def check_structural_zero(rec, n, where):
    """[R2] a STRUCTURAL_ZERO parameter: its gradient is float round-off in every arm (about 1e-10), so a relative
    comparison is meaningless. Bounds instead, in each arm: |grad| and |clipped grad| <= COVERAGE_MIN, |exp_avg| <=
    COVERAGE_MIN, exp_avg_sq <= COVERAGE_MIN ** 2; the AdamW update of a round-off gradient is round-off noise of up to
    a few lr per step, so |update| <= 5 lr."""
    bounds = {"grad": COVERAGE_MIN, "clipped": COVERAGE_MIN, "exp_avg": COVERAGE_MIN, "exp_avg_sq": COVERAGE_MIN ** 2,
              "delta": 5 * SL.learning_rate}
    for key, bound in bounds.items():
        m = rec[key][n].abs().max().item()
        assert m <= bound, f"{where}: structural zero {n}: max |{key}| {m:.3e} > {bound:.1e}"


def compare(got, ref, arm, exact):
    """Every recorded tensor of one lockstep update: losses, logits, gradients (raw and clipped), AdamW moments and
    parameter updates; the sampled indices and the RNG state after the update always bitwise. Not exact: the
    STRUCTURAL_ZERO parameters by their bounds (check_structural_zero), in both records."""
    def check(a, b, where, key=None):
        if exact:
            check_equal(a, b, where)
        else:
            check_close(a, b, where, atol_scale(key))

    assert got["heads"] == ref["heads"], f"{arm}: heads {got['heads']} vs {ref['heads']}"
    assert got["no_grad"] == ref["no_grad"], f"{arm}: parameters without a gradient differ"
    assert len(got["draws"]) == len(ref["draws"]), f"{arm}: {len(got['draws'])} vs {len(ref['draws'])} randperm draws"
    for i, (a, b) in enumerate(zip(got["draws"], ref["draws"])):
        check_equal(a, b, f"{arm}: sampled indices, draw {i}")
    check_equal(got["rng"], ref["rng"], f"{arm}: RNG state after the update")
    for h in ref["heads"]:
        check(got["loss"][h], ref["loss"][h], f"{arm}: {h} loss")
        if exact:
            check_equal(got["logits"][h], ref["logits"][h], f"{arm}: {h} logits")
        else:
            check_logits(got["logits"][h], ref["logits"][h], f"{arm}: {h} logits")
        check_equal(got["y"][h], ref["y"][h], f"{arm}: {h} targets")
    compare_parameters(got, ref, arm, check, exact)


def compare_parameters(got, ref, arm, check, exact):
    """compare(): the per-parameter tensors (gradients raw and clipped, AdamW moments and updates)."""
    for key in ("grad", "clipped", "exp_avg", "exp_avg_sq", "delta"):
        assert got[key].keys() == ref[key].keys(), f"{arm}: {key} names"
        for n in ref[key]:
            if not exact and n in STRUCTURAL_ZERO:
                continue
            check(got[key][n], ref[key][n], f"{arm}: {key} {n}", key)
    if not exact:
        for n in STRUCTURAL_ZERO:
            check_structural_zero(got, n, arm)
            check_structural_zero(ref, n, f"{arm} reference")


def lockstep_update(model, forward, state, opt_state, rng, batch, sample, sample_size):
    """One training update from (weights, AdamW state, RNG state), recorded. forward: one of the arms."""
    model.load_state_dict(state)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=SL.learning_rate)
    if opt_state is not None:
        opt.load_state_dict(copy.deepcopy(opt_state))  # load_state_dict keeps the tensors: no aliasing
    torch.set_rng_state(rng)
    with RandpermLog() as draws:
        results = forward(*batch, sample=sample, sample_size=sample_size)
    loss = sum(out.loss for out in results.values())
    opt.zero_grad()
    loss.backward()
    names = [n for n, _ in model.named_parameters()]

    def grads():  # an unused head's parameters have no gradient (None): recorded as zeros, and in "no_grad"
        return {n: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}

    grad = grads()
    no_grad = sorted(n for n, p in model.named_parameters() if p.grad is None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    clipped = grads()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    params = dict(model.named_parameters())
    return {
        "heads": list(results), "draws": draws.draws, "rng": torch.get_rng_state(),
        "loss": {h: o.loss.detach().clone() for h, o in results.items()},
        "logits": {h: o.logits.detach().clone() for h, o in results.items()},
        "y": {h: o.y.clone() for h, o in results.items()},
        "grad": grad, "clipped": clipped, "no_grad": no_grad,
        "delta": {n: params[n].detach() - before[n] for n in names},
        "exp_avg": {n: opt.state[params[n]]["exp_avg"].clone() for n in names},
        "exp_avg_sq": {n: opt.state[params[n]]["exp_avg_sq"].clone() for n in names},
        "state": {k: v.clone() for k, v in model.state_dict().items()}, "opt_state": copy.deepcopy(opt.state_dict()),
    }


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def games():
    halfturns, tokens = [], []
    for name in FIXTURE_LOGS:
        found = list(GATE_GAMES.glob(f"*/*/*/{name}"))
        assert len(found) == 1, (name, found)
        h, t = process(str(found[0]))
        halfturns.append(h)
        tokens.append(t)
    return halfturns, tokens


@pytest.fixture(scope="module")
def setup():
    """One model (seed 0, train mode, dropout on) and its compiled callable, shared by the module: one compile."""
    threads = torch.get_num_threads()
    torch.set_num_threads(SL.TORCH_THREADS)
    torch.manual_seed(SL.MODEL_SEED)
    model = EncoderConvModel(verbose=False).to(device).train()
    compiled = SL.compile_head_core(model)
    yield model, compiled
    torch.set_num_threads(threads)


def arms(model, compiled):
    return {
        "reference": lambda *a, **k: reference_forward(model, *a, **k),
        "eager": lambda *a, **k: model(*a, **k),
        "split": lambda *a, **k: model(*a, head_core=model.head_core, **k),  # the compiled path's split, uncompiled
        "compiled": lambda *a, **k: model(*a, head_core=compiled, **k),
    }


def one_head_batch(model, games, head, n, edit=None):
    """A patched prepare_batches for `model`: only `head`, its first n samples (edit(tensors) may change them); the
    other heads empty. Returns the restore function."""
    real = TransformerTensorProcessor().prepare_batches(*games)
    k = ("during", "discard", "post").index(head)

    def cut(t, m):
        return (t[0][:m], tuple(x[:m] for x in t[1]), t[2][:m].clone(), t[3][:m].clone(), t[4][:m])

    out = tuple(cut(t, n if i == k else 0) for i, t in enumerate(real))
    if edit is not None:
        edit(out[k])
    model.tensor_processor.prepare_batches = lambda *a, **kw: out

    def restore():
        del model.tensor_processor.prepare_batches

    return restore


# ----------------------------------------------------------------------------------------------- tests
def test_inventory_and_checkpoint_keys_are_c7580b1():
    model = EncoderConvModel(verbose=False)
    assert tuple((n, tuple(p.shape)) for n, p in model.named_parameters()) == C7580B1_PARAMS
    assert tuple((n, tuple(b.shape)) for n, b in model.named_buffers()) == C7580B1_BUFFERS
    assert list(model.state_dict()) == C7580B1_STATE_KEYS


def run_arms(model, forwards, start, inputs, graph_calls):
    """{arm: lockstep_update record} from the same start (state, opt_state, rng); appends (inductor graph calls,
    heads) of the compiled arm to graph_calls; the other arms must run no compiled graph."""
    rec = {}
    for arm, fwd in forwards.items():
        with InductorCalls() as calls:
            rec[arm] = lockstep_update(model, fwd, *start, *inputs)
        if arm == "compiled":
            graph_calls.append((calls.n, len(rec[arm]["heads"])))
        else:
            assert calls.n == 0, f"{arm} ran a compiled graph"
    return rec


def test_compiled_and_eager_match_c7580b1_in_lockstep(setup, games):
    """Updates from non-initial weights (two reference updates first), each from the reference's state: all heads
    sampled (sample_size 40: one randperm per head), unsampled, the production cap 2250 crossed by the discard head
    only (the two games three times: 3456 discard samples, 990 post, 132 during), and a batch of one discard sample
    and no other head (B = 1, empty heads; its own compiled graph: dynamo specializes a size-1 dimension)."""
    model, compiled = setup
    forwards = arms(model, compiled)
    torch.manual_seed(7)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state, rng = None, torch.get_rng_state()
    big = (games[0] * 3, games[1] * 3)
    plan = [(games, True, 40, None), (games, True, 40, None),  # warm-up: reference only
            (games, True, 40, 3), (games, False, 2250, 0), (big, True, 2250, 1), (games, True, 2250, "B=1")]
    coverage = {n: 0.0 for n, _ in model.named_parameters()}
    graph_calls = []
    from torch._dynamo.utils import counters
    frames0 = dict(counters["frames"])
    for k, (batch, sample, size, draws) in enumerate(plan):
        restore = one_head_batch(model, games, "discard", 1) if draws == "B=1" else None
        try:
            rec = run_arms(model, forwards if draws is not None else {"reference": forwards["reference"]},
                           (state, opt_state, rng), (batch, sample, size), graph_calls)
        finally:
            if restore is not None:
                restore()
        ref = rec["reference"]
        if draws is not None:
            if draws == "B=1":
                assert ref["heads"] == ["discard"] and ref["y"]["discard"].shape == (1,)
            else:
                assert len(ref["draws"]) == draws and len(ref["heads"]) == 3, (len(ref["draws"]), ref["heads"])
            compare(rec["eager"], ref, f"update {k + 1} eager", exact=True)
            compare(rec["split"], ref, f"update {k + 1} split (uncompiled head_core)", exact=True)
            compare(rec["compiled"], ref, f"update {k + 1} compiled", exact=False)
            for n, g in ref["grad"].items():
                coverage[n] = max(coverage[n], g.abs().max().item())
            for arm in ("reference", "compiled"):
                for n in STRUCTURAL_ZERO:
                    g = rec[arm]["grad"][n].abs().max().item()
                    assert g <= COVERAGE_MIN, f"update {k + 1} {arm}: structural zero {n} has |grad| {g:.3e}"
        state, opt_state, rng = ref["state"], ref["opt_state"], ref["rng"]
    assert all(c == 2 * h for c, h in graph_calls), f"(inductor graph calls, heads) per compiled update {graph_calls}"
    uncovered = sorted(n for n, c in coverage.items() if c <= COVERAGE_MIN and n not in STRUCTURAL_ZERO)
    assert not uncovered, f"parameters without a gradient > {COVERAGE_MIN} in any fixture: {uncovered}"
    frames = {k: v - frames0.get(k, 0) for k, v in counters["frames"].items()}
    assert frames.get("total", 0) == frames.get("ok", 0), f"an eager fallback: frames {frames}"
    assert not counters["graph_break"], f"graph breaks: {dict(counters['graph_break'])}"


class ReferenceCaller:
    """The frozen c7580b1 forward behind the interface ValSet.evaluate uses (eval / train / call)."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, *a, **k):
        return reference_forward(self.model, *a, **k)


def test_evaluation_stays_eager_after_compiled_updates(setup, games):
    """[P11] train (compiled) -> eval -> train (compiled): the evaluation runs no compiled graph, and at the same
    weights its per-game output and the RNG state after it equal the c7580b1 evaluation bitwise; the update after it
    still matches the reference."""
    model, compiled = setup
    val = SL.ValSet(list(FIXTURE_LOGS), [build_game(h, t) for h, t in zip(*games)], TransformerTensorProcessor())
    torch.manual_seed(11)
    state, rng = {k: v.clone() for k, v in model.state_dict().items()}, torch.get_rng_state()
    forwards = arms(model, compiled)
    first = lockstep_update(model, forwards["compiled"], state, None, rng, games, True, 40)
    model.load_state_dict(first["state"])
    torch.set_rng_state(first["rng"])
    with InductorCalls() as calls:
        out = val.evaluate(model)
    rng_after = torch.get_rng_state()
    assert calls.n == 0, f"evaluation ran {calls.n} compiled graph calls"
    assert model.training
    torch.set_rng_state(first["rng"])
    expected = val.evaluate(ReferenceCaller(model))
    assert out == expected, "evaluation output differs from the c7580b1 evaluation"
    check_equal(rng_after, torch.get_rng_state(), "RNG state after the evaluation")
    assert all(n > 0 for h in out for _, n, _, _ in out[h][:1]) and set(out) == set(SL.HEADS)
    rec = {arm: lockstep_update(model, forwards[arm], first["state"], first["opt_state"], rng_after, games, True, 40)
           for arm in ("reference", "compiled")}
    compare(rec["compiled"], rec["reference"], "update after the evaluation", exact=False)


def _set_bad_type_token(t):
    t[0][0, 0] = (1 << 13) | (7 << 10)  # a non-empty event with embedding type 7: outside the 4-row type table


INVALID_INPUTS = {
    "non-binary mask": (lambda t: t[2].__setitem__((0, 0), 2), AssertionError, "action_mask_batch must be binary"),
    "row without a legal action": (lambda t: t[2][0].zero_(), AssertionError,
                                   "Each sample must have at least one allowed action"),
    "masked target": (lambda t: t[2].__setitem__((0, int(t[3][0])), 0), AssertionError,
                      "Target action is masked out as invalid!"),
    "negative target": (lambda t: (t[3].__setitem__(0, -1), t[2].__setitem__((0, -1), 1)), IndexError,
                        "Target -1 is out of bounds."),
    "embedding type out of its table": (_set_bad_type_token, IndexError, "index out of range in self"),
}


@pytest.mark.parametrize("case", sorted(INVALID_INPUTS))
def test_invalid_inputs_raise_the_c7580b1_exception_on_the_compiled_path(setup, games, case):
    """[P1] the safety checks run eager, before the compiled network: each invalid input raises the same exception
    type and message in the c7580b1 path, the eager path and the compiled path, and no compiled graph runs."""
    model, compiled = setup
    edit, exc, message = INVALID_INPUTS[case]
    for arm, fwd in arms(model, compiled).items():
        restore = one_head_batch(model, games, "discard", 8, edit)
        try:
            with InductorCalls() as calls, pytest.raises(exc) as info:
                fwd(*games, sample=False)
        finally:
            restore()
        # first line: pytest's assert rewriting appends an explanation to the asserts of this module (the reference)
        assert str(info.value).split("\n")[0] == message, f"{case} {arm}: {info.value!r}"
        assert calls.n == 0, f"{case} {arm}: a compiled graph ran"


def test_fully_masked_attention_row_compiled(setup):
    """The safe-softmax branch of the single-query attention (a row with no key to attend: zero weights, finite
    gradients), which real batches never reach (the state token is always valid): compiled vs eager, train mode with
    dropout, same RNG: outputs and every gradient by the [P8] rule, RNG state equal."""
    torch.manual_seed(3)
    attn = MultiHeadAttention(8, 8).train()
    x = torch.randn(5, 7, 64)
    mask = torch.ones(5, 1, 7, dtype=torch.bool)
    mask[1] = False  # fully masked
    mask[3, 0, 2:] = False
    attn_c = torch.compile(attn, dynamic=True)
    out = {}
    for name, f in (("eager", attn), ("compiled", attn_c)):
        attn.zero_grad()
        xi = x.clone().requires_grad_(True)
        torch.manual_seed(5)
        y = f(xi, xi, xi[:, -1:], mask)
        (y * torch.arange(64.0)).sum().backward()
        out[name] = (y.detach(), xi.grad.clone(), {n: p.grad.clone() for n, p in attn.named_parameters()},
                     torch.get_rng_state())
    (ye, xe, ge, re_), (yc, xc, gc, rc) = out["eager"], out["compiled"]
    assert torch.isfinite(xc).all() and (ye[1] == ye[1, 0]).all()  # the masked row: the projection bias only
    check_close(yc, ye, "output")
    check_close(xc, xe, "input gradient")
    scale = max(g.abs().max().item() for g in ge.values())
    for n in ge:
        if n == "key_proj.bias":  # [R2] structural zero: round-off in both arms, bounded relative to the other grads
            for arm, g in (("eager", ge[n]), ("compiled", gc[n])):
                assert g.abs().max().item() <= 1e-6 * scale, f"{arm}: structural zero {n} |grad| {g.abs().max():.3e}"
            continue
        check_close(gc[n], ge[n], f"grad {n}")
    check_equal(rc, re_, "RNG state")


def reference_record(model, games, seed, updates=3):
    """The reference's lockstep record of its `updates`-th update from the model's weights (non-initial weights and
    AdamW state from the second update on; all heads, sample 40)."""
    forward = arms(model, None)["reference"]
    torch.manual_seed(seed)
    state, opt_state, rng = {k: v.clone() for k, v in model.state_dict().items()}, None, torch.get_rng_state()
    for _ in range(updates):
        rec = lockstep_update(model, forward, state, opt_state, rng, games, True, 40)
        state, opt_state, rng = rec["state"], rec["opt_state"], rec["rng"]
    assert len(rec["heads"]) == 3
    return rec


def test_comparator_fails_on_small_errors(setup, games):
    """The [P8] comparator can fail on every tensor it compares (stage-5 review: a fixed atol of 1e-5 passed any value
    of the AdamW second moments and of small gradients, e.g. during_naki_embedding's). On a real reference record, a
    relative error of 1e-3 in any one compared tensor of any parameter but the structural zeros fails it (1e-2 in the
    AdamW update, whose atol scale is ATOL_SCALE_UPDATE), and so does a 1e-3 error of the valid logits or the loss of
    any head. The dropped negative control (2), one parameter's gradient scaled by 1.01, fails compare() on
    token_embedder.during_naki_embedding.weight."""
    model, _ = setup
    ref = reference_record(model, games, seed=13)
    tried = 0
    for key in ("grad", "clipped", "exp_avg", "exp_avg_sq", "delta"):
        error = 1e-2 if key == "delta" else 1e-3
        for n, b in ref[key].items():
            if n in STRUCTURAL_ZERO:
                continue
            with pytest.raises(AssertionError, match="elements outside"):
                check_close(b * (1 + error), b, f"{key} {n}", atol_scale(key))
            tried += 1
    for h in ref["heads"]:
        b = ref["logits"][h]
        assert (b == MASKED_LOGIT).any() and (b != MASKED_LOGIT).any()
        with pytest.raises(AssertionError, match="elements outside"):
            check_logits(torch.where(b == MASKED_LOGIT, b, b * (1 + 1e-3)), b, f"{h} logits")
        with pytest.raises(AssertionError, match="elements outside"):
            check_close(ref["loss"][h] * (1 + 1e-3), ref["loss"][h], f"{h} loss")
    assert tried == 5 * (len(C7580B1_PARAMS) - len(STRUCTURAL_ZERO))
    n = "token_embedder.during_naki_embedding.weight"
    assert ref["grad"][n].abs().max().item() > COVERAGE_MIN
    got = copy.deepcopy(ref)
    got["grad"][n] = got["grad"][n] * 1.01
    with pytest.raises(AssertionError, match=f"grad {re.escape(n)}: "):
        compare(got, ref, "negative control (2)", exact=False)


def test_negative_control_valid_logit_defect_fails_the_comparison(setup, games):
    """[P9 (1)] the negative control, committed: the compiled path with a defect between its network and the loss
    (the valid, unmasked logits scaled by 1 + 1e-3, the loss computed from them) fails the lockstep comparison with the
    reference, while the same update without the defect passes it."""
    model, compiled = setup

    def defective(head, unique_codes, inverse, x1, x2, x3, action_mask_batch, y_tensor, padding_mask_batch):
        logits, _ = compiled(head, unique_codes, inverse, x1, x2, x3, action_mask_batch, y_tensor, padding_mask_batch)
        logits = torch.where(action_mask_batch == 1, logits * (1 + 1e-3), logits)
        return logits, F.cross_entropy(logits, y_tensor, reduction="mean")

    forwards = dict(arms(model, compiled), defective=lambda *a, **k: model(*a, head_core=defective, **k))
    start = reference_record(model, games, seed=17, updates=2)
    start = (start["state"], start["opt_state"], start["rng"])
    rec = {arm: lockstep_update(model, forwards[arm], *start, games, True, 40)
           for arm in ("reference", "compiled", "defective")}
    compare(rec["compiled"], rec["reference"], "compiled", exact=False)
    with pytest.raises(AssertionError, match="defective: .* (loss|logits): .* elements outside"):
        compare(rec["defective"], rec["reference"], "defective", exact=False)


B1_RECOMPILE = ("Recompiling function head_core in encoder_conv.py:231\n    triggered by the following guard failure(s):\n"
                "    - 0/2: L['head'] == 'post'                                         \n"
                "    - 0/1: 2 <= L['inverse'].size()[0]                                   # _dynamo/output_graph.py:463 in "
                "init_ambient_guards\n    - 0/0: L['head'] == 'during'                                       ")


@pytest.mark.parametrize("guards, kind", [
    (["L['head'] == 'during'"], "head"),
    (["L['head'] == 'during'", "L['head'] == 'discard'"], "head"),
    (["L['head'] == 'post'", "2 <= L['inverse'].size()[0]                 # _dynamo/output_graph.py:463"],
     "batch-size-1"),
    (["tensor 'L['x1']' size mismatch at index 0. expected 1, actual 40"], "batch-size-1"),
    (["2 <= L['inverse'].size()[0]", "2 <= L['x1'].size()[0]"], "other"),
    (["tensor 'L['x1']' size mismatch at index 0. expected 40, actual 39"], "other"),
    (["L['head'] == 'post'", "___check_obj_id(L['self']._modules['ln_f'].training, 4314100920)"], "other"),
    ([], "other"),
])
def test_classify_recompile(guards, kind):
    """The monitor's reading of dynamo's recompile message (one failed guard per cached graph): only another head's
    first compile and the 0/1 batch-size specialization are expected."""
    message = "Recompiling function head_core in x.py:1\n    triggered by the following guard failure(s):\n" + "".join(
        f"    - 0/{k}: {g}\n" for k, g in enumerate(guards))
    assert SL.recompile_guards(message) == [g.split("  #")[0].strip() for g in guards]
    assert SL.classify_recompile(message) == kind
    assert SL.classify_recompile(B1_RECOMPILE) == "batch-size-1"


def test_compile_monitor_counts_recompiles_against_the_heads(games):
    """[R4] the trainer's CompileMonitor in a fresh dynamo state (as in a training process): one graph per head; a
    1-sample batch of a head adds one recompile, logged with its guard and classified batch-size-1, not unexpected,
    and only once per head; any other recompile (here a compiled update in eval mode) is unexpected, and so is a head
    that runs without a graph of this monitor's (graphs below the heads seen: a monitor started after the compiles)."""
    torch._dynamo.reset()  # the module's compiled callable recompiles if it is used after this test
    threads = torch.get_num_threads()
    torch.set_num_threads(SL.TORCH_THREADS)
    try:
        torch.manual_seed(SL.MODEL_SEED)
        model = EncoderConvModel(verbose=False).to(device).train()
        head_core = SL.compile_head_core(model)
        monitor, lines = SL.CompileMonitor(), []

        def update(head=None, n=1, mon=monitor):
            restore = one_head_batch(model, games, head, n) if head else None
            try:
                results = model(*games, sample=True, sample_size=40, head_core=head_core)
                sum(o.loss for o in results.values()).backward()
            finally:
                if restore is not None:
                    restore()
            mon.update_done(results)
            return {k: mon.state()[k] for k in ("graphs", "expected_graphs", "heads_seen", "size1_recompiles",
                                                "unexpected")}

        def want(graphs, expected, heads, size1, unexpected):
            return dict(graphs=graphs, expected_graphs=expected, heads_seen=heads, size1_recompiles=size1,
                        unexpected=unexpected)

        assert update() == want(3, 3, 3, 0, 0)
        assert update("discard") == want(4, 4, 3, 1, 0)
        assert update() == want(4, 4, 3, 1, 0)
        assert update("discard") == want(4, 4, 3, 1, 0)
        monitor.report(lines.append, 4)
        assert len(lines) == 4 and "COMPILE-ALERT" not in lines[-1], lines  # 3 recompiles (2 heads, 1 size), 1 line
        kinds = [" recompile 1 (head)", " recompile 2 (head)", " recompile 3 (batch-size-1)"]
        assert [ln.split(":")[1] for ln in lines[:3]] == kinds, lines
        assert "2 <= L['inverse'].size()[0]" in lines[2]
        late = SL.CompileMonitor()  # started after the compiles: its heads run graphs it did not see compiled
        assert update(mon=late) == want(0, 3, 3, 0, 6)
        model.eval()
        try:
            state = update("post", 8)
        finally:
            model.train()
        assert state["graphs"] == 5 and state["unexpected"] == 3, state  # 1 frame over, 1 graph over, 1 "other"
        monitor.report(lines.append, 5)
        assert "(other): " in lines[-2] and lines[-1].endswith("COMPILE-ALERT"), lines[-2:]
    finally:
        torch.set_num_threads(threads)


RESUME_STEPS, RESUME_FROM = 9, 4  # a compiled run to step 9 (checkpoints 4, 8, 9; a row every step)


def _train_in_this_process(monkeypatch, root, run_dir, log_path, compile_model):
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    monkeypatch.setattr(SL, "VAL_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    SL.train("encoder_conv", RESUME_STEPS, str(run_dir), SL.RunLog(str(log_path)), eval_every=1,
             save_every=RESUME_FROM, compile_model=compile_model)


def _train_in_a_fresh_process(root, run_dir, log_path, compile_model):
    code = (f"import pathlib\nfrom MahjongAI import supervised_learning as SL\nSL.DATA_ROOT = pathlib.Path({str(root)!r})\n"
            f"SL.VAL_GAMES = 2\nSL.PRIOR_GAMES = 2\nSL.train('encoder_conv', {RESUME_STEPS}, {str(run_dir)!r}, "
            f"SL.RunLog({str(log_path)!r}), eval_every=1, save_every={RESUME_FROM}, compile_model={compile_model})\n")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, check=True, timeout=600)


def _files_hashes(log_path):
    """{step: consumed-file hash} of the per-step lines."""
    pattern = r"\] step (\d+) \| .* \| files ([0-9a-f]+)\n"
    return {int(m.group(1)): m.group(2) for m in re.finditer(pattern, open(log_path).read())}


def test_resume_compiled_to_eager_in_a_fresh_process(tmp_path, monkeypatch, train_data_root):
    """[P6] a compiled run's checkpoint holds the c7580b1 keys (no _orig_mod.) and resumes, strict, in a fresh
    process with compile off. Its updates 5..9 equal the compiled run's: the same files (loader position), RNG state
    (so the same sampled indices), scheduler and optimizer step; train and val CE per head by the [P8] rule. The
    compiled run logs its compile activity at every eval: one graph per head, nothing unexpected [R4]."""
    threads = torch.get_num_threads()
    a, b = tmp_path / "a", tmp_path / "b"
    torch._dynamo.reset()  # the trainer's monitor counts compiles from a fresh dynamo state, as in a training process
    try:
        _train_in_this_process(monkeypatch, train_data_root, a, tmp_path / "a.log", compile_model=True)
        shutil.copytree(a, b)
        for s in (8, 9):
            os.remove(b / "checkpoints" / f"step_{s}.pt")
        SL.Metrics(str(b / "metrics.jsonl")).truncate_after(RESUME_FROM)
        _train_in_a_fresh_process(train_data_root, b, tmp_path / "b.log", compile_model=False)
    finally:
        torch.set_num_threads(threads)
    log_a, log_b = open(tmp_path / "a.log").read(), open(tmp_path / "b.log").read()
    assert "torch.compile: on" in log_a and "torch.compile: off" in log_b
    assert f"resuming at step {RESUME_FROM};" in log_b, log_b[-2000:]
    monitor = re.findall(r"at step (\d+): compile: frames (\d+)/(\d+) ok, graphs (\d+) \(expected (\d+)\), graph breaks 0, "
                         r"heads seen (\d), batch-size-1 recompiles (\d+); unexpected \(recompile / fallback / break\) "
                         r"(\d+)", log_a)
    assert [int(m[0]) for m in monitor] == list(range(RESUME_STEPS + 1)), monitor
    assert all(m[7] == "0" for m in monitor) and "COMPILE-ALERT" not in log_a, monitor
    assert monitor[-1][5] == "3" and monitor[-1][3] == monitor[-1][4] == "3", monitor[-1]
    ckpt = torch.load(a / "checkpoints" / f"step_{RESUME_FROM}.pt", weights_only=True)
    assert list(ckpt["model_state_dict"]) == C7580B1_STATE_KEYS
    EncoderConvModel(verbose=False).load_state_dict(ckpt["model_state_dict"], strict=True)
    files_a, files_b = _files_hashes(tmp_path / "a.log"), _files_hashes(tmp_path / "b.log")
    assert [files_a[s] for s in range(RESUME_FROM + 1, RESUME_STEPS + 1)] == \
        [files_b[s] for s in range(RESUME_FROM + 1, RESUME_STEPS + 1)]
    for s in (8, 9):
        ca = torch.load(a / "checkpoints" / f"step_{s}.pt", weights_only=True)
        cb = torch.load(b / "checkpoints" / f"step_{s}.pt", weights_only=True)
        check_equal(ca["torch_rng_state"], cb["torch_rng_state"], f"step {s} RNG state")
        assert ca["loader_state"] == cb["loader_state"] and ca["scheduler_state_dict"] == cb["scheduler_state_dict"]
        assert [v["step"] for v in ca["optimizer_state_dict"]["state"].values()] == \
            [v["step"] for v in cb["optimizer_state_dict"]["state"].values()]
        assert list(cb["model_state_dict"]) == C7580B1_STATE_KEYS
    _, rows_a = SL.Metrics(str(a / "metrics.jsonl")).read()
    _, rows_b = SL.Metrics(str(b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == list(range(RESUME_STEPS + 1))
    for ra, rb in zip(rows_a, rows_b):
        for h in SL.HEADS:
            for key in ("ce", "train_ce"):
                if ra["heads"][h][key] is None:
                    assert rb["heads"][h][key] is None
                    continue
                check_close(torch.tensor(rb["heads"][h][key], dtype=torch.float64),
                            torch.tensor(ra["heads"][h][key], dtype=torch.float64), f"step {ra['step']} {h} {key}")
            assert ra["heads"][h]["n"] == rb["heads"][h]["n"] and ra["heads"][h]["train_n"] == rb["heads"][h]["train_n"]
