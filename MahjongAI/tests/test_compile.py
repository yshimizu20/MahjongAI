"""T-050 Part 2: the trainer's opt-in --compile (agents/compiled_heads.py), one torch.compile(dynamic=False) callable
per head for EncoderConvModel.head_network (the network after the token embedder). Plan
task/plans/T-050-torch-compile-impl.md, "Tests T", with the review fold-in and the stage-3 decision of 2026-10-26.

CPU (every machine, Linux CI too; a counting stub stands in for torch.compile where a test calls the cores):
- the region: forward_heads through the cores equals forward_heads without them bitwise (losses, logits, every
  gradient, RNG state); the tapped test region equals it too.
- the callables stay outside the model: state_dict keys as main's (no `_orig_mod.`), no HeadCore in the module tree.
- the trainer: a --compile run stopped at step 8 and resumed without --compile ends equal to an eager run (rows, step
  lines, checkpoints); evaluation calls no compiled callable; the startup line and the summary line are logged.
- plumbing: --compile needs positions mode (train and the CLI); the run header names it.
- the lockstep comparator: bounded round-off on a structural zero passes; a 1e-6 gradient on one fails; a x(1 + 1e-3)
  error on any other tensor fails.

MPS (marked mps: Linux CI deselects them; locally they must pass):
- lockstep, dropout off: compiled vs eager from the same weights, AdamW state and inputs, 3 updates per (head, B)
  shape of the production sizes (discard 1024; during 96 / 192; post 320 / 640), from non-initial weights: loss,
  valid logits, every gradient, the AdamW update and moments by the 5fdd603 rule; the structural zeros
  (blocks.*.self_attention.key_proj.bias) by absolute bounds. One graph per (head, B) signature, no graph break, only
  allowed fallbacks. Negative control: valid logits x (1 + 1e-3) inside the region must fail.
- dropout on: compiled draws are not eager's (other draw order on MPS); their distribution must match: the encoder
  output (first TAP_ROWS rows) over DRAW_SEEDS seeds, mean squared z < MEAN_Z2 and variance ratio in VAR_RATIO
  (test_mps.py's bars), eager vs eager (calibration) and compiled vs eager, modes "all" and "attention". Negative
  controls in mode "attention" (compiled with attention dropout 0.4; output x 1.05) must fail.
- activation: profiler events CompiledFunction / CompiledFunctionBackward per head in every update; the eager bypass
  (--compile asked, eager cores) must fail that check.
- the trainer on MPS: a --compile run, then an eager resume; summary graphs == signatures, no break.
Each MPS test resets dynamo and its counters first (fresh_dynamo), so no test's graphs count in another's."""
import copy
import re
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from MahjongAI import supervised_learning as SL
from MahjongAI.agents import compiled_heads as CH
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.tests.test_mps import (ARCH, DRAW_SEEDS, MEAN_Z2, REPO, TRAIN_SIZES, VAR_RATIO, _without_dropout,
                                      draw_stats, fixture_batches, load_games)
from MahjongAI.tests.test_positions_train import _run, assert_runs_equal, step_lines

MPS = torch.device("mps")
# production (head, B) shapes on MPS (plan Part 2 "Facts", sizes 96,1024,320): a step takes the discard size and 0,
# N or 2 N during / post rows; the two combos cover the five shapes
COMBOS = ({"during": 96, "discard": 1024, "post": 320}, {"during": 192, "discard": 1024, "post": 640})
LOCKSTEP_UPDATES = 3  # per combo
ATOL_SCALE, RTOL, ATOL_FLOOR = 1e-4, 1e-4, 1e-30  # the 5fdd603 rule: |a - b| <= scale * max|ref| + floor + RTOL |b|
ATOL_SCALE_BY_CLASS = {"delta": 5e-3}  # the AdamW update divides by sqrt(v): a round-off gradient moves it most
STRUCTURAL_ZERO = "self_attention.key_proj.bias"  # every block: softmax ignores a shift shared by every key (P2-5)
ZERO_BOUND = {"grad": 1e-8, "exp_avg": 1e-8, "exp_avg_sq": 1e-16, "delta": 5 * SL.learning_rate}
LOGIT_DEFECT = 1 + 1e-3  # the negative control of the lockstep comparator
TAP_ROWS = 200  # the dropout draw: each head's encoder output, first TAP_ROWS rows (probe.py, G2)
RNG_FALLBACK_OK = ("rand", "bernoulli", "dropout", "seed")  # the ops fallback_random leaves to aten (P2-1)
EXTERN_OK = ("aten.linear_backward",)  # MPS Inductor has no lowering for it (stage-3 decision 2026-10-26)


# ---------------------------------------------------------------------------------------------- shared helpers
@pytest.fixture
def compile_globals(monkeypatch):
    """Restore what CompiledHeads sets process-wide (Inductor's fallback_random, the fallback spy) after the test."""
    import torch._inductor.config as inductor_config
    import torch._inductor.ir as ir
    monkeypatch.setattr(inductor_config, "fallback_random", inductor_config.fallback_random)
    monkeypatch.setattr(ir.FallbackKernel, "create", ir.FallbackKernel.__dict__["create"])


class CountingCompiler:
    """A stand-in for compile_core: the core itself, eager, with a call counter."""

    def __init__(self):
        self.calls = 0

    def __call__(self, core):
        def run(*args):
            self.calls += 1
            return core(*args)
        return run


class TapCore(CH.HeadCore):
    """The region with a tap (probe.py, G2): also returns the encoder output's first TAP_ROWS rows (the state row:
    the last block computes that row alone). The same ops as EncoderConvModel.head_network
    (test_the_cores_run_the_region_of_forward_heads_bitwise checks it)."""

    def forward(self, history, state, padding, mask, y):
        m = self.model
        fused = torch.cat([history, m.state_net(state).unsqueeze(1)], dim=1)
        state_mask = torch.ones(padding.size(0), 1, dtype=torch.bool, device=padding.device)
        x = m.encode(fused, torch.cat([padding, state_mask], dim=1).unsqueeze(1))
        logits = m.heads[self.head](m.ln_f(x[:, -1, :])).masked_fill(mask == 0, -1e9)
        return logits, F.cross_entropy(logits, y, reduction="mean"), x[:TAP_ROWS, -8:]


class ScaledLogitsCore(CH.HeadCore):
    """The broken region of the comparator's negative control: logits x LOGIT_DEFECT, the loss from them."""

    def forward(self, history, state, padding, mask, y):
        logits, _ = self.model.head_network(self.head, history, state, padding, mask, y)
        logits = logits * LOGIT_DEFECT
        return logits, F.cross_entropy(logits, y, reduction="mean")


class EagerCores:
    """{head: core(model, head)} called as CompiledHeads is, uncompiled."""

    def __init__(self, model, core=CH.HeadCore):
        self.cores = {h: core(model, h) for h in SL.HEADS}

    def __call__(self, head, *args):
        return self.cores[head](*args)


def lockstep_problems(got: dict, ref: dict) -> list:
    """Problems of one compiled update record vs the eager one ({"<class>.<name>": CPU tensor}; classes loss, logits
    (valid ones), grad, delta (the AdamW update), exp_avg, exp_avg_sq). Each tensor finite in both; the 5fdd603 rule
    per tensor (ATOL_SCALE_BY_CLASS for the update); a structural zero by ZERO_BOUND in both arms instead."""
    problems = [f"keys differ: {sorted(got.keys() ^ ref.keys())}"] if got.keys() != ref.keys() else []
    for k in sorted(got.keys() & ref.keys()):
        cls, name = k.split(".", 1)
        a, b = got[k].double(), ref[k].double()
        if a.shape != b.shape:
            problems.append(f"{k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            problems.append(f"{k}: not finite")
            continue
        if name.endswith(STRUCTURAL_ZERO):
            for arm, t in (("compiled", a), ("eager", b)):
                if t.numel() and t.abs().max().item() > ZERO_BOUND[cls]:
                    problems.append(f"{k} ({arm}): structural zero max |x| {t.abs().max().item():.3e} > "
                                    f"{ZERO_BOUND[cls]:.1e}")
            continue
        if not b.numel():
            continue
        atol = ATOL_SCALE_BY_CLASS.get(cls, ATOL_SCALE) * b.abs().max().item() + ATOL_FLOOR
        d = (a - b).abs()
        bad = d > atol + RTOL * b.abs()
        if bad.any():
            problems.append(f"{k}: {int(bad.sum())} of {b.numel()} outside, max |d| {d.max().item():.3e} "
                            f"(atol {atol:.3e})")
    return problems


def sized_batch(tensors: dict, sizes: dict, seed: int) -> dict:
    """{head: forward_heads tuple} of exactly sizes[head] rows, drawn from the fixture tensors (a seeded permutation
    repeated as needed: the fixture has fewer during rows than 192)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for h in SL.HEADS:
        tokens, state, mask, y, padding = tensors[h]
        n = len(y)
        idx = torch.cat([torch.randperm(n, generator=g) for _ in range(-(-sizes[h] // n))])[:sizes[h]]
        out[h] = (tokens[idx], tuple(t[idx] for t in state), mask[idx], y[idx], padding[idx])
    return out


def to_device(batch: dict, device) -> dict:
    return {h: (t.to(device), tuple(s.to(device) for s in st), m.to(device), y.to(device), p.to(device))
            for h, (t, st, m, y, p) in batch.items()}


# ---------------------------------------------------------------------------------------------- CPU
@pytest.fixture(scope="module")
def games():
    return load_games()


def _region_run(games, cores_fn, core=CH.HeadCore):
    """One train-mode forward_heads + backward of the seed-0 128 x 2 CPU model, dropout on, seed 5; cores_fn(model)
    gives the cores (None: forward_heads' own)."""
    torch.manual_seed(0)
    model = EncoderConvModel(**ARCH, verbose=False).train()
    cores = cores_fn(model)
    torch.manual_seed(5)
    results = model.forward_heads(fixture_batches(games), cores=cores)
    sum(r.loss for r in results.values()).backward()
    return results, [p.grad for p in model.parameters()], torch.get_rng_state()


def test_the_cores_run_the_region_of_forward_heads_bitwise(games, monkeypatch, compile_globals):
    """CompiledHeads (its compiler a counting stub) and the tapped test region compute forward_heads' own ops:
    equal labels, logits, losses, gradients and RNG state, bit for bit; the cores ran once per head."""
    calls = CountingCompiler()
    monkeypatch.setattr(CH, "compile_core", calls)
    ref, grads_ref, rng_ref = _region_run(games, lambda model: None)
    arms = {"CompiledHeads": lambda model: CH.CompiledHeads(model),
            "TapCore": lambda model: (lambda h, *a: EagerCores(model, TapCore)(h, *a)[:2])}
    for arm, cores_fn in arms.items():
        got, grads, rng = _region_run(games, cores_fn)
        assert list(got) == list(ref) == list(SL.HEADS), arm
        for h in ref:
            for field in ("logits", "loss", "y"):
                assert torch.equal(getattr(got[h], field), getattr(ref[h], field)), (arm, h, field)
        assert all(torch.equal(a, b) for a, b in zip(grads, grads_ref)), arm
        assert torch.equal(rng, rng_ref), arm
    assert calls.calls == len(SL.HEADS)


def test_the_compiled_callables_stay_outside_the_model(compile_globals):
    """Real torch.compile wrappers (not called, so nothing compiles): the model's state_dict keys are main's, none
    with `_orig_mod.`; no HeadCore in its module tree; a deepcopy (the trainer's eval copy) has the same keys."""
    torch.manual_seed(0)
    keys = list(EncoderConvModel(**ARCH, verbose=False).state_dict())
    torch.manual_seed(0)
    model = EncoderConvModel(**ARCH, verbose=False)
    cores = CH.CompiledHeads(model)
    assert set(cores.cores) == set(SL.HEADS)
    for h, c in cores.cores.items():
        assert isinstance(c, torch._dynamo.eval_frame.OptimizedModule) and isinstance(c._orig_mod, CH.HeadCore), h
    assert list(model.state_dict()) == keys and not [k for k in keys if "_orig_mod" in k]
    assert not [m for m in model.modules() if isinstance(m, CH.HeadCore)]
    assert list(copy.deepcopy(model).state_dict()) == keys


def test_a_compiled_run_resumes_eager_to_the_eager_run(tmp_path, monkeypatch, train_data_root, compile_globals):
    """CPU, positions mode, the compiler a counting stub (eager cores): a --compile run stopped at step 8, resumed
    without --compile to step 12, ends equal to the eager run (rows, step lines, checkpoints 4, 8, 12: the same keys
    and values). Every evaluation calls no core; the compiled part logs the startup line and the summary line, the
    eager part neither."""
    calls = CountingCompiler()
    monkeypatch.setattr(CH, "compile_core", calls)
    evaluate, during_eval = SL.ValSet.evaluate, []

    def spy(self, model):
        before = calls.calls
        out = evaluate(self, model)
        during_eval.append(calls.calls - before)
        return out

    monkeypatch.setattr(SL.ValSet, "evaluate", spy)
    a = _run(tmp_path, monkeypatch, train_data_root, "a")
    assert calls.calls == 0
    c = _run(tmp_path, monkeypatch, train_data_root, "c", stop_after_step=8, compile_heads=True)
    trained = calls.calls
    assert trained >= 8  # every compiled update called the cores (the discard head is in every step)
    log = (tmp_path / "c.log").read_text()
    assert log.count(CH.COMPILE_ON_LINE) == 1
    summary = re.findall(r"compile summary \| graphs (\d+) \| signatures (\d+) \| graph breaks (\d+) \| "
                         r"fallbacks \[(.*)\]", log)
    assert len(summary) == 1 and int(summary[0][1]) >= 1, summary
    assert log.index("compile summary") < log.index("stopped after update 8")
    _run(tmp_path, monkeypatch, train_data_root, "c")
    assert calls.calls == trained, "the eager resume called a compiled core"
    log = (tmp_path / "c.log").read_text()
    assert log.count(CH.COMPILE_ON_LINE) == 1 and log.count("compile summary") == 1
    assert "resuming at step 8" in log
    assert during_eval and not any(during_eval), during_eval
    assert_runs_equal(a, c, checkpoints=(4, 8, 12))


def test_compile_needs_positions_mode_and_the_header_names_it(tmp_path):
    with pytest.raises(AssertionError, match="--compile is a positions-mode setting"):
        SL.train("encoder_conv", 10, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "run.log")), batch_mode="games",
                 compile_heads=True)
    assert not (tmp_path / "run").exists()
    from MahjongAI.tests.test_worker_sweep import _args
    plain = SL.run_header(_args())
    assert SL.run_header(_args(compile=True)) == plain + ", compile"
    assert SL.run_header(_args(compile=False)) == plain
    done = subprocess.run([sys.executable, "-m", "MahjongAI.supervised_learning", "encoder_conv", "10", "--compile"],
                          cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 2 and "--compile needs --batch-mode positions" in done.stderr, done.stderr[-500:]
    done = subprocess.run([sys.executable, "-m", "MahjongAI.supervised_learning", "encoder_conv", "10", "--compile",
                           "--batch-mode", "positions", "--device", "cpu"], cwd=REPO, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 2 and "--compile needs --device mps" in done.stderr, done.stderr[-500:]


def _record(seed=0) -> dict:
    """A synthetic lockstep record of the comparator classes, a structural zero included."""
    g = torch.Generator().manual_seed(seed)
    rec = {f"{c}.w": torch.randn(50, generator=g) for c in ("loss", "logits", "grad", "delta", "exp_avg")}
    rec["exp_avg_sq.w"] = torch.rand(50, generator=g) * 1e-6
    zero = f"blocks.0.{STRUCTURAL_ZERO}"
    rec.update({f"grad.{zero}": torch.full((8,), 1e-11), f"delta.{zero}": torch.full((8,), SL.learning_rate),
                f"exp_avg.{zero}": torch.full((8,), 1e-12), f"exp_avg_sq.{zero}": torch.full((8,), 1e-22)})
    return rec


def test_the_lockstep_comparator_passes_round_off_and_fails_real_errors():
    """P2-5's comparator controls: bounded round-off (1e-10) on a structural zero passes; a 1e-6 gradient on one fails;
    a x(1 + 1e-3) error on any other tensor fails, except the AdamW update, whose atol scale is 5e-3 by the plan: its
    control is x(1 + 1e-2) (the moments it is computed from keep the 1e-3 control); a NaN fails."""
    ref = _record()
    assert lockstep_problems(copy.deepcopy(ref), ref) == []
    zero = f"blocks.0.{STRUCTURAL_ZERO}"
    got = copy.deepcopy(ref)
    got[f"grad.{zero}"] = torch.full((8,), 1e-10)
    assert lockstep_problems(got, ref) == []
    got[f"grad.{zero}"] = torch.full((8,), 1e-6)
    assert [p for p in lockstep_problems(got, ref) if p.startswith(f"grad.{zero} (compiled)")]
    for k in [k for k in ref if not k.endswith(STRUCTURAL_ZERO)]:
        got = copy.deepcopy(ref)
        got[k] = got[k] * (1 + (1e-2 if k.startswith("delta.") else 1e-3))
        assert [p for p in lockstep_problems(got, ref) if p.startswith(k + ":")], k
    got = copy.deepcopy(ref)
    got["grad.w"][3] = float("nan")
    assert lockstep_problems(got, ref) == ["grad.w: not finite"]


# ---------------------------------------------------------------------------------------------- MPS
@pytest.fixture
def fresh_dynamo(compile_globals):
    """Dynamo caches, its counters and the fallback record cleared before and after the test."""
    from torch._dynamo.utils import counters

    def clear():
        torch._dynamo.reset()
        counters.clear()
        CH._fallbacks.clear()
        torch.mps.empty_cache()

    clear()
    yield counters
    clear()


@pytest.fixture(scope="module")
def base(games):
    """(fixture tensors, model state, AdamW state) of the seed-0 128 x 2 MPS model after 3 eager training updates,
    dropout on, at the production sizes: the non-initial weights and moments every MPS test starts from."""
    tensors = fixture_batches(games)
    torch.manual_seed(0)
    model = EncoderConvModel(**ARCH, verbose=False).to(MPS).train()
    opt = torch.optim.AdamW(model.parameters(), lr=SL.learning_rate)
    torch.manual_seed(1)
    for k in range(3):
        results = model.forward_heads(sized_batch(tensors, COMBOS[k % 2], seed=100 + k))
        opt.zero_grad()
        sum(r.loss for r in results.values()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        opt.step()
        del results
    torch.mps.synchronize()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return tensors, state, copy.deepcopy(opt.state_dict())


def model_at(state: dict, dropout: str = "all"):
    """A 128 x 2 MPS model in train mode with these weights; dropout "all" (the model's), "attention" (attention
    dropout only), "off"."""
    model = EncoderConvModel(**ARCH, verbose=False).to(MPS)
    model.load_state_dict(state)
    model.train()
    if dropout == "off":
        _without_dropout(model)
    elif dropout == "attention":
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
    return model


def lockstep_record(model, cores, batch: dict, state: dict, opt_state: dict) -> tuple:
    """(record, state after, AdamW state after) of one trainer update (forward_heads, backward, clip, AdamW step) from
    the given weights and AdamW state. record: {"<class>.<name>": CPU tensor} (lockstep_problems' classes)."""
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=SL.learning_rate)
    opt.load_state_dict(copy.deepcopy(opt_state))
    torch.manual_seed(1)
    results = model.forward_heads(batch, cores=cores)
    rec = {}
    for h, r in results.items():
        rec[f"loss.{h}"] = r.loss.detach().clone()
        rec[f"logits.{h}"] = r.logits.detach()[batch[h][2].to(MPS) != 0]
    opt.zero_grad()
    sum(r.loss for r in results.values()).backward()
    del results  # one autograd graph alive at a time
    params = dict(model.named_parameters())
    rec.update({f"grad.{n}": torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                for n, p in params.items()})
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    before = {n: p.detach().clone() for n, p in params.items()}
    opt.step()
    for n, p in params.items():
        rec[f"delta.{n}"] = p.detach() - before[n]
        rec[f"exp_avg.{n}"] = opt.state[p]["exp_avg"]
        rec[f"exp_avg_sq.{n}"] = opt.state[p]["exp_avg_sq"]
    rec = {k: v.detach().cpu().clone() for k, v in rec.items()}
    after = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = rec, after, copy.deepcopy(opt.state_dict())
    torch.mps.empty_cache()
    return out


def run_lockstep(base, core=CH.HeadCore, combos=COMBOS, updates=LOCKSTEP_UPDATES) -> tuple:
    """(problems, CompiledHeads) of the lockstep: per combo, `updates` updates, both arms from the eager arm's
    weights and AdamW state of the previous update, dropout off."""
    tensors, state, opt_state = base
    eager, compiled = model_at(state, "off"), model_at(state, "off")
    cores = CH.CompiledHeads(compiled, core=core)
    problems = []
    for i, sizes in enumerate(combos):
        for k in range(updates):
            batch = sized_batch(tensors, sizes, seed=1000 * (i + 1) + k)
            ref, state_next, opt_next = lockstep_record(eager, None, batch, state, opt_state)
            got, _, _ = lockstep_record(compiled, cores, batch, state, opt_state)
            problems += [f"{sizes} update {k + 1}: {p}" for p in lockstep_problems(got, ref)]
            state, opt_state = state_next, opt_next
    return problems, cores


def bad_fallbacks() -> list:
    return [f for f in CH.fallback_ops() if not any(k in f for k in RNG_FALLBACK_OK + EXTERN_OK)]


@pytest.mark.mps
def test_mps_is_available_for_the_compile_tests():
    assert torch.backends.mps.is_available(), "T-050 test_compile needs MPS on this machine (a skip is a fail)"


@pytest.mark.mps
def test_compiled_equals_eager_in_lockstep_with_dropout_off(base, fresh_dynamo):
    counters = fresh_dynamo
    problems, cores = run_lockstep(base)
    graphs, breaks = counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())
    print(f"\nlockstep dropout off: {len(problems)} problems; graphs {graphs}, signatures {sorted(cores.signatures)}, "
          f"breaks {breaks}, fallbacks {CH.fallback_ops()}")
    assert not problems, problems[:20]
    assert len(cores.signatures) == 5 and graphs == len(cores.signatures), (graphs, cores.signatures)
    assert breaks == 0 and not bad_fallbacks(), (dict(counters["graph_break"]), CH.fallback_ops())


@pytest.mark.mps
def test_the_lockstep_fails_scaled_valid_logits(base, fresh_dynamo):
    """The negative control: valid logits x (1 + 1e-3) inside the compiled region must fail the comparison."""
    problems, _ = run_lockstep(base, core=ScaledLogitsCore, combos=COMBOS[:1], updates=1)
    print(f"\nlogits x {LOGIT_DEFECT}: {len(problems)} problems, e.g. {problems[:3]}")
    assert {h for h in SL.HEADS if any(f"logits.{h}:" in p for p in problems)} == set(SL.HEADS), problems


def activation(model, cores, batch: dict) -> dict:
    """Runtime proof (P2-1): one profiled forward_heads + backward; the AOT autograd compiled function's forward and
    backward events, and the heads of the batch."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        results = model.forward_heads(batch, cores=cores)
        sum(r.loss for r in results.values()).backward()
        torch.mps.synchronize()
    del results
    names = [e.name for e in prof.events()]
    model.zero_grad(set_to_none=True)
    return {"fwd": names.count("CompiledFunction"), "bwd": names.count("CompiledFunctionBackward"), "heads": len(batch)}


def activation_ok(a: dict) -> bool:
    return a["heads"] > 0 and a["fwd"] >= a["heads"] and a["bwd"] >= a["heads"]


@pytest.mark.mps
def test_the_compiled_code_runs_and_the_eager_bypass_fails(base, fresh_dynamo, monkeypatch):
    """Dropout on (the production graphs): after one update per combo (the compiles), a profiled update per combo shows
    the compiled forward and backward of every head; one graph per signature, no break, only allowed fallbacks. The
    bypass (--compile asked, the compiler replaced by the eager core) must fail the activation check."""
    counters = fresh_dynamo
    tensors, state, opt_state = base
    model = model_at(state, "all")
    cores = CH.CompiledHeads(model)
    batches = [sized_batch(tensors, sizes, seed=7 + i) for i, sizes in enumerate(COMBOS)]
    for b in batches:
        activation(model, cores, b)  # compiles fwd + bwd
    acts = [activation(model, cores, b) for b in batches]
    monkeypatch.setattr(CH, "compile_core", lambda core: core)
    bypass = activation(model, CH.CompiledHeads(model), batches[0])
    graphs = counters["stats"]["unique_graphs"]
    print(f"\nactivation {acts}; bypass {bypass}; graphs {graphs}, signatures {sorted(cores.signatures)}, breaks "
          f"{dict(counters['graph_break'])}, fallbacks {CH.fallback_ops()}")
    assert all(activation_ok(a) for a in acts), acts
    assert not activation_ok(bypass), bypass
    assert graphs == len(cores.signatures) == 5, (graphs, cores.signatures)
    assert not counters["graph_break"] and not bad_fallbacks(), (dict(counters["graph_break"]), CH.fallback_ops())


def draws(model, cores, batch: dict, seeds) -> torch.Tensor:
    """[len(seeds), n]: per seed, one train-mode forward of the tapped cores (gradients on, as in training: torch 2.13
    MPS SDPA raises on dropout under no_grad), the token embedder eager first; the draw = every head's tap, flat."""
    out = []
    for s in seeds:
        torch.manual_seed(s)
        parts = []
        for h in SL.HEADS:
            tokens, state, mask, y, padding = batch[h]
            parts.append(cores(h, model.token_embedder(tokens), state, padding, mask, y)[2].detach().float().cpu())
        out.append(torch.cat([p.flatten() for p in parts]))
    torch.mps.empty_cache()
    return torch.stack(out)


def stats_ok(s: tuple) -> bool:
    return s[0] < MEAN_Z2 and VAR_RATIO[0] <= s[1] <= VAR_RATIO[1]


@pytest.fixture(scope="module")
def eager_draws(base):
    """mode -> (model, batch on MPS, eager draws of seeds 1000..): the reference set, shared by the tests below."""
    tensors, state, _ = base
    memo = {}

    def get(mode):
        if mode not in memo:
            model = model_at(state, mode)
            batch = to_device(sized_batch(tensors, COMBOS[0], seed=11), MPS)
            memo[mode] = model, batch, draws(model, EagerCores(model, TapCore), batch, range(1000, 1000 + DRAW_SEEDS))
        return memo[mode]
    return get


@pytest.mark.mps
@pytest.mark.parametrize("mode", ["all", "attention"])
def test_compiled_dropout_keeps_the_eager_distribution(eager_draws, fresh_dynamo, mode):
    """Dropout on: eager vs eager (the calibration) and compiled vs eager pass the distribution check."""
    model, batch, ref = eager_draws(mode)
    calib = draw_stats(draws(model, EagerCores(model, TapCore), batch, range(2000, 2000 + DRAW_SEEDS)), ref)
    comp = draw_stats(draws(model, CH.CompiledHeads(model, core=TapCore), batch, range(3000, 3000 + DRAW_SEEDS)), ref)
    print(f"\ndropout {mode}, mean z^2 / variance ratio (draw {tuple(ref.shape)}): eager vs eager {calib[0]:.3f} / "
          f"{calib[1]:.4f}; compiled vs eager {comp[0]:.3f} / {comp[1]:.4f}")
    assert stats_ok(calib) and stats_ok(comp), (calib, comp)


@pytest.mark.mps
@pytest.mark.parametrize("broken", ["attention dropout 0.4", "output x 1.05"])
def test_the_dropout_check_fails_a_broken_compiled_region(eager_draws, fresh_dynamo, broken):
    """Mode "attention": the compiled region with attention dropout 0.4 (default 0.2), or its draws x 1.05, must
    fail."""
    model, batch, ref = eager_draws("attention")
    model = copy.deepcopy(model)
    if broken == "attention dropout 0.4":
        for m in model.modules():
            if hasattr(m, "attn_dropout_p"):
                m.attn_dropout_p = 0.4
    got = draws(model, CH.CompiledHeads(model, core=TapCore), batch, range(4000, 4000 + DRAW_SEEDS))
    s = draw_stats(got * (1.05 if broken == "output x 1.05" else 1.0), ref)
    print(f"\nattention dropout, compiled, {broken}: mean z^2 {s[0]:.3f}, variance ratio {s[1]:.4f}")
    assert not stats_ok(s), (broken, s)


@pytest.mark.mps
def test_a_compiled_mps_run_resumes_eager(tmp_path, monkeypatch, train_data_root, fresh_dynamo):
    """The trainer on MPS, 128 x 2: a --compile run stopped at step 3 (one startup line; a summary line with graphs ==
    signatures, no break, only allowed fallbacks; no core call inside an evaluation), checkpoint keys as main's, then
    an eager resume to step 5 (no compile line, no core call)."""
    calls, call = [], CH.CompiledHeads.__call__
    monkeypatch.setattr(CH.CompiledHeads, "__call__", lambda self, *a: calls.append(a[0]) or call(self, *a))
    evaluate, during_eval = SL.ValSet.evaluate, []

    def spy(self, model):
        before = len(calls)
        out = evaluate(self, model)
        during_eval.append(len(calls) - before)
        return out

    monkeypatch.setattr(SL.ValSet, "evaluate", spy)
    arch = {"embd_size": ARCH["embd_size"], "n_layers": ARCH["n_layers"], "row0_tol": 0.1}
    _run(tmp_path, monkeypatch, train_data_root, "m", steps=5, save_every=5, sizes=TRAIN_SIZES, device="mps",
         stop_after_step=3, compile_heads=True, **arch)
    compiled_calls = len(calls)
    log = (tmp_path / "m.log").read_text()
    summary = re.findall(r"compile summary \| graphs (\d+) \| signatures (\d+) \| graph breaks (\d+) \| "
                         r"fallbacks \[(.*)\]", log)
    print(f"\n{summary}; core calls {compiled_calls}; evaluations {during_eval}")
    assert log.count(CH.COMPILE_ON_LINE) == 1 and len(summary) == 1, summary
    graphs, signatures, breaks, _ = summary[0]
    assert int(graphs) == int(signatures) >= 1 and breaks == "0" and not bad_fallbacks(), summary
    assert compiled_calls >= 3
    keys = list(EncoderConvModel(**ARCH, verbose=False).state_dict())
    ckpt = torch.load(tmp_path / "m" / "checkpoints" / "step_3.pt", weights_only=True)
    assert list(ckpt["model_state_dict"]) == keys
    _run(tmp_path, monkeypatch, train_data_root, "m", steps=5, save_every=5, sizes=TRAIN_SIZES, device="mps", **arch)
    log = (tmp_path / "m.log").read_text()
    assert len(calls) == compiled_calls and log.count(CH.COMPILE_ON_LINE) == 1 and "resuming at step 3" in log
    assert "Training completed." in log and sorted(step_lines(tmp_path / "m.log")) == [1, 2, 3, 4, 5]
    assert during_eval and not any(during_eval), during_eval
