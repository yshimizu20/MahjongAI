"""T-055 G1 [R7], on this machine's GPU (MPS). These tests must run: a machine without MPS fails them, never skips.

Parity: dropout off, the same weights and tensors (two fixture games, every head): per-head loss within 1e-4, every
gradient within 1e-4 relative (per parameter: max |diff| / max |CPU grad|, the scale floored at 1e-6 x the largest
CPU gradient) and the parameters after one AdamW update (the trainer's lr 1e-3) within 1e-4; a perturbed weight
(ln_f x 1.5) and a NaN gradient must fail it (every loss, gradient and updated parameter is checked finite before the
maxima, which are NaN-aware). The floor is for the attention key bias: softmax ignores a shift of a query's scores, so
its gradient is zero by math and both devices give rounding noise (~1e-11) whose ratio means nothing.
Eval: the val rows of an MPS run (computed on the trainer's CPU copy) equal a CPU model's val of the same weights.
Resume [R6]: an MPS positions-mode run with dropout on, killed after an eval row and resumed. Its data path (read
and sample hashes, loader and batcher states, RNG states, row counters) equals the unbroken MPS run exactly; its
weights and val CE are within 10 x the measured run-to-run noise of two unbroken MPS runs (nn.Embedding backward on
MPS is not deterministic on torch 2.5.1). A resume without the optimizer state or the MPS RNG state must fail that.
Load [R14]: an MPS checkpoint loads on the CPU in the viewer (infer.load_model) and the self-play player
(ModelPlayer); their logits equal the CPU model of the same weights."""
import glob
import json
import math
import os
import re

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.engine import players
from MahjongAI.process_xml import process
from MahjongAI.tests.conftest import build_train_data_root
from MahjongAI.tests.test_legacy_forward_ref import GAMES, GATE_GAMES
from MahjongAI.tests.test_positions_train import _run, step_lines
from MahjongAI.tests.test_resume_rows import _assert_equal
from MahjongAI.turn import HalfTurn
from MahjongAI.viewer import infer

TOL = 1e-4
MPS = torch.device("mps")


def test_mps_is_available():
    assert torch.backends.mps.is_available(), "T-055 G1 needs MPS on this machine (a skip is a fail)"


@pytest.fixture(scope="module")
def games():
    return [process(glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", g))[0]) for g in GAMES]


def _without_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, "attn_dropout_p"):
            m.attn_dropout_p = 0.0
    return model.train()


def assert_finite(label: str, named) -> None:
    """Every (name, tensor or float) value finite; else an AssertionError naming them."""
    bad = [n for n, v in named if not (bool(torch.isfinite(v).all()) if torch.is_tensor(v) else math.isfinite(v))]
    assert not bad, f"non-finite {label}: {bad[:5]}"


def nan_max(values) -> float:
    """max, NaN when any value is NaN (Python's max skips a NaN that comes after a finite value)."""
    values = list(values)
    return math.nan if any(math.isnan(v) for v in values) else max(values)


def _step(model, batches):
    """Per-head loss, gradients and the parameters after one AdamW update (lr 1e-3), all on the CPU."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    results = model.forward_heads(batches)
    sum(r.loss for r in results.values()).backward()
    losses = {h: r.loss.item() for h, r in results.items()}
    grads = [p.grad.cpu().clone() for p in model.parameters()]
    optimizer.step()
    return losses, grads, [p.detach().cpu().clone() for p in model.parameters()]


def parity_diffs(games, perturb=False, nan_grad=False) -> dict:
    """The largest MPS - CPU differences: per-head loss (abs), gradients (relative), updated parameters (abs).
    nan_grad: a hook turns the MPS during_head.bias gradient into NaN (the negative control of the finite checks)."""
    torch.manual_seed(0)
    cpu = _without_dropout(EncoderConvModel(verbose=False))
    halfturns, tokens = [g[0] for g in games], [g[1] for g in games]
    batches = dict(zip(SL.HEADS, cpu.tensor_processor.prepare_batches(halfturns, tokens)))
    gpu = _without_dropout(EncoderConvModel(verbose=False))
    gpu.load_state_dict(cpu.state_dict())
    gpu.to(MPS)
    if perturb:
        with torch.no_grad():
            gpu.ln_f.weight.mul_(1.5)
    if nan_grad:
        gpu.during_head.bias.register_hook(lambda g: torch.full_like(g, float("nan")))
    loss_c, grads_c, params_c = _step(cpu, batches)
    loss_g, grads_g, params_g = _step(gpu, batches)
    assert next(gpu.parameters()).device.type == "mps"
    names = [n for n, _ in cpu.named_parameters()]
    for device, losses, grads, params in (("CPU", loss_c, grads_c, params_c), ("MPS", loss_g, grads_g, params_g)):
        assert_finite(f"{device} loss", losses.items())
        assert_finite(f"{device} gradient", zip(names, grads))
        assert_finite(f"{device} updated parameter", zip(names, params))
    floor = 1e-6 * nan_max(g.abs().max().item() for g in grads_c)
    return {"loss": nan_max(abs(loss_c[h] - loss_g[h]) for h in SL.HEADS),
            "grad_rel": nan_max((a - b).abs().max().item() / max(a.abs().max().item(), floor)
                                for a, b in zip(grads_c, grads_g)),
            "param": nan_max((a - b).abs().max().item() for a, b in zip(params_c, params_g))}


def test_dropout_off_parity_of_loss_gradients_and_one_update(games):
    diffs = parity_diffs(games)
    print(f"\nMPS vs CPU parity (dropout off): {diffs}")
    assert all(v <= TOL for v in diffs.values()), diffs


def test_parity_fails_on_a_perturbed_weight(games):
    diffs = parity_diffs(games, perturb=True)
    print(f"\nMPS (ln_f x 1.5) vs CPU: {diffs}")
    assert diffs["loss"] > TOL and diffs["grad_rel"] > TOL, diffs


def test_parity_fails_on_a_nan_gradient(games):
    """A NaN gradient of one parameter (finite losses; Adam then makes that parameter NaN) must fail the gate
    (Codex round 2: Python's max can skip a NaN after a finite value)."""
    with pytest.raises(AssertionError, match="non-finite"):
        print(f"\nMPS (NaN during_head.bias grad) vs CPU: {parity_diffs(games, nan_grad=True)}")


# ---------------------------------------------------------------- runs on MPS
@pytest.fixture(scope="module")
def mps_root(tmp_path_factory):
    return build_train_data_root(tmp_path_factory.mktemp("mps_data") / "data")


# MPS backward (nn.Embedding) is not deterministic on torch 2.5.1; user decision 2026-10-22: tolerance vs the
# measured run-to-run noise. Two unbroken MPS runs differ by ~1e-8 after one step, so a resumed MPS run is held to
# NOISE_TIMES x the unbroken-vs-unbroken difference at the same step (plus an absolute floor); its data path and RNG
# states stay exact.
NOISE_TIMES, WEIGHT_FLOOR, CE_FLOOR = 10, 1e-6, 1e-5


def _kill_and_resume(tmp_path, mp, root, name, patch=None):
    """A run killed after its row-10 write (step_12.pt and row 12 removed), resumed from step_8.pt with
    `patch` (object, attribute, replacement) in force."""
    run = _run(tmp_path, mp, root, name, device="mps")
    os.remove(run / "checkpoints" / "step_12.pt")
    header, rows = SL.Metrics(str(run / "metrics.jsonl")).read()
    SL.Metrics(str(run / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
    with pytest.MonkeyPatch.context() as patched:
        if patch:
            patched.setattr(*patch)
        _run(tmp_path, mp, root, name, device="mps")
    assert "resuming at step 8;" in open(tmp_path / f"{name}.log").read()


def _without_mps_rng(state, device):
    torch.set_rng_state(state["torch_rng_state"])  # the MPS generator keeps whatever state it has


@pytest.fixture(scope="module")
def mps_runs(tmp_path_factory, mps_root):
    """MPS positions-mode runs, dropout on: "a" and "c" unbroken (a checkpoint every 2 steps; c measures the
    run-to-run noise), "b" killed after its row-10 write and resumed, and two broken resumes (the negative controls):
    "no_opt" restores no optimizer state, "no_rng" restores no MPS RNG state."""
    tmp_path = tmp_path_factory.mktemp("mps")
    with pytest.MonkeyPatch.context() as mp:
        for name in ("a", "c"):
            _run(tmp_path, mp, mps_root, name, save_every=2, device="mps")
        _kill_and_resume(tmp_path, mp, mps_root, "b")
        _kill_and_resume(tmp_path, mp, mps_root, "no_opt",
                         (torch.optim.AdamW, "load_state_dict", lambda self, state: None))
        _kill_and_resume(tmp_path, mp, mps_root, "no_rng", (SL, "set_rng_state", _without_mps_rng))
    return tmp_path


def run_diffs(root, x, y, step=12) -> dict:
    """Run x vs run y at `step`: max |weight diff| (step_<step>.pt) and per-head |val CE diff| (row <step>)."""
    sx, sy = (torch.load(root / r / "checkpoints" / f"step_{step}.pt", map_location="cpu",
                         weights_only=True)["model_state_dict"] for r in (x, y))
    rx, ry = ({r["step"]: r for r in SL.Metrics(str(root / name / "metrics.jsonl")).read()[1]}[step]
              for name in (x, y))
    for name, weights, row in ((x, sx, rx), (y, sy, ry)):
        assert_finite(f"weight in {name}/step_{step}.pt", weights.items())
        assert_finite(f"val CE in {name} row {step}", ((h, row["heads"][h]["ce"]) for h in SL.HEADS))
    return {"weight": nan_max((sx[k].double() - sy[k].double()).abs().max().item() for k in sx),
            "ce": {h: abs(rx["heads"][h]["ce"] - ry["heads"][h]["ce"]) for h in SL.HEADS}}


def within_noise(diffs: dict, noise: dict) -> bool:
    """Every diff <= NOISE_TIMES x the run-to-run noise of the same quantity + its absolute floor."""
    return diffs["weight"] <= NOISE_TIMES * noise["weight"] + WEIGHT_FLOOR and all(
        diffs["ce"][h] <= NOISE_TIMES * noise["ce"][h] + CE_FLOOR for h in SL.HEADS)


def test_run_diffs_fails_on_a_nan_weight(tmp_path):
    """A NaN in the last weight of a checkpoint (after finite ones, where Python's max skips it) must fail."""
    row = {"type": "eval", "step": 12, "heads": {h: {"ce": 1.0} for h in SL.HEADS}}
    for name, last in (("x", 1.0), ("y", math.nan)):
        (tmp_path / name / "checkpoints").mkdir(parents=True)
        weights = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([last])}
        torch.save({"model_state_dict": weights}, tmp_path / name / "checkpoints" / "step_12.pt")
        (tmp_path / name / "metrics.jsonl").write_text(json.dumps({"type": "header"}) + "\n" + json.dumps(row) + "\n")
    assert run_diffs(tmp_path, "x", "x")["weight"] == 0
    with pytest.raises(AssertionError, match=r"non-finite weight in y/step_12.pt: \['b'\]"):
        run_diffs(tmp_path, "x", "y")


def test_mps_resume_is_within_the_run_to_run_noise(mps_runs):
    ckpt = torch.load(mps_runs / "a" / "checkpoints" / "step_12.pt", map_location="cpu", weights_only=True)
    assert ckpt["config"]["device"] == "mps" and "mps_rng_state" in ckpt
    noise, resumed = run_diffs(mps_runs, "a", "c"), run_diffs(mps_runs, "a", "b")
    print(f"\nMPS step 12, unbroken vs unbroken (noise): {noise}\nMPS step 12, resumed vs unbroken: {resumed}")
    assert within_noise(resumed, noise), (resumed, noise)


@pytest.mark.parametrize("broken", ["no_opt", "no_rng"])
def test_the_noise_check_fails_a_broken_resume(mps_runs, broken):
    noise, diffs = run_diffs(mps_runs, "a", "c"), run_diffs(mps_runs, "a", broken)
    print(f"\nMPS step 12, {broken} resume vs unbroken: {diffs} (noise {noise})")
    assert not within_noise(diffs, noise), (broken, diffs, noise)


def test_mps_resume_repeats_the_data_path_of_the_unbroken_mps_run(mps_runs):
    """Exact on MPS: every step reads the same files and trains on the same rows (read hash, sample digest); the
    loader and batcher states and the CPU and MPS RNG states of every checkpoint, and the batcher counters of every
    row, are equal."""
    tail = re.compile(r" \| read [0-9a-f]{12} \| sample [0-9a-f]{12}$")
    lines_a, lines_b = step_lines(mps_runs / "a.log"), step_lines(mps_runs / "b.log")
    assert sorted(lines_a) == sorted(lines_b) == list(range(1, 13))
    assert {s: tail.search(ln).group() for s, ln in lines_a.items()} == {s: tail.search(ln).group()
                                                                         for s, ln in lines_b.items()}
    for step in (4, 8, 12):
        a, b = (torch.load(mps_runs / r / "checkpoints" / f"step_{step}.pt", map_location="cpu", weights_only=True)
                for r in ("a", "b"))
        for key in ("current_step", "loader_state", "batcher_state", "config", "val_ids", "torch_rng_state",
                    "mps_rng_state"):
            _assert_equal(a[key], b[key], f"step_{step}.pt.{key}")
    rows_a, rows_b = (SL.Metrics(str(mps_runs / r / "metrics.jsonl")).read()[1] for r in ("a", "b"))
    _assert_equal([r.get("positions") for r in rows_a], [r.get("positions") for r in rows_b], "rows positions")


def test_val_rows_of_an_mps_run_equal_cpu_eval_of_its_weights(mps_runs, mps_root):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SL, "DATA_ROOT", mps_root)
        mp.setattr(SL, "VAL_GAMES", 2)
        val = SL.ValSet.load(SL.RunLog(str(mps_runs / "val.log")))
    _, rows = SL.Metrics(str(mps_runs / "a" / "metrics.jsonl")).read()
    checked, threads = 0, torch.get_num_threads()
    torch.set_num_threads(SL.TORCH_THREADS)  # the trainer's: the thread count moves CPU logits by float rounding
    for row in rows:
        path = mps_runs / "a" / "checkpoints" / f"step_{row['step']}.pt"
        if not path.exists():
            continue
        model = EncoderConvModel.from_checkpoint(torch.load(path, map_location="cpu", weights_only=True))
        per_game = val.evaluate(model)
        for h in SL.HEADS:
            assert [list(g) for g in per_game[h]] == row["heads"][h]["games"], (row["step"], h)
        checked += 1
    torch.set_num_threads(threads)
    assert checked == 2  # rows 10 and 12


def test_an_mps_checkpoint_loads_in_the_viewer_and_the_self_play_player(mps_runs, games):
    path = str(mps_runs / "a" / "checkpoints" / "step_12.pt")
    ref = EncoderConvModel.from_checkpoint(torch.load(path, map_location="cpu", weights_only=True)).eval()
    viewer, player = infer.load_model(path), players.ModelPlayer(path)
    assert {p.device.type for m in (viewer, player.model) for p in m.parameters()} == {"cpu"}
    n = {"viewer": 0, "player": 0}
    for halfturns, tokens in zip(*games[0]):
        if n["viewer"] > 200:
            break
        for halfturn in halfturns:
            for seat in infer.deciding_seats(halfturn):
                sample = infer.build_sample(halfturn, tokens, seat)
                assert torch.equal(infer.predict(viewer, sample)[1], infer.predict(ref, sample)[1])
                n["viewer"] += 1
            if halfturn.type_ == HalfTurn.DISCARD:
                sample = players.build_sample(halfturn, tokens, halfturn.player)
                assert torch.equal(players.predict(player.model, sample), players.predict(ref, sample))
                n["player"] += 1
    assert n["viewer"] > 50 and n["player"] > 20, n
