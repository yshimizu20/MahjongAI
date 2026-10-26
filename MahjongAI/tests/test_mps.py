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
checkpoints are compared in full: the same keys at every level, every value exact except the weights, the Adam moments
and the train loss sums, which (with the val CE) are within 10 x the measured run-to-run noise of two unbroken MPS runs
(nn.Embedding backward on MPS is not deterministic on torch 2.5.1). A resume without the optimizer state or the MPS RNG
state, and a final checkpoint without its optimizer state or prior, with wrong Adam moments or train sums, must fail.
Load [R14]: an MPS checkpoint loads on the CPU in the viewer (infer.load_model) and the self-play player
(ModelPlayer); their logits equal the CPU model of the same weights.
Driver memory (T-056): a training loop whose batches change the token embedder's unique count U nearly every step
keeps torch.mps.driver_allocated_memory flat (the U padding to UNIQUE_BUCKET); without the padding it grows (the
negative control). The padding leaves the embedder output bitwise equal and its gradients equal within rounding.
Full-row chunks (T-058, 128 x 2): off the CPU the first block runs on ROW_CHUNK-row chunks (the CPU path unchanged).
Dropout off, train and eval: losses, logits, every gradient, the update and the AdamW moments of the chunked model are
within the measured unchunked-vs-unchunked MPS noise (floored at the chunking's float32 rounding, measured on the CPU),
the RNG states after the step equal. Dropout on (draws differ by design): per-element mean and variance over seeds
match the unchunked model's (the draws run with gradients on, as in training: T-064). A train-mode 128 x 2 step with
every dropout on runs on MPS (T-064 guard). A 128 x 2 resume is within the run-to-run noise. Driver memory stays flat
with chunks and grows without (each control in its own process).
Torch >= 2.13 (T-064): the two grows-without controls are skipped with LEAK_CONTROL_REASON (that torch has no driver
leak to reproduce); their stays-flat partners run on every version."""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents import encoder_conv as EC
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.engine import players
from MahjongAI.process_xml import process
from MahjongAI.tests.conftest import build_train_data_root
from MahjongAI.tests.test_legacy_forward_ref import GAMES, GATE_GAMES
from MahjongAI.tests.test_positions_train import _run, step_lines
from MahjongAI.tests.test_resume_rows import _assert_equal
from MahjongAI.turn import HalfTurn
from MahjongAI.viewer import infer

pytestmark = pytest.mark.mps  # Linux CI runs `pytest -m "not mps"`; locally `pytest` runs these and needs MPS

TOL = 1e-4
MPS = torch.device("mps")


def test_mps_is_available():
    assert torch.backends.mps.is_available(), "T-055 G1 needs MPS on this machine (a skip is a fail)"


def load_games():
    return [process(glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", g))[0]) for g in GAMES]


@pytest.fixture(scope="module")
def games():
    return load_games()


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
# NOISE_TIMES x the unbroken-vs-unbroken difference at the same step (plus an absolute floor) for the values MPS
# training moves (weights, Adam moments, train loss sums, val CE); every other checkpoint value is exact.
# The floors are each quantity's float32 rounding scale (a draw where the two unbroken runs agree exactly must not
# fail a resume). Measured noise at step 12: weight 1.5e-6, exp_avg 9.3e-9, exp_avg_sq 2.9e-11, train loss sums 0
# (steps 4 and 8: sums of ~1e3; one float32 ulp of a batch loss x its rows is ~1e-4).
NOISE_TIMES, CE_FLOOR = 10, 1e-5
FLOORS = {"weight": 1e-6, "exp_avg": 1e-7, "exp_avg_sq": 1e-10, "train_loss": 1e-3}


def _kill_and_resume(tmp_path, mp, root, name, patch=None, **arch):
    """A run killed after its row-10 write (step_12.pt and row 12 removed), resumed from step_8.pt with
    `patch` (object, attribute, replacement) in force. arch: SL.train's embd_size / n_layers / row0_tol."""
    run = _run(tmp_path, mp, root, name, device="mps", **arch)
    os.remove(run / "checkpoints" / "step_12.pt")
    header, rows = SL.Metrics(str(run / "metrics.jsonl")).read()
    SL.Metrics(str(run / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
    with pytest.MonkeyPatch.context() as patched:
        if patch:
            patched.setattr(*patch)
        _run(tmp_path, mp, root, name, device="mps", **arch)
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


def noise_class(path: tuple):
    """The noise class of a checkpoint value (its key path), or None: an exact value."""
    if path[0] == "model_state_dict":
        return "weight"
    if path[:2] == ("optimizer_state_dict", "state") and path[-1] in ("exp_avg", "exp_avg_sq"):
        return path[-1]
    if path[0] == "train_sums" and path[-1] == 0:  # [loss x n, n]: the loss sum
        return "train_loss"
    return None


def checkpoint_diffs(x: dict, y: dict, names=("x", "y")) -> dict:
    """{"mismatch": [...], noise class: max |diff|} of two checkpoints: the key sets equal at every level (the
    optimizer and scheduler states, prior_bias and train_sums included), every value exact but the noisy ones
    (noise_class), which must be finite (an AssertionError otherwise)."""
    out = {"mismatch": [], **{c: 0.0 for c in FLOORS}}

    def walk(a, b, path):
        where = ".".join(map(str, path))
        if isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()):
                keys = [sorted(map(str, v)) if isinstance(v, dict) else type(v).__name__ for v in (a, b)]
                out["mismatch"].append(f"{where or '<top>'}: keys {keys[0]} vs {keys[1]}")
                return
            for k in a:
                walk(a[k], b[k], path + (k,))
        elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)):
                out["mismatch"].append(f"{where}: {a!r:.80} vs {b!r:.80}")
                return
            for i, (u, v) in enumerate(zip(a, b)):
                walk(u, v, path + (i,))
        elif torch.is_tensor(a) != torch.is_tensor(b) or torch.is_tensor(a) and (a.dtype, a.shape) != (b.dtype, b.shape):
            out["mismatch"].append(f"{where}: type, dtype or shape differs")
        elif (cls := noise_class(path)) is not None:
            for name, v in zip(names, (a, b)):
                assert_finite(f"{cls} in {name}", [(where, v)])
            d = (a.double() - b.double()).abs().max().item() if torch.is_tensor(a) else abs(a - b)
            out[cls] = max(out[cls], d)
        elif not (torch.equal(a, b) if torch.is_tensor(a) else type(a) is type(b) and a == b):
            out["mismatch"].append(f"{where}: differs")

    walk(x, y, ())
    return out


def run_diffs(x, y, step=12) -> dict:
    """Run dir x vs run dir y at `step`: checkpoint_diffs of step_<step>.pt, and "ce": per-head |val CE diff| of row
    <step> (empty when the step has no row)."""
    paths = [d / "checkpoints" / f"step_{step}.pt" for d in (x, y)]
    names = [f"{p.parent.parent.name}/{p.name}" for p in paths]
    diffs = checkpoint_diffs(*(torch.load(p, map_location="cpu", weights_only=True) for p in paths), names)
    rows = [{r["step"]: r for r in SL.Metrics(str(d / "metrics.jsonl")).read()[1]}.get(step) for d in (x, y)]
    assert (rows[0] is None) == (rows[1] is None), f"row {step} in one run only"
    diffs["ce"] = {}
    if rows[0] is not None:
        for name, row in zip(names, rows):
            assert_finite(f"val CE in {name} row {step}", ((h, row["heads"][h]["ce"]) for h in SL.HEADS))
        diffs["ce"] = {h: abs(rows[0]["heads"][h]["ce"] - rows[1]["heads"][h]["ce"]) for h in SL.HEADS}
    return diffs


def within_noise(diffs: dict, noise: dict) -> bool:
    """No exact value differs, and every noisy one <= NOISE_TIMES x the run-to-run noise of the same quantity + its
    absolute floor."""
    return not diffs["mismatch"] and all(diffs[c] <= NOISE_TIMES * noise[c] + FLOORS[c] for c in FLOORS) and all(
        diffs["ce"][h] <= NOISE_TIMES * noise["ce"][h] + CE_FLOOR for h in diffs["ce"])


def test_run_diffs_fails_on_a_nan_weight(tmp_path):
    """A NaN in the last weight of a checkpoint (after finite ones, where Python's max skips it) must fail."""
    row = {"type": "eval", "step": 12, "heads": {h: {"ce": 1.0} for h in SL.HEADS}}
    for name, last in (("x", 1.0), ("y", math.nan)):
        (tmp_path / name / "checkpoints").mkdir(parents=True)
        weights = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([last])}
        torch.save({"model_state_dict": weights}, tmp_path / name / "checkpoints" / "step_12.pt")
        (tmp_path / name / "metrics.jsonl").write_text(json.dumps({"type": "header"}) + "\n" + json.dumps(row) + "\n")
    assert run_diffs(tmp_path / "x", tmp_path / "x")["weight"] == 0
    with pytest.raises(AssertionError, match=r"non-finite weight in y/step_12.pt: \['model_state_dict.b'\]"):
        run_diffs(tmp_path / "x", tmp_path / "y")


def test_mps_resume_is_within_the_run_to_run_noise(mps_runs):
    """Every checkpoint of the resumed run (4 and 8 before the kill, 12 after the resume) against the unbroken run's:
    the complete schema, exact values exact, noisy values within the noise of the two unbroken runs at that step."""
    ckpt = torch.load(mps_runs / "a" / "checkpoints" / "step_12.pt", map_location="cpu", weights_only=True)
    assert ckpt["config"]["device"] == "mps" and "mps_rng_state" in ckpt
    for step in (4, 8, 12):
        noise = run_diffs(mps_runs / "a", mps_runs / "c", step)
        resumed = run_diffs(mps_runs / "a", mps_runs / "b", step)
        print(f"\nMPS step {step}, unbroken vs unbroken (noise): {noise}\nMPS step {step}, resumed vs unbroken: {resumed}")
        assert not noise["mismatch"], noise["mismatch"]
        assert within_noise(resumed, noise), (step, resumed, noise)


@pytest.mark.parametrize("broken", ["no_opt", "no_rng"])
def test_the_noise_check_fails_a_broken_resume(mps_runs, broken):
    noise, diffs = run_diffs(mps_runs / "a", mps_runs / "c"), run_diffs(mps_runs / "a", mps_runs / broken)
    print(f"\nMPS step 12, {broken} resume vs unbroken: {diffs} (noise {noise})")
    assert not within_noise(diffs, noise), (broken, diffs, noise)


def _scale_largest_exp_avg(state):
    k = max(state, key=lambda i: state[i]["exp_avg"].abs().max().item())
    state[k]["exp_avg"].mul_(2.0)


def _zero_an_exp_avg_sq(state):
    state[max(state)]["exp_avg_sq"].zero_()


DOCTORS = {"no_optimizer_state": lambda c: c.pop("optimizer_state_dict"),
           "scaled_exp_avg": lambda c: _scale_largest_exp_avg(c["optimizer_state_dict"]["state"]),
           "zeroed_exp_avg_sq": lambda c: _zero_an_exp_avg_sq(c["optimizer_state_dict"]["state"]),
           "no_prior_bias": lambda c: c.pop("prior_bias"),
           "train_sums_off": lambda c: c["train_sums"]["discard"].__setitem__(1, c["train_sums"]["discard"][1] + 1)}


@pytest.mark.parametrize("doctor", sorted(DOCTORS))
def test_the_checkpoint_comparison_fails_a_doctored_resumed_checkpoint(mps_runs, tmp_path, doctor):
    """The resumed run's final checkpoint with a state a resume would need missing or wrong (weights, val rows and
    data path untouched, so the weight / CE checks alone pass it) must fail the comparison (Codex round 2)."""
    step = 12 if doctor != "train_sums_off" else 8  # step 12 is an eval step: its train sums are zero
    shutil.copytree(mps_runs / "b", tmp_path / "doctored")
    path = tmp_path / "doctored" / "checkpoints" / f"step_{step}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    DOCTORS[doctor](ckpt)
    torch.save(ckpt, path)
    noise, diffs = run_diffs(mps_runs / "a", mps_runs / "c", step), run_diffs(mps_runs / "a", tmp_path / "doctored", step)
    print(f"\nMPS step {step}, resumed run with {doctor} vs unbroken: {diffs} (noise {noise})")
    assert diffs["weight"] <= NOISE_TIMES * noise["weight"] + FLOORS["weight"]  # the weight check alone passes it
    assert not within_noise(diffs, noise), (doctor, diffs, noise)


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


# ---------------------------------------------------------------- driver memory (T-056)
MIB = 2 ** 20
MEMORY_STEPS, MEMORY_EVERY, MEMORY_FROM = 300, 25, 100  # the slope is fitted over steps MEMORY_FROM..MEMORY_STEPS
MEMORY_ROWS = {"during": 48, "discard": 512, "post": 160}  # rows per step, drawn at random from the fixture games
MEMORY_SLOPE = 0.5  # MiB/step: the padded loop must stay below; the unpadded control must exceed it


def leak_controls_skipped(version: str) -> bool:
    """True on torch >= 2.13 (T-064). TorchVersion compares release numbers: as strings, "2.5.1" >= "2.13"."""
    return torch.torch_version.TorchVersion(version) >= (2, 13)


# T-064 (pytest.ini names this exception to "a skip is a fail"): torch 2.13 has no T-056 / T-058 driver leak, so these
# negative controls (the fix removed -> the slope must exceed MEMORY_SLOPE) have nothing to reproduce there. They run
# on torch 2.5.1; on torch >= 2.13 they are skipped with exactly this reason (the suite gate allows no other skip).
TORCH_2_13 = leak_controls_skipped(torch.__version__)
LEAK_CONTROL_REASON = ("T-064: torch >= 2.13 has no T-056 / T-058 MPS driver leak to reproduce (T-060 Part 3: "
                       "slopes 0.0018 / 0.0105 MiB/step, need >= 0.5); this control runs on torch 2.5.1")
VERSION_GATED = {"test_mps_driver_memory_grows_without_the_unique_padding",
                 "test_mps_driver_memory_grows_without_row_chunks"}


def driver_memory_slope(games, seed: int, steps=MEMORY_STEPS, arch=None, rows=None) -> tuple:
    """(slope MiB/step, {step: driver MiB}) of torch.mps.driver_allocated_memory in a 64 x 1 MPS training loop
    (forward_heads, backward, AdamW, one loss read per step as in the trainer), sampled every MEMORY_EVERY steps, the
    least-squares slope over steps MEMORY_FROM..steps. Each step's rows are a fresh random subset of the fixture
    games' rows (drawn with `seed`), so U changes nearly every step. MPS keeps a compiled shape for the process: a
    second loop with the same seed and padding sees no new shape, so each test draws its own rows. arch: the
    model's embd_size / n_layers (64 x 1 when None); rows(step): {head: rows} of that step (MEMORY_ROWS when None)."""
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False, **(arch or {})).to(MPS)
    full = dict(zip(SL.HEADS, model.tensor_processor.prepare_batches([g[0] for g in games], [g[1] for g in games])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pick, samples = torch.Generator().manual_seed(seed), {}
    for step in range(1, steps + 1):
        batches, sizes = {}, MEMORY_ROWS if rows is None else rows(step)
        for h, (tokens, state, mask, y, padding) in full.items():
            if h not in sizes:
                continue
            idx = torch.randperm(len(y), generator=pick)[:sizes[h]]
            batches[h] = (tokens[idx], tuple(t[idx] for t in state), mask[idx], y[idx], padding[idx])
        loss = sum(r.loss for r in model.forward_heads(batches).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert math.isfinite(loss.item())
        if step % MEMORY_EVERY == 0:
            samples[step] = torch.mps.driver_allocated_memory() / MIB
    pts = [(s, v) for s, v in samples.items() if s >= MEMORY_FROM]
    mx, my = sum(s for s, _ in pts) / len(pts), sum(v for _, v in pts) / len(pts)
    return sum((s - mx) * (v - my) for s, v in pts) / sum((s - mx) ** 2 for s, _ in pts), samples


def test_mps_driver_memory_stays_flat_when_the_unique_count_changes(games):
    assert all(len(full_y) >= MEMORY_ROWS[h] for h, full_y in
               zip(SL.HEADS, (t[3] for t in EC.TransformerTensorProcessor().prepare_batches(*zip(*games)))))
    slope, samples = driver_memory_slope(games, seed=0)
    print(f"\nMPS driver memory, U padded to {EC.UNIQUE_BUCKET}: slope {slope:.3f} MiB/step; {samples}")
    assert slope < MEMORY_SLOPE, (slope, samples)


@pytest.mark.skipif(TORCH_2_13, reason=LEAK_CONTROL_REASON)
def test_mps_driver_memory_grows_without_the_unique_padding(games, monkeypatch):
    """The negative control: UNIQUE_BUCKET 1 (no padding, the raw U) must fail the flat-memory check."""
    monkeypatch.setattr(EC, "UNIQUE_BUCKET", 1)
    slope, samples = driver_memory_slope(games, seed=1)
    print(f"\nMPS driver memory, raw U: slope {slope:.3f} MiB/step; {samples}")
    assert slope >= MEMORY_SLOPE, (slope, samples)


PADDED = 1024  # the bucket of the padded embedder; 1 is the raw U
RAW_RUNS = 3  # raw-vs-raw pairs of these runs measure the MPS gradient noise
NOISE_FACTOR, NOISE_FLOOR = 3, 1e-7  # an MPS padded-vs-raw gradient may differ by up to 3 x the noise + 1e-7 relative


def embedder_run(tokens, device, bucket: int, monkeypatch) -> tuple:
    """(output, {table: gradient}) of a seed-0 TokenEmbedder with UNIQUE_BUCKET = bucket, on the CPU after the run."""
    monkeypatch.setattr(EC, "UNIQUE_BUCKET", bucket)
    torch.manual_seed(0)
    embedder = EC.TokenEmbedder().to(device)
    x = embedder(tokens.to(device))
    (x * torch.linspace(-1, 1, x.numel(), device=device).view_as(x)).sum().backward()
    return x.detach().cpu(), {n: p.grad.cpu() for n, p in embedder.named_parameters()}


def relative_grad_diffs(grads: dict, ref: dict) -> dict:
    """{table: max |grads - ref| / max |ref|}. Every gradient must be finite: a NaN diff compares False against any
    limit, so a NaN gradient would pass the noise check."""
    assert grads.keys() == ref.keys(), (grads.keys(), ref.keys())
    bad = sorted(n for n in ref if not (torch.isfinite(grads[n]).all() and torch.isfinite(ref[n]).all()))
    assert not bad, f"non-finite gradients: {bad}"
    return {n: (grads[n] - g).abs().max().item() / max(g.abs().max().item(), 1e-12) for n, g in ref.items()}


def test_relative_grad_diffs_rejects_a_non_finite_gradient():
    """The negative control of the finite check (Codex T-062 round 2): a NaN or inf in either side raises."""
    good = {"w": torch.ones(3), "b": torch.ones(2)}
    assert relative_grad_diffs(good, good) == {"w": 0.0, "b": 0.0}
    for value in (float("nan"), float("inf")):
        bad = {**good, "w": torch.tensor([1.0, value, 1.0])}
        for a, b in ((bad, good), (good, bad)):
            with pytest.raises(AssertionError, match=r"non-finite gradients: \['w'\]"):
                relative_grad_diffs(a, b)


def all_tokens(games):
    return torch.cat([b[0] for b in EC.TransformerTensorProcessor().prepare_batches(*zip(*games))])


def test_unique_padding_is_exact_on_the_cpu(games, monkeypatch):
    """The padded path forced on the CPU (pads_unique_codes patched; production pads off the CPU only): the output
    and the gradient of every embedder table equal the raw-U path's bit for bit, so padding changes no value."""
    tokens = all_tokens(games)
    rows, embed = [], EC.TokenEmbedder._embed_codes
    monkeypatch.setattr(EC.TokenEmbedder, "_embed_codes", lambda self, t, p: rows.append(t.numel()) or embed(self, t, p))
    monkeypatch.setattr(EC, "pads_unique_codes", lambda device: True)
    out, grads = {}, {}
    for bucket in (PADDED, 1):
        out[bucket], grads[bucket] = embedder_run(tokens, "cpu", bucket, monkeypatch)
    u = rows[1]
    assert u % PADDED and rows == [u + (-u % PADDED), u], rows  # the padded run computed padded rows
    print(f"\nCPU padded ({PADDED}) vs raw (1), U {u}: max abs output diff "
          f"{(out[PADDED] - out[1]).abs().max().item():.3g}, max relative gradient diff "
          f"{max(relative_grad_diffs(grads[PADDED], grads[1]).values()):.3g}")
    assert torch.equal(out[PADDED], out[1])
    assert grads[PADDED].keys() == grads[1].keys()
    for n, g in grads[1].items():
        assert torch.equal(grads[PADDED][n], g), n


def test_unique_padding_keeps_the_embedder_output_and_its_gradients(games, monkeypatch):
    """On MPS: the padded embedder (U -> a multiple of PADDED) gives the raw-U embedder's output bit for bit; the
    gradient of every table is within NOISE_FACTOR x the measured raw-vs-raw noise of that table + NOISE_FLOOR
    (relative): MPS embedding backward is not deterministic (T-055 rule: MPS checks use a measured noise tolerance,
    the CPU stays exact). The CPU computes U rows, MPS the padded count."""
    tokens = all_tokens(games)
    out, grads = embedder_run(tokens, MPS, PADDED, monkeypatch)
    raw = [embedder_run(tokens, MPS, 1, monkeypatch) for _ in range(RAW_RUNS)]
    pairs = [relative_grad_diffs(raw[j][1], raw[i][1]) for i in range(RAW_RUNS) for j in range(i + 1, RAW_RUNS)]
    noise = {n: max(p[n] for p in pairs) for n in pairs[0]}
    diff = relative_grad_diffs(grads, raw[0][1])
    print(f"\nMPS padded ({PADDED}) vs raw (1): max abs output diff {(out - raw[0][0]).abs().max().item():.3g}; "
          "per table relative gradient diff / raw-vs-raw noise: "
          + ", ".join(f"{n} {diff[n]:.3g} / {noise[n]:.3g}" for n in diff))
    assert all(torch.equal(r[0], raw[0][0]) for r in raw)
    assert torch.equal(out, raw[0][0])
    bad = {n: (diff[n], noise[n]) for n in diff if diff[n] > NOISE_FACTOR * noise[n] + NOISE_FLOOR}
    assert not bad, bad
    monkeypatch.setattr(EC, "UNIQUE_BUCKET", PADDED)
    rows, embed = [], EC.TokenEmbedder._embed_codes
    monkeypatch.setattr(EC.TokenEmbedder, "_embed_codes", lambda self, t, p: rows.append(t.numel()) or embed(self, t, p))
    u = len(torch.unique(tokens * EC.MAX_ACTION_LEN + torch.arange(tokens.size(1))))
    EC.TokenEmbedder()(tokens)
    EC.TokenEmbedder().to(MPS)(tokens.to(MPS))
    assert u % PADDED and rows == [u, u + (-u % PADDED)], (u, rows)  # the CPU computes U rows, MPS the padded count


# ---------------------------------------------------------------- full-row chunks (T-058)
ARCH = {"embd_size": 128, "n_layers": 2}  # the first block computes every row, the last one the state row only
CHUNK_RAW_RUNS = 3  # unchunked MPS runs: their pairwise differences measure the noise of each value
DRAW_SEEDS = 32  # dropout on: draws per path for the distribution check
MEAN_Z2, VAR_RATIO = 1.5, (0.9, 1.1)  # dropout on: bound of the mean squared z of the per-element means; variance band
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWITCH_ROWS = (1024, 96, 320, 640, 192)  # the row counts of a trainer stream; the memory loop cycles through them
CHUNK_MEMORY_STEPS = 600  # a no-chunk loop moves +-500 MiB between samples (the allocator frees and grows): 300 steps
# read 0.204 MiB/step, 600 steps 1.431


def set_chunks(monkeypatch, chunked: bool, cpu: bool = False):
    """chunked: the production rule (chunks off the CPU; on the CPU too when cpu); else the fix removed."""
    monkeypatch.setattr(EC, "chunks_full_rows", (lambda device: cpu or device.type != "cpu") if chunked
                        else (lambda device: False))


def chunk_model(device, dropout):
    """The seed-0 128 x 2 model in train mode; dropout: True / "all" (the model's), "attention" (attention dropout
    only: the draws inside SDPA), False (none)."""
    torch.manual_seed(0)
    model = EncoderConvModel(**ARCH, verbose=False)
    if dropout in (True, "all"):
        return model.train().to(device)
    _without_dropout(model)
    if dropout == "attention":
        for block in model.blocks:
            block.self_attention.attn_dropout_p = EC.DROPOUT_RATIO
    return model.to(device)


def fixture_batches(games) -> dict:
    return dict(zip(SL.HEADS, EC.TransformerTensorProcessor().prepare_batches(*zip(*games))))


def block_rows(model, batches, monkeypatch) -> list:
    """[(block index, rows)] of every EncoderBlock call in one forward_heads."""
    calls, forward = [], EC.EncoderBlock.forward
    index = {id(b): i for i, b in enumerate(model.blocks)}

    def counted(self, x, attn_mask, state_row_only=False):
        calls.append((index[id(self)], x.size(0)))
        return forward(self, x, attn_mask, state_row_only)

    with monkeypatch.context() as mp:
        mp.setattr(EC.EncoderBlock, "forward", counted)
        model.forward_heads(batches)
    return calls


def test_full_row_blocks_run_on_fixed_chunks_off_the_cpu(games, monkeypatch):
    """The CPU runs the first block on each head's rows (unchanged); MPS only on ROW_CHUNK-row chunks (the last one
    of a head padded); the last block sees each head's rows on both."""
    batches = fixture_batches(games)
    rows = [len(batches[h][3]) for h in SL.HEADS]
    assert any(n % EC.ROW_CHUNK for n in rows), rows  # a padded chunk is exercised
    calls = {d: block_rows(chunk_model(d, dropout=False), batches, monkeypatch) for d in ("cpu", "mps")}
    print(f"\nrows per head {rows}; first-block calls: CPU {[n for i, n in calls['cpu'] if i == 0]}, MPS "
          f"{[n for i, n in calls['mps'] if i == 0]}")
    assert [n for i, n in calls["cpu"] if i == 0] == rows
    assert [n for i, n in calls["mps"] if i == 0] == [EC.ROW_CHUNK] * sum(-(-n // EC.ROW_CHUNK) for n in rows)
    for d in ("cpu", "mps"):
        assert [n for i, n in calls[d] if i == 1] == rows, d


def chunk_step(batches, device, chunked: bool, mode: str, monkeypatch, cpu_chunks: bool = False) -> dict:
    """{name: CPU tensor} of one step of the seed-0 128 x 2 model, dropout off, in `mode` ("train" / "eval"), the CPU
    and MPS generators seeded alike: per head the loss and the legal logits, every gradient, every parameter and AdamW
    moment after one update (the trainer's lr), the RNG states after it."""
    set_chunks(monkeypatch, chunked, cpu=cpu_chunks)
    model = chunk_model(device, dropout=False).train(mode == "train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=SL.learning_rate)
    torch.manual_seed(1)  # the CPU and the MPS generator
    results = model.forward_heads(batches)
    sum(r.loss for r in results.values()).backward()
    out = {}
    for h, r in results.items():
        out[f"loss.{h}"] = r.loss
        out[f"logits.{h}"] = r.logits[r.logits > -1e8]  # the legal actions (masked ones are -1e9)
    out.update({f"grad.{n}": p.grad.clone() for n, p in model.named_parameters()})
    optimizer.step()
    for n, p in model.named_parameters():
        out[f"param.{n}"] = p
        out.update({f"{k}.{n}": optimizer.state[p][k] for k in ("exp_avg", "exp_avg_sq")})
    out["rng.cpu"] = torch.get_rng_state()
    if device.type == "mps":
        out["rng.mps"] = torch.mps.get_rng_state()
    values = {k: v.detach().cpu().clone() for k, v in out.items()}
    assert_finite(f"{device.type} {mode} step value", ((k, v) for k, v in values.items() if not k.startswith("rng")))
    return values


def step_diffs(x: dict, ref: dict) -> dict:
    """{name: max |x - ref| / scale} of the float values; scale: max |ref| floored at 1e-6 x the largest of its class
    (grad, param, ...): a value zero by math (the attention key bias gradient) leaves rounding noise only."""
    assert x.keys() == ref.keys(), (x.keys() ^ ref.keys())
    cls_max = {}
    for k, v in ref.items():
        if v.is_floating_point():
            cls = k.split(".")[0]
            cls_max[cls] = max(cls_max.get(cls, 0.0), v.abs().max().item())
    return {k: (x[k] - v).abs().max().item() / max(v.abs().max().item(), 1e-6 * cls_max[k.split(".")[0]], 1e-30)
            for k, v in ref.items() if v.is_floating_point()}


# The fixed bound of chunked vs unchunked, relative (step_diffs), per value class: chunking changes the matmul shapes
# and sums each block weight's gradient over chunk partial sums, so float32 rounds in another order (the CPU is not
# bitwise; its unchunked path is). Set after one CPU measurement, before this comparator (2026-10-23, 128 x 2,
# dropout off, worst value per class): loss 0, logits 2.8e-7, grad / exp_avg 2.4e-5, exp_avg_sq 1.1e-5, param 2.9e-4
# (one AdamW step divides each gradient by its own magnitude: a small gradient's rounding moves its parameter most).
CHUNK_TOL = {"loss": 1e-5, "logits": 1e-5, "grad": 1e-4, "exp_avg": 1e-4, "exp_avg_sq": 1e-4, "param": 1e-3}
ZERO_GRAD = "self_attention.key_proj.bias"  # softmax ignores a shift shared by every key: this gradient is 0 by math
CHUNK_MUTATION = 1 + 1e-4  # the negative control scales every chunk's block output by this


def chunk_parity(batches, mode: str, monkeypatch) -> tuple:
    """(bad, report) of the chunked step vs the unchunked one (chunk_step, matched weights, inputs and generator
    states). CPU (the chunks forced on): every value within CHUNK_TOL of its class. MPS: within NOISE_FACTOR x the
    unchunked-vs-unchunked MPS noise of that value + NOISE_FLOOR (the T-055 rule) + CHUNK_TOL (the same reordering);
    the noise comes from unchunked runs only, so the candidate cannot widen its limit. A parameter whose gradient is 0
    by math (ZERO_GRAD) holds AdamW-normalized rounding (+-lr after one step): it is checked within 2 x lr absolute
    (its gradient keeps the grad rule: step_diffs scales it by the class floor). The RNG states after the step must be
    equal.
    bad: {"cpu|mps name": (diff, limit)}."""
    cpu = torch.device("cpu")
    bad, report = {}, []

    def check(leg, x, ref, noise):
        diff = step_diffs(x, ref)
        limit = {k: NOISE_FACTOR * noise.get(k, 0.0) + NOISE_FLOOR + CHUNK_TOL[k.split(".")[0]] for k in diff}
        for k in [k for k in diff if k.startswith("param.") and k.endswith(ZERO_GRAD)]:
            diff[k], limit[k] = (x[k] - ref[k]).abs().max().item(), 2 * SL.learning_rate
        bad.update({f"{leg} {k}": (diff[k], limit[k]) for k in diff if diff[k] > limit[k]})
        for k in [k for k in ref if k.startswith("rng.")]:
            if not torch.equal(x[k], ref[k]):
                bad[f"{leg} {k}"] = ("differs", "equal")
        by_class = {}
        for k in diff:
            c = k.split(".")[0]
            by_class[c] = max(by_class.get(c, 0.0), diff[k] / limit[k])
        report.append(f"{leg} {mode}: largest diff / limit per class "
                      + ", ".join(f"{c} {v:.3g}" for c, v in by_class.items()))

    check("cpu", chunk_step(batches, cpu, True, mode, monkeypatch, cpu_chunks=True),
          chunk_step(batches, cpu, False, mode, monkeypatch), {})
    raw = [chunk_step(batches, MPS, False, mode, monkeypatch) for _ in range(CHUNK_RAW_RUNS)]
    noise = {}
    for i in range(CHUNK_RAW_RUNS):
        for j in range(i + 1, CHUNK_RAW_RUNS):
            for k, v in step_diffs(raw[j], raw[i]).items():
                noise[k] = max(noise.get(k, 0.0), v)
    report.append(f"mps {mode}: unchunked-vs-unchunked noise 0 in {sum(v == 0 for v in noise.values())} of "
                  f"{len(noise)} values, max {max(noise.values()):.3g}")
    check("mps", chunk_step(batches, MPS, True, mode, monkeypatch), raw[0], noise)
    return bad, report


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_row_chunks_keep_losses_gradients_and_the_update(games, monkeypatch, mode):
    """128 x 2, dropout off: chunk_parity passes on the CPU and on MPS."""
    bad, report = chunk_parity(fixture_batches(games), mode, monkeypatch)
    print("\n" + "\n".join(report))
    assert not bad, bad


def test_row_chunk_check_fails_a_scaled_chunk(games, monkeypatch):
    """The negative control: the same comparator with every chunk's block output x CHUNK_MUTATION, a patch held over
    the CPU and the MPS legs alike (an unchunked call has a head's rows, never ROW_CHUNK: 91 / 1,190 / 344), must
    fail on both legs."""
    batches = fixture_batches(games)
    assert EC.ROW_CHUNK not in [len(b[3]) for b in batches.values()]
    forward, hits = EC.EncoderBlock.forward, []

    def scaled(self, x, attn_mask, state_row_only=False):
        y = forward(self, x, attn_mask, state_row_only)
        if not state_row_only and x.size(0) == EC.ROW_CHUNK:
            hits.append(x.device.type)
            return y * CHUNK_MUTATION
        return y

    monkeypatch.setattr(EC.EncoderBlock, "forward", scaled)
    bad, report = chunk_parity(batches, "train", monkeypatch)
    print(f"\nevery chunk x {CHUNK_MUTATION}: {len(bad)} values over the limit, e.g. {dict(list(bad.items())[:3])}\n"
          + "\n".join(report))
    assert {"cpu", "mps"} <= set(hits)
    for leg in ("cpu", "mps"):
        assert any(k.startswith(leg + " ") for k in bad), (leg, bad)


def encode_draws(seeds, chunked: bool, monkeypatch, dropout="all", chunk_patch=None) -> torch.Tensor:
    """[len(seeds), B, 8, EMBD] first-block outputs (the last 8 positions, the state row included) of the seed-0
    128 x 2 model on MPS, dropout on, one encode per seed, on fixed random inputs (B = 200 rows: one full chunk and one
    padded). The first block's output is the last block's input (a pre-hook reads it): the chunked values themselves.
    Gradients stay on, as in training (T-064): torch 2.13's MPS SDPA raises on dropout under no_grad; the read
    values are detached, so no graph outlives its encode.
    chunk_patch(block, x, mask, forward): replaces a chunk call (a negative control)."""
    set_chunks(monkeypatch, chunked)
    model = chunk_model(MPS, dropout=dropout)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(200, 151, ARCH["embd_size"], generator=g).to(MPS)
    mask = torch.rand(200, 1, 151, generator=g) > 0.3
    mask[..., -1] = True
    forward = EC.EncoderBlock.forward
    outs = []
    hook = model.blocks[-1].register_forward_pre_hook(lambda module, args: outs.append(args[0][:, -8:].detach().cpu()))
    assert torch.is_grad_enabled()
    with monkeypatch.context() as mp:
        if chunk_patch:
            mp.setattr(EC.EncoderBlock, "forward", lambda self, x, m, state_row_only=False: forward(
                self, x, m, state_row_only) if state_row_only else chunk_patch(self, x, m, forward))
        for s in seeds:
            torch.manual_seed(s)
            model.encode(x, mask.to(MPS))
    hook.remove()
    assert len(outs) == len(seeds)
    return torch.stack(outs)


def draw_stats(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """(mean over elements of the squared z of the per-element mean difference, mean variance ratio a / b) of two
    [K, ...] stacks of draws."""
    va, vb = a.var(0), b.var(0)
    z2 = (a.mean(0) - b.mean(0)) ** 2 / ((va + vb) / a.size(0)).clamp_min(1e-30)
    return z2.mean().item(), (va.mean() / vb.mean()).item()


def _attention_dropout(p):
    def patch(block, x, mask, forward):
        saved, block.self_attention.attn_dropout_p = block.self_attention.attn_dropout_p, p
        try:
            return forward(block, x, mask)
        finally:
            block.self_attention.attn_dropout_p = saved
    return patch


CHUNK_BREAKS = {"attention dropout 0.4 in the chunks": _attention_dropout(0.4),
                "chunk output x 1.05": lambda block, x, mask, forward: forward(block, x, mask) * 1.05}


@pytest.mark.parametrize("dropout", ["all", "attention"])
def test_row_chunks_keep_the_dropout_distribution(monkeypatch, dropout):
    """Dropout on: the chunked draws differ from the full call's (one draw per chunk), so values cannot match; their
    distribution must. Over DRAW_SEEDS seeds, the per-element means agree (mean squared z < MEAN_Z2) and the variances
    too (ratio in VAR_RATIO), as two unchunked seed sets do (the calibration). Each set has its own seeds. "attention":
    the SDPA draws alone (with every dropout on, the output and FFN dropouts dominate the variance)."""
    raw = encode_draws(range(DRAW_SEEDS), False, monkeypatch, dropout)
    calib = draw_stats(encode_draws(range(1000, 1000 + DRAW_SEEDS), False, monkeypatch, dropout), raw)
    chunked = draw_stats(encode_draws(range(2000, 2000 + DRAW_SEEDS), True, monkeypatch, dropout), raw)
    print(f"\ndropout {dropout}, mean z^2 / variance ratio: unchunked vs unchunked {calib[0]:.3f} / {calib[1]:.4f}; "
          f"chunked vs unchunked {chunked[0]:.3f} / {chunked[1]:.4f}")
    for z2, ratio in (calib, chunked):
        assert z2 < MEAN_Z2 and VAR_RATIO[0] <= ratio <= VAR_RATIO[1], (calib, chunked)


@pytest.mark.parametrize("broken", sorted(CHUNK_BREAKS))
def test_the_dropout_distribution_check_fails_a_broken_chunk(monkeypatch, broken):
    """Attention dropout only (the draws the chunks move): a wrong dropout rate or a scaled chunk must fail."""
    raw = encode_draws(range(DRAW_SEEDS), False, monkeypatch, "attention")
    z2, ratio = draw_stats(encode_draws(range(2000, 2000 + DRAW_SEEDS), True, monkeypatch, "attention",
                                        CHUNK_BREAKS[broken]), raw)
    print(f"\nattention dropout, {broken}: mean z^2 {z2:.3f}, variance ratio {ratio:.4f}")
    assert not (z2 < MEAN_Z2 and VAR_RATIO[0] <= ratio <= VAR_RATIO[1]), (broken, z2, ratio)


def test_a_train_mode_128x2_mps_step_with_attention_dropout_runs(games, monkeypatch):
    """T-064 guard, on torch 2.5.1 and 2.13: the trainer's case on MPS (the shipped chunk rule, every dropout on,
    gradients on) runs forward_heads and backward with finite losses and gradients. A spy proves the full-row
    attention reached SDPA with dropout_p > 0 on MPS tensors that need a gradient, the call that torch 2.13's MPS
    SDPA accepts (under no_grad it raises: MultiHeadAttention.forward)."""
    calls, sdpa = [], torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        calls.append((q.device.type, kwargs.get("dropout_p"), q.requires_grad))
        return sdpa(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    model = chunk_model(MPS, dropout=True)
    torch.manual_seed(1)
    results = model.forward_heads(fixture_batches(games))
    sum(r.loss for r in results.values()).backward()
    print(f"\n128x2 MPS train step, dropout on: {len(calls)} SDPA calls, e.g. {calls[:1]}; losses "
          f"{ {h: round(r.loss.item(), 4) for h, r in results.items()} }")
    assert calls and set(calls) == {("mps", EC.DROPOUT_RATIO, True)}, calls
    assert_finite("loss", ((h, r.loss.detach()) for h, r in results.items()))
    assert_finite("gradient", ((n, p.grad) for n, p in model.named_parameters()))


@pytest.fixture(scope="module")
def chunk_runs(tmp_path_factory, mps_root):
    """128 x 2 MPS positions-mode runs, dropout on, the first block chunked (rows 60 / 400 / 1,600 pad to whole
    chunks): "a" and "c" unbroken, "b" killed after its row-10 write and resumed, "no_rng" resumed without the MPS RNG
    state (the negative control)."""
    arch = {**ARCH, "row0_tol": 0.1}  # the 128-wide row-0 tolerance of the T-056 proof run
    tmp_path = tmp_path_factory.mktemp("mps_128x2")
    with pytest.MonkeyPatch.context() as mp:
        for name in ("a", "c"):
            _run(tmp_path, mp, mps_root, name, save_every=2, device="mps", **arch)
        _kill_and_resume(tmp_path, mp, mps_root, "b", **arch)
        _kill_and_resume(tmp_path, mp, mps_root, "no_rng", (SL, "set_rng_state", _without_mps_rng), **arch)
    return tmp_path


def test_chunked_128x2_mps_resume_is_within_the_run_to_run_noise(chunk_runs):
    ckpt = torch.load(chunk_runs / "a" / "checkpoints" / "step_12.pt", map_location="cpu", weights_only=True)
    assert (ckpt["config"]["device"], ckpt["config"]["embd_size"], ckpt["config"]["n_layers"]) == ("mps", 128, 2)
    for step in (4, 8, 12):
        noise = run_diffs(chunk_runs / "a", chunk_runs / "c", step)
        resumed = run_diffs(chunk_runs / "a", chunk_runs / "b", step)
        print(f"\n128x2 MPS step {step}, unbroken vs unbroken (noise): {noise}\nresumed vs unbroken: {resumed}")
        assert not noise["mismatch"], noise["mismatch"]
        assert within_noise(resumed, noise), (step, resumed, noise)


def test_the_noise_check_fails_a_128x2_resume_without_the_mps_rng(chunk_runs):
    noise, diffs = run_diffs(chunk_runs / "a", chunk_runs / "c"), run_diffs(chunk_runs / "a", chunk_runs / "no_rng")
    print(f"\n128x2 MPS step 12, no_rng resume vs unbroken: {diffs} (noise {noise})")
    assert not within_noise(diffs, noise), (diffs, noise)


def chunk_memory_in_a_fresh_process(chunked: bool, seed: int) -> tuple:
    """(slope, samples) of driver_memory_slope at 128 x 2, chunked as shipped or with the chunks off, in a new Python
    process: MPS keeps compiled shapes and cached memory per process, so the two controls must not share one. Each step
    trains the discard head on SWITCH_ROWS[step % 5] rows: a fixed set of sizes every step (48 / 512 / 160) stayed flat
    without the chunks too; a row count that changes from step to step grew (T-058 replay c4_switch: 2.26 MiB/step)."""
    lines = ["import json", "from MahjongAI.agents import encoder_conv as EC",
             "from MahjongAI.tests import test_mps as T"]
    if not chunked:  # chunked: the shipped rule, untouched
        lines.append("EC.chunks_full_rows = lambda device: False")
    lines.append(f"print(json.dumps(T.driver_memory_slope(T.load_games(), seed={seed}, steps=T.CHUNK_MEMORY_STEPS, "
                 "arch=T.ARCH, rows=lambda step: {'discard': T.SWITCH_ROWS[step % len(T.SWITCH_ROWS)]})))")
    done = subprocess.run([sys.executable, "-c", "\n".join(lines)], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    slope, samples = json.loads(done.stdout.strip().splitlines()[-1])
    return slope, samples


def test_mps_driver_memory_stays_flat_with_row_chunks():
    slope, samples = chunk_memory_in_a_fresh_process(True, seed=2)
    print(f"\n128x2 MPS driver memory, full-row chunks: slope {slope:.3f} MiB/step; {samples}")
    assert slope < MEMORY_SLOPE, (slope, samples)


@pytest.mark.skipif(TORCH_2_13, reason=LEAK_CONTROL_REASON)
def test_mps_driver_memory_grows_without_row_chunks():
    """The negative control: the first block on the step's rows (a size that changes every step) must fail."""
    slope, samples = chunk_memory_in_a_fresh_process(False, seed=3)
    print(f"\n128x2 MPS driver memory, no chunks: slope {slope:.3f} MiB/step; {samples}")
    assert slope >= MEMORY_SLOPE, (slope, samples)


# ---------------------------------------------------------------- the MPS memory log lines (T-056)
TRAIN_SIZES = {"during": 60, "discard": 200, "post": 60}


@pytest.mark.parametrize("stop, samples", [
    (None, [(0, " (post-eval/save)"), (2, ""), (4, ""), (5, " (post-eval/save)"), (5, " (final)")]),
    (3, [(0, " (post-eval/save)"), (2, ""), (3, " (final)")]),  # --stop-after-step: after the stop checkpoint
])
def test_the_mps_memory_line(tmp_path, monkeypatch, train_data_root, stop, samples):
    """The every-MPS_MEMORY_EVERY-steps line, a line after each eval / save (row 0 included) and a final line after
    the last save of the run (T-056)."""
    monkeypatch.setattr(SL, "MPS_MEMORY_EVERY", 2)
    _run(tmp_path, monkeypatch, train_data_root, "m", steps=5, save_every=5, sizes=TRAIN_SIZES, device="mps",
         stop_after_step=stop)
    lines = [ln for ln in open(tmp_path / "m.log") if "mps memory" in ln]
    found = [re.search(r"mps memory at step (\d+)(.*?): driver ", ln) for ln in lines]
    assert [(int(m[1]), m[2]) for m in found] == samples, lines
    assert all(" MB, current " in ln and ln.rstrip().endswith(" MB") for ln in lines)
    assert sorted(step_lines(tmp_path / "m.log")) == list(range(1, (stop or 5) + 1))  # a memory line is no step line
