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
(ModelPlayer); their logits equal the CPU model of the same weights."""
import glob
import json
import math
import os
import re
import shutil

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
# NOISE_TIMES x the unbroken-vs-unbroken difference at the same step (plus an absolute floor) for the values MPS
# training moves (weights, Adam moments, train loss sums, val CE); every other checkpoint value is exact.
# The floors are each quantity's float32 rounding scale (a draw where the two unbroken runs agree exactly must not
# fail a resume). Measured noise at step 12: weight 1.5e-6, exp_avg 9.3e-9, exp_avg_sq 2.9e-11, train loss sums 0
# (steps 4 and 8: sums of ~1e3; one float32 ulp of a batch loss x its rows is ~1e-4).
NOISE_TIMES, CE_FLOOR = 10, 1e-5
FLOORS = {"weight": 1e-6, "exp_avg": 1e-7, "exp_avg_sq": 1e-10, "train_loss": 1e-3}


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
