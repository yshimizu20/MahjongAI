"""T-055 G1 [R7], on this machine's GPU (MPS). These tests must run: a machine without MPS fails them, never skips.

Parity: dropout off, the same weights and tensors (two fixture games, every head): per-head loss within 1e-4, every
gradient within 1e-4 relative (per parameter: max |diff| / max |CPU grad|, the scale floored at 1e-6 x the largest
CPU gradient) and the parameters after one AdamW update (the trainer's lr 1e-3) within 1e-4; a perturbed weight
(ln_f x 1.5) must fail it. The floor is for the attention key bias: softmax ignores a shift of a query's scores, so
its gradient is zero by math and both devices give rounding noise (~1e-11) whose ratio means nothing.
Eval: the val rows of an MPS run (computed on the trainer's CPU copy) equal a CPU model's val of the same weights.
Resume [R6]: an MPS positions-mode run with dropout on, killed after an eval row, resumes to the unbroken MPS run:
rows, step lines and every checkpoint field (MPS RNG state included). A miss reports the largest weight difference.
In torch 2.5.1 it misses: nn.Embedding backward on MPS is not deterministic (two identical runs differ by ~1e-8
after one step), so this test fails until that is fixed; the data path of the resume is checked exactly apart.
Load [R14]: an MPS checkpoint loads on the CPU in the viewer (infer.load_model) and the self-play player
(ModelPlayer); their logits equal the CPU model of the same weights."""
import glob
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
from MahjongAI.tests.test_positions_train import _run, assert_runs_equal, step_lines
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


def _step(model, batches):
    """Per-head loss, gradients and the parameters after one AdamW update (lr 1e-3), all on the CPU."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    results = model.forward_heads(batches)
    sum(r.loss for r in results.values()).backward()
    losses = {h: r.loss.item() for h, r in results.items()}
    grads = [p.grad.cpu().clone() for p in model.parameters()]
    optimizer.step()
    return losses, grads, [p.detach().cpu().clone() for p in model.parameters()]


def parity_diffs(games, perturb=False) -> dict:
    """The largest MPS - CPU differences: per-head loss (abs), gradients (relative), updated parameters (abs)."""
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
    loss_c, grads_c, params_c = _step(cpu, batches)
    loss_g, grads_g, params_g = _step(gpu, batches)
    assert next(gpu.parameters()).device.type == "mps"
    floor = 1e-6 * max(g.abs().max().item() for g in grads_c)
    return {"loss": max(abs(loss_c[h] - loss_g[h]) for h in SL.HEADS),
            "grad_rel": max((a - b).abs().max().item() / max(a.abs().max().item(), floor)
                            for a, b in zip(grads_c, grads_g)),
            "param": max((a - b).abs().max().item() for a, b in zip(params_c, params_g))}


def test_dropout_off_parity_of_loss_gradients_and_one_update(games):
    diffs = parity_diffs(games)
    print(f"\nMPS vs CPU parity (dropout off): {diffs}")
    assert all(v <= TOL for v in diffs.values()), diffs


def test_parity_fails_on_a_perturbed_weight(games):
    diffs = parity_diffs(games, perturb=True)
    print(f"\nMPS (ln_f x 1.5) vs CPU: {diffs}")
    assert diffs["loss"] > TOL and diffs["grad_rel"] > TOL, diffs


# ---------------------------------------------------------------- runs on MPS
@pytest.fixture(scope="module")
def mps_root(tmp_path_factory):
    return build_train_data_root(tmp_path_factory.mktemp("mps_data") / "data")


@pytest.fixture(scope="module")
def mps_runs(tmp_path_factory, mps_root):
    """An unbroken MPS positions-mode run "a" (dropout on; a checkpoint every 2 steps) and a run "b" killed after
    its row-10 write (step_12.pt and row 12 removed), then resumed."""
    tmp_path = tmp_path_factory.mktemp("mps")
    with pytest.MonkeyPatch.context() as mp:
        _run(tmp_path, mp, mps_root, "a", save_every=2, device="mps")
        b = _run(tmp_path, mp, mps_root, "b", device="mps")
        os.remove(b / "checkpoints" / "step_12.pt")
        header, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
        SL.Metrics(str(b / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
        _run(tmp_path, mp, mps_root, "b", device="mps")
    return tmp_path


def _max_weight_diff(a, b):
    sa = torch.load(a, map_location="cpu", weights_only=True)["model_state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=True)["model_state_dict"]
    return max((sa[k].double() - sb[k].double()).abs().max().item() for k in sa)


def test_mps_resume_equals_the_unbroken_mps_run(mps_runs):
    ckpt = torch.load(mps_runs / "a" / "checkpoints" / "step_12.pt", map_location="cpu", weights_only=True)
    assert ckpt["config"]["device"] == "mps" and "mps_rng_state" in ckpt
    assert "resuming at step 8;" in open(mps_runs / "b.log").read()
    diff = _max_weight_diff(mps_runs / "a" / "checkpoints" / "step_12.pt", mps_runs / "b" / "checkpoints" / "step_12.pt")
    print(f"\nMPS resume vs unbroken MPS run, step_12 max |weight diff|: {diff}")
    assert_runs_equal(mps_runs, "a", "b", (4, 8, 12))


def test_mps_resume_repeats_the_data_path_of_the_unbroken_mps_run(mps_runs):
    """The device-free part of the resume: every step reads the same files and trains on the same rows (read hash,
    sample digest), the loader and batcher states of every checkpoint are equal, and the batcher counters of every
    row. This holds even where the MPS weights drift (nn.Embedding backward on MPS is not deterministic)."""
    tail = re.compile(r" \| read [0-9a-f]{12} \| sample [0-9a-f]{12}$")
    lines_a, lines_b = step_lines(mps_runs / "a.log"), step_lines(mps_runs / "b.log")
    assert sorted(lines_a) == sorted(lines_b) == list(range(1, 13))
    assert {s: tail.search(ln).group() for s, ln in lines_a.items()} == {s: tail.search(ln).group()
                                                                         for s, ln in lines_b.items()}
    for step in (4, 8, 12):
        a, b = (torch.load(mps_runs / r / "checkpoints" / f"step_{step}.pt", map_location="cpu", weights_only=True)
                for r in ("a", "b"))
        for key in ("current_step", "loader_state", "batcher_state", "config", "val_ids"):
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
