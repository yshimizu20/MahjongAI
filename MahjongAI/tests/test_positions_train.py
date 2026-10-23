"""T-055 G1, trainer level, positions mode on the CPU (fixture games; steps that cross an epoch).

Exact resume: a kill after an eval row and before the next save, and a stop inside an eval row's write, resume to the
unbroken run: metrics rows (with the batcher counters), per-step log lines (losses, read hash, sample digest) and
every checkpoint field (batcher buffers and provenance included).
Failure boundaries: a stop inside a fill (an interrupt after a cluster draw) and a failure after the forward and
before the commit (a forced clip_grad_norm_ error) each save a checkpoint from which the resume ends equal to the
unbroken run; the second one's stop checkpoint equals the unbroken run's checkpoint of that step except the fill
(RNG from before the failed forward; the buffers whole, with the rows the failed step's fill drew).
Identity: batch mode, sizes and device are run-identity keys; a config without them reads as games mode on the CPU."""
import os
import re
import shutil

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.tests.test_resume_rows import _assert_equal

STEPS, EVAL_EVERY, SAVE_EVERY = 12, 5, 4  # rows 0, 5, 10, 12; checkpoints 4, 8, 12
SIZES = {"during": 60, "discard": 1600, "post": 400}  # ~3-4 fixture games a step: 12 steps cross epoch 0


def _run(tmp_path, monkeypatch, root, name, steps=STEPS, save_every=SAVE_EVERY, sizes=SIZES, device="cpu",
         batch_mode="positions"):
    monkeypatch.setattr(SL, "DATA_ROOT", root)
    monkeypatch.setattr(SL, "VAL_GAMES", 2)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    threads = torch.get_num_threads()
    try:
        SL.train("encoder_conv", steps, str(tmp_path / name), SL.RunLog(str(tmp_path / f"{name}.log")),
                 eval_every=EVAL_EVERY, save_every=save_every, batch_mode=batch_mode,
                 position_sizes=sizes if batch_mode == "positions" else None, device=device)
    finally:
        torch.set_num_threads(threads)
    return tmp_path / name


def step_lines(path):
    """{step: the per-step log line without its timestamp}, the last one of each step (a resume re-logs it)."""
    out = {}
    for line in open(path):
        m = re.match(r"\[[^]]+\] (step (\d+) \| (?!val|positions).*)$", line.rstrip("\n"))
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def assert_runs_equal(tmp_path, a, b, checkpoints, ckpt_b=None):
    ha, rows_a = SL.Metrics(str(tmp_path / a / "metrics.jsonl")).read()
    hb, rows_b = SL.Metrics(str(tmp_path / b / "metrics.jsonl")).read()
    assert [r["step"] for r in rows_a] == [r["step"] for r in rows_b] == [0, 5, 10, 12]
    _assert_equal({k: v for k, v in ha.items() if k != "started"}, {k: v for k, v in hb.items() if k != "started"},
                  "header")
    _assert_equal(rows_a, rows_b, "rows")
    lines_a, lines_b = step_lines(tmp_path / f"{a}.log"), step_lines(tmp_path / f"{b}.log")
    assert sorted(lines_a) == list(range(1, STEPS + 1)) and lines_a == lines_b
    for step in checkpoints:
        _assert_equal(torch.load(tmp_path / a / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      torch.load(tmp_path / (ckpt_b or b) / "checkpoints" / f"step_{step}.pt", weights_only=True),
                      f"step_{step}.pt")


@pytest.fixture
def unbroken(tmp_path, monkeypatch, train_data_root):
    """The unbroken positions-mode run "a" (save every 2 steps: a checkpoint for every even step)."""
    _run(tmp_path, monkeypatch, train_data_root, "a", save_every=2)
    return tmp_path / "a"


def test_unbroken_run_records_positions(tmp_path, unbroken):
    header, rows = SL.Metrics(str(unbroken / "metrics.jsonl")).read()
    config = header["config"]
    assert (config["batch_mode"], config["position_sizes"], config["device"], config["batch_size"]) == (
        "positions", SIZES, "cpu", 1)
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", header["git_commit"])
    last = rows[-1]["positions"]
    assert last["games_admitted"] > 40 and set(last["games_by_epoch"]) >= {"0", "1"}  # crossed an epoch
    assert last["taken"]["discard"] == {str(SIZES["discard"]): STEPS}
    for h in SL.HEADS:
        assert sum(last["rows_in"][h].values()) - sum(last["rows_out"][h].values()) == last["buffered"][h]
        assert rows[-1]["heads"][h]["train_n"] == sum(int(k) * n for k, n in last["taken"][h].items()) - sum(
            int(k) * n for k, n in rows[-2]["positions"]["taken"][h].items())
    lines = step_lines(tmp_path / "a.log")
    assert all(re.search(r" \| read [0-9a-f]{12} \| sample [0-9a-f]{12}$", ln) for ln in lines.values())
    ckpt = torch.load(unbroken / "checkpoints" / "step_12.pt", weights_only=True)
    assert ckpt["batcher_state"]["sizes"] == SIZES and "mps_rng_state" not in ckpt


def test_kill_after_an_eval_row_before_the_save_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root,
                                                                             unbroken):
    b = _run(tmp_path, monkeypatch, train_data_root, "b")
    os.remove(b / "checkpoints" / "step_12.pt")
    header, rows = SL.Metrics(str(b / "metrics.jsonl")).read()
    SL.Metrics(str(b / "metrics.jsonl"))._write([header] + [r for r in rows if r["step"] <= 10])
    resumed_from = torch.load(b / "checkpoints" / "step_8.pt", weights_only=True)
    assert resumed_from["batcher_state"]["buffers"]["discard"]["cols"] is not None  # carried rows in the state
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 8;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


def test_stop_inside_an_eval_row_write_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root, unbroken):
    append = SL.Metrics.append

    def stop_at_row_10(self, row):
        if row.get("type") == "eval" and row["step"] == 10:
            raise RuntimeError("stopped inside the row-10 write")
        append(self, row)

    with monkeypatch.context() as m:
        m.setattr(SL.Metrics, "append", stop_at_row_10)
        with pytest.raises(RuntimeError, match="stopped inside the row-10 write"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert "resuming at step 10;" in open(tmp_path / "b.log").read()
    assert_runs_equal(tmp_path, "a", "b", (4, 8, 12))


def test_stop_inside_a_fill_resumes_to_the_unbroken_run(tmp_path, monkeypatch, train_data_root, unbroken):
    """An interrupt right after the 9th cluster draw (inside step 3's fill, its game not admitted yet): the draw is
    undone, the stop checkpoint keeps the admitted games, the resume ends equal to the unbroken run."""
    make = SL.train_cluster

    def stopping_cluster(*a, **k):
        cluster = make(*a, **k)
        cls, calls = type(cluster), [0]

        def next_then_stop(self):
            out = cls.__next__(self)
            calls[0] += 1
            if calls[0] == 9:
                raise KeyboardInterrupt("stop after draw 9")
            return out

        cluster.__class__ = type("Stopping", (cls,), {"__next__": next_then_stop})
        return cluster

    with monkeypatch.context() as m:
        m.setattr(SL, "train_cluster", stopping_cluster)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    stop = re.search(r"STOP at step (\d+): KeyboardInterrupt", open(tmp_path / "b.log").read())
    assert stop and 0 < int(stop.group(1)) < 6, stop
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert_runs_equal(tmp_path, "a", "b", (8, 12))


def test_failure_after_the_forward_before_the_commit_keeps_the_step(tmp_path, monkeypatch, train_data_root, unbroken):
    """Update 7 fails in clip_grad_norm_ (after its forward and backward). The stop checkpoint is step_6.pt: model,
    optimizer, scheduler, RNG (from before the failed forward), train sums and the committed counters equal the
    unbroken run's step_6.pt; the buffers stay whole [R2] (the unbroken step_6 rows, then the rows of the games step
    7's fill drew, in order). The resume ends equal to the unbroken run."""
    clip, calls = torch.nn.utils.clip_grad_norm_, [0]

    def fail_once(*a, **k):
        calls[0] += 1
        if calls[0] == 7:
            raise RuntimeError("forced failure in update 7")
        return clip(*a, **k)

    with monkeypatch.context() as m:
        m.setattr(torch.nn.utils, "clip_grad_norm_", fail_once)
        with pytest.raises(RuntimeError, match="forced failure in update 7"):
            _run(tmp_path, monkeypatch, train_data_root, "b")
    b = tmp_path / "b" / "checkpoints"
    assert sorted(os.listdir(b)) == ["step_4.pt", "step_6.pt"]
    ref = torch.load(unbroken / "checkpoints" / "step_6.pt", weights_only=True)
    stop = torch.load(b / "step_6.pt", weights_only=True)
    filled = ("loader_state", "batcher_state")
    _assert_equal({k: v for k, v in ref.items() if k not in filled}, {k: v for k, v in stop.items() if k not in filled},
                  "stop checkpoint step_6.pt")
    rb, sb = ref["batcher_state"], stop["batcher_state"]
    for key in ("sizes", "rows_out", "taken"):
        _assert_equal(rb[key], sb[key], f"batcher_state.{key}")
    assert sb["games_admitted"] != rb["games_admitted"] and sb["read_since_update"]  # step 7's fill admitted games
    for h in SL.HEADS:
        n, extra = len(rb["buffers"][h]["provenance"]), sum(sb["rows_in"][h].values()) - sum(rb["rows_in"][h].values())
        assert len(sb["buffers"][h]["provenance"]) == n + extra, h
        _assert_equal(rb["buffers"][h]["provenance"], sb["buffers"][h]["provenance"][:n], f"{h} provenance prefix")
        if n:
            _assert_equal(rb["buffers"][h]["cols"], [c[:n] for c in sb["buffers"][h]["cols"]], f"{h} rows prefix")
    _run(tmp_path, monkeypatch, train_data_root, "b")
    assert_runs_equal(tmp_path, "a", "b", (8, 12))


# ---------------------------------------------------------------- run identity
def test_config_mismatch_reads_a_config_without_t055_keys_as_games_on_the_cpu():
    legacy = {"model": "encoder_conv", "total_steps": 100}
    games = dict(legacy, batch_mode="games", position_sizes=None, device="cpu")
    assert SL.config_mismatch(legacy, games) == {} and SL.config_mismatch(games, legacy) == {}
    assert SL.config_mismatch(legacy, dict(games, batch_mode="positions", position_sizes=SIZES)) == {
        "batch_mode": ("games", "positions"), "position_sizes": (None, SIZES)}
    assert SL.config_mismatch(games, dict(games, device="mps")) == {"device": ("cpu", "mps")}


def _strip_t055(run_dir, step):
    path = run_dir / "checkpoints" / f"step_{step}.pt"
    ckpt = torch.load(path, weights_only=True)
    ckpt["config"] = {k: v for k, v in ckpt["config"].items() if k not in SL.LEGACY_RUN}
    torch.save(ckpt, path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    header["config"] = {k: v for k, v in header["config"].items() if k not in SL.LEGACY_RUN}
    metrics._write([header] + rows)


def test_a_legacy_games_run_resumes_and_other_settings_are_rejected(tmp_path, monkeypatch, train_data_root):
    run = _run(tmp_path, monkeypatch, train_data_root, "g", steps=1, batch_mode="games")
    _strip_t055(run, 1)  # a run dir written before T-055
    with pytest.raises(AssertionError, match=r"step_1.pt is from another run config: batch_mode 'games' -> "
                                             r"'positions', batch_size 2 -> 1, position_sizes None -> "):
        _run(tmp_path, monkeypatch, train_data_root, "g", steps=2)
    with pytest.raises(AssertionError, match=r"step_1.pt is from another run config: device 'cpu' -> 'mps'"):
        _run(tmp_path, monkeypatch, train_data_root, "g", steps=2, batch_mode="games", device="mps")
    _run(tmp_path, monkeypatch, train_data_root, "g", steps=2, batch_mode="games")
    assert "resuming at step 1;" in open(tmp_path / "g.log").read()


def test_a_positions_run_rejects_other_sizes(tmp_path, monkeypatch, train_data_root):
    _run(tmp_path, monkeypatch, train_data_root, "p", steps=1)
    with pytest.raises(AssertionError, match=r"another run config: position_sizes .*'post': 400.* -> .*'post': 300"):
        _run(tmp_path, monkeypatch, train_data_root, "p", steps=2, sizes=dict(SIZES, post=300))
    with pytest.raises(AssertionError, match="position sizes are a positions-mode setting"):
        SL.train("encoder_conv", 1, str(tmp_path / "x"), SL.RunLog(str(tmp_path / "x.log")), position_sizes=SIZES)
