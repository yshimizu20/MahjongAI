"""The run identity: a run dir resumes only under the same config. torch_threads is part of it (T-011 T1: a changed
TORCH_THREADS default makes old run dirs non-resumable, on purpose).

The T-010 architecture tests start from one of two 1-step runs (64 x 1; 128 x 2 with row0_tol 0.1) built once per
module and copied into each test's tmp_path (run dir and log), on the same data root (T-063): each test's claim is
about its second call on that run dir, and the copy holds the same bytes as a fresh first call."""
import os
import re
import shutil

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.tests.conftest import TEST_LR_CYCLE_GAMES, read_only_run, run_log


def test_config_mismatch_names_the_changed_keys():
    saved = {"model": "encoder_conv", "total_steps": 100, "torch_threads": 10, "lr": 1e-3}
    assert SL.config_mismatch(saved, dict(saved, total_steps=200)) == {}  # total_steps may change on a resume
    assert SL.config_mismatch(saved, dict(saved, torch_threads=4)) == {"torch_threads": (10, 4)}
    assert SL.config_mismatch(saved, {k: v for k, v in saved.items() if k != "lr"}) == {"lr": (1e-3, "<absent>")}
    assert SL.config_mismatch(dict(saved, extra=None), saved) == {"extra": (None, "<absent>")}


def test_resume_with_other_torch_threads_is_rejected(tmp_path, monkeypatch, train_data_root):
    """A 1-step run on fixture games saves a checkpoint under one TORCH_THREADS; starting again on the same run dir
    under another value stops before any work, naming torch_threads."""
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "VAL_GAMES", 1)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = SL.RunLog(str(run_dir / "train.log"))
    threads = torch.get_num_threads()
    try:
        monkeypatch.setattr(SL, "TORCH_THREADS", 2)
        SL.train("encoder_conv", 1, str(run_dir), log, eval_every=1, save_every=1, lr_cycle_games=TEST_LR_CYCLE_GAMES)
        assert os.path.exists(run_dir / "checkpoints" / "step_1.pt")
        monkeypatch.setattr(SL, "TORCH_THREADS", 3)
        with pytest.raises(AssertionError, match=r"another run config: torch_threads 2 -> 3;"):
            SL.train("encoder_conv", 2, str(run_dir), log, eval_every=1, save_every=1,
                     lr_cycle_games=TEST_LR_CYCLE_GAMES)
        monkeypatch.setattr(SL, "TORCH_THREADS", 2)  # the same config resumes (the run extended to 2 steps)
        SL.train("encoder_conv", 2, str(run_dir), log, eval_every=1, save_every=1, lr_cycle_games=TEST_LR_CYCLE_GAMES)
        assert os.path.exists(run_dir / "checkpoints" / "step_2.pt")
    finally:
        torch.set_num_threads(threads)


def test_step_log_line_hashes_the_consumed_files_in_order(tmp_path, monkeypatch, train_data_root):
    """Each per-step log line ends with `files <12 hex>`: batch_files_hash of the files the update consumed, in
    order. A resume check compares these lines to compare the consumed-file sequence."""
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "VAL_GAMES", 1)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    draws = []

    class Recording(SL.DataLoaderCluster):
        def __next__(self):
            out = super().__next__()
            draws.append(list(self.last_batch_files))
            return out

    make = SL.train_cluster

    def recording_cluster(*a, **k):
        cluster = make(*a, **k)
        cluster.__class__ = Recording
        return cluster

    monkeypatch.setattr(SL, "train_cluster", recording_cluster)
    threads = torch.get_num_threads()
    try:
        SL.train("encoder_conv", 3, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "train.log")), 3, 3,
                 lr_cycle_games=TEST_LR_CYCLE_GAMES)
    finally:
        torch.set_num_threads(threads)
    lines = [ln for ln in open(tmp_path / "train.log") if re.search(r"\] step \d+ \| (?!val)", ln)]
    got = [re.search(r" \| files ([0-9a-f]{12})$", ln.rstrip("\n")).group(1) for ln in lines]
    assert len(draws) == len(got) == 3 and all(draws)  # each draw gave one update on the fixture games
    assert got == [SL.batch_files_hash(d) for d in draws]
    assert SL.batch_files_hash(["/a/x.log", "y.log"]) == SL.batch_files_hash(["x.log", "/b/y.log"])
    assert SL.batch_files_hash(["x.log", "y.log"]) != SL.batch_files_hash(["y.log", "x.log"])


# ---------------------------------------------------------------- T-010: the architecture keys of the run config
def test_config_mismatch_reads_a_config_without_architecture_keys_as_64_by_1():
    legacy = {"model": "encoder_conv", "total_steps": 100, "lr": 1e-3}
    arch = lambda e, n: dict(legacy, embd_size=e, n_layers=n, n_params=SL.count_parameters(e, n))
    assert SL.config_mismatch(legacy, arch(64, 1)) == {} and SL.config_mismatch(arch(64, 1), legacy) == {}
    assert SL.config_mismatch(legacy, legacy) == {}
    assert SL.config_mismatch(legacy, arch(128, 2)) == {"embd_size": (64, 128), "n_layers": (1, 2),
                                                        "n_params": (201_311, 693_919)}
    assert SL.config_mismatch(arch(128, 2), arch(128, 1)) == {"n_layers": (2, 1), "n_params": (693_919, 495_647)}
    partial = dict(legacy, embd_size=64)  # a config WITH an architecture key is not filled: it must match exactly
    assert SL.config_mismatch(partial, arch(64, 1)) == {"n_layers": ("<absent>", 1), "n_params": ("<absent>", 201_311)}


def _arch_run(tmp_path, monkeypatch, train_data_root, steps, embd_size=64, n_layers=1, name="run", row0_tol=None):
    monkeypatch.setattr(SL, "DATA_ROOT", train_data_root)
    monkeypatch.setattr(SL, "VAL_GAMES", 1)
    monkeypatch.setattr(SL, "PRIOR_GAMES", 2)
    run_dir = tmp_path / name
    threads = torch.get_num_threads()
    try:
        SL.train("encoder_conv", steps, str(run_dir), SL.RunLog(str(tmp_path / f"{name}.log")), eval_every=1,
                 save_every=1, embd_size=embd_size, n_layers=n_layers, row0_tol=row0_tol,
                 lr_cycle_games=TEST_LR_CYCLE_GAMES)
    finally:
        torch.set_num_threads(threads)
    return run_dir


def _strip_arch(run_dir, step):
    """Rewrite the run dir as a pre-T-010 run's: no architecture keys in the checkpoint's or the header's config."""
    path = run_dir / "checkpoints" / f"step_{step}.pt"
    ckpt = torch.load(path, weights_only=True)
    ckpt["config"] = {k: v for k, v in ckpt["config"].items() if k not in ("embd_size", "n_layers", "n_params")}
    torch.save(ckpt, path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    header["config"] = {k: v for k, v in header["config"].items() if k not in ("embd_size", "n_layers", "n_params")}
    metrics._write([header] + rows)


def _shared_run(root, name, **arch):
    """A 1-step run named `name` next to the module's data root (built once; tests copy it)."""
    with pytest.MonkeyPatch.context() as m:
        return _arch_run(root.parent, m, root, 1, name=name, **arch)


@pytest.fixture(scope="module")
def run_64(module_data_root):
    """The 1-step 64 x 1 run (_arch_run(..., 1)), read only."""
    with read_only_run(_shared_run(module_data_root, "run64")) as run:
        yield run


@pytest.fixture(scope="module")
def run_128(module_data_root):
    """The 1-step 128 x 2 run with row0_tol 0.1 (_arch_run(..., 1, 128, 2, row0_tol=0.1)), read only. The fixture's
    single val game puts the 128 x 2 init model's post CE 0.059 below uniform-legal; the run passes the wider row-0
    limit the way the T-010 production run does (--row0-tol 0.1)."""
    with read_only_run(_shared_run(module_data_root, "run128", embd_size=128, n_layers=2, row0_tol=0.1)) as run:
        yield run


def _copy_run(src, tmp_path, name="run"):
    """tmp_path/<name> and tmp_path/<name>.log: copies of a shared run dir and its log."""
    shutil.copytree(src, tmp_path / name)
    shutil.copy(run_log(src), tmp_path / f"{name}.log")
    return tmp_path / name


def test_header_and_checkpoint_record_the_architecture_and_a_128x2_run_resumes(tmp_path, monkeypatch,
                                                                              module_data_root, run_128):
    run_dir = _copy_run(run_128, tmp_path)
    header, _ = SL.Metrics(str(run_dir / "metrics.jsonl")).read()
    ckpt = torch.load(run_dir / "checkpoints" / "step_1.pt", weights_only=True)
    for config in (header["config"], ckpt["config"]):
        assert (config["embd_size"], config["n_layers"], config["n_params"]) == (128, 2, 693_919)
    assert "model encoder_conv: embd_size 128, n_layers 2, 693,919 parameters" in open(tmp_path / "run.log").read()
    model = SL.EncoderConvModel.from_checkpoint(ckpt)
    assert model.embd_size == 128 and len(model.blocks) == 2
    _arch_run(tmp_path, monkeypatch, module_data_root, 2, 128, 2, row0_tol=0.1)
    assert os.path.exists(run_dir / "checkpoints" / "step_2.pt")


def test_a_run_dir_without_architecture_keys_resumes_as_64x1(tmp_path, monkeypatch, module_data_root, run_64):
    run_dir = _copy_run(run_64, tmp_path)
    _strip_arch(run_dir, 1)
    _arch_run(tmp_path, monkeypatch, module_data_root, 2)
    assert "resuming at step 1;" in open(tmp_path / "run.log").read()
    assert os.path.exists(run_dir / "checkpoints" / "step_2.pt")


@pytest.mark.parametrize("embd_size,n_layers,named", [(128, 1, "embd_size 64 -> 128"), (64, 2, "n_layers 1 -> 2")])
@pytest.mark.parametrize("legacy", [False, True], ids=["arch_keys", "legacy_config"])
def test_resume_under_another_architecture_is_rejected(tmp_path, monkeypatch, module_data_root, run_64, embd_size,
                                                       n_layers, named, legacy):
    run_dir = _copy_run(run_64, tmp_path)
    if legacy:
        _strip_arch(run_dir, 1)
    with pytest.raises(AssertionError, match=rf"step_1.pt is from another run config: .*{named}"):
        _arch_run(tmp_path, monkeypatch, module_data_root, 2, embd_size, n_layers)
    assert not os.path.exists(run_dir / "checkpoints" / "step_2.pt")


def test_header_only_architecture_mismatch_is_rejected(tmp_path, monkeypatch, module_data_root, run_64):
    """The checkpoint matches the flags, the metrics header does not: the header check stops the resume."""
    run_dir = _copy_run(run_64, tmp_path)
    metrics = SL.Metrics(str(run_dir / "metrics.jsonl"))
    header, rows = metrics.read()
    header["config"]["embd_size"] = 128
    metrics._write([header] + rows)
    with pytest.raises(AssertionError, match=r"metrics.jsonl header is from another run: embd_size 128 -> 64"):
        _arch_run(tmp_path, monkeypatch, module_data_root, 2)


def test_row0_check_uses_the_default_limit_unless_row0_tol_widens_it(tmp_path, monkeypatch, module_data_root,
                                                                     run_128):
    # the same 128 x 2 init model (post gap -0.059 on the fixture) stops at row 0 under the default 0.05 limit
    with pytest.raises(AssertionError, match="row 0 is not near uniform-legal"):
        _arch_run(tmp_path, monkeypatch, module_data_root, 1, 128, 2, name="default")
    assert "(|gap| <= 0.05 required)" in open(tmp_path / "default.log").read()
    # the "wide" run: the module's 128 x 2 run with row0_tol 0.1 (the same call), past row 0 to its step-1 checkpoint
    assert "(|gap| <= 0.1 required)" in open(run_log(run_128)).read()
    assert os.path.exists(run_128 / "checkpoints" / "step_1.pt")
