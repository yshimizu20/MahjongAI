"""The run identity: a run dir resumes only under the same config. torch_threads is part of it (T-011 T1: a changed
TORCH_THREADS default makes old run dirs non-resumable, on purpose)."""
import os
import re

import pytest
import torch

from MahjongAI import supervised_learning as SL


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
        SL.train("encoder_conv", 1, str(run_dir), log, eval_every=1, save_every=1)
        assert os.path.exists(run_dir / "checkpoints" / "step_1.pt")
        monkeypatch.setattr(SL, "TORCH_THREADS", 3)
        with pytest.raises(AssertionError, match=r"another run config: torch_threads 2 -> 3;"):
            SL.train("encoder_conv", 2, str(run_dir), log, eval_every=1, save_every=1)
        monkeypatch.setattr(SL, "TORCH_THREADS", 2)  # the same config resumes (the run extended to 2 steps)
        SL.train("encoder_conv", 2, str(run_dir), log, eval_every=1, save_every=1)
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
        SL.train("encoder_conv", 3, str(tmp_path / "run"), SL.RunLog(str(tmp_path / "train.log")), 3, 3)
    finally:
        torch.set_num_threads(threads)
    lines = [ln for ln in open(tmp_path / "train.log") if re.search(r"\] step \d+ \| (?!val)", ln)]
    got = [re.search(r" \| files ([0-9a-f]{12})$", ln.rstrip("\n")).group(1) for ln in lines]
    assert len(draws) == len(got) == 3 and all(draws)  # each draw gave one update on the fixture games
    assert got == [SL.batch_files_hash(d) for d in draws]
    assert SL.batch_files_hash(["/a/x.log", "y.log"]) == SL.batch_files_hash(["x.log", "/b/y.log"])
    assert SL.batch_files_hash(["x.log", "y.log"]) != SL.batch_files_hash(["y.log", "x.log"])
