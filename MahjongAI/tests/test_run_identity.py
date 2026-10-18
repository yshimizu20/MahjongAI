"""The run identity: a run dir resumes only under the same config. torch_threads is part of it (T-011 T1: a changed
TORCH_THREADS default makes old run dirs non-resumable, on purpose)."""
import os
import pathlib

import pytest
import torch

from MahjongAI import supervised_learning as SL

GATE_GAMES = pathlib.Path(__file__).resolve().parent / "data" / "gate_games"


def test_config_mismatch_names_the_changed_keys():
    saved = {"model": "encoder_conv", "total_steps": 100, "torch_threads": 10, "lr": 1e-3}
    assert SL.config_mismatch(saved, dict(saved, total_steps=200)) == {}  # total_steps may change on a resume
    assert SL.config_mismatch(saved, dict(saved, torch_threads=4)) == {"torch_threads": (10, 4)}
    assert SL.config_mismatch(saved, {k: v for k, v in saved.items() if k != "lr"}) == {"lr": (1e-3, "<absent>")}
    assert SL.config_mismatch(dict(saved, extra=None), saved) == {"extra": (None, "<absent>")}


def _data_root(root: pathlib.Path) -> pathlib.Path:
    """Fixture games as a data root: every 2021 fixture game but the day VAL_FROM for training, and VAL_FROM's games
    as validation day 2022-01-01 (every 2022 fixture game is an expected skip). Linked file by file: the loader's
    os.walk does not enter linked directories."""
    for f in sorted(GATE_GAMES.glob("2021/*/*/*.log")):
        day, sub = f.parent.parent.name, f.parent.name
        dest = root / "2022" / "20220101" / "20220101_00" if day == VAL_FROM else root / "2021" / day / sub
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f.name).symlink_to(f)
    return root


VAL_FROM = "20210103"


def test_resume_with_other_torch_threads_is_rejected(tmp_path, monkeypatch):
    """A 1-step run on fixture games saves a checkpoint under one TORCH_THREADS; starting again on the same run dir
    under another value stops before any work, naming torch_threads."""
    monkeypatch.setattr(SL, "DATA_ROOT", _data_root(tmp_path / "data"))
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
