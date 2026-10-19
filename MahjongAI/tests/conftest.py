"""Data roots built from the committed fixture games (tests/data/gate_games), linked file by file into tmp_path: the
loader's os.walk does not enter linked directories."""
import pathlib

import pytest

GATE_GAMES = pathlib.Path(__file__).resolve().parent / "data" / "gate_games"
VAL_FROM = "20210103"  # the 2021 fixture day served as validation day 2022-01-01 (every 2022 fixture game is a skip)


def link_games(files, root: pathlib.Path, day_map=None) -> pathlib.Path:
    """Link each <day>/<day>_<nn>/<game>.log under root, keeping the loader's layout; day_map renames days."""
    for f in files:
        f = pathlib.Path(f)
        day, sub = f.parent.parent.name, f.parent.name
        if day_map and day in day_map:
            day, sub = day_map[day], day_map[day] + sub[len(day):]
        dest = root / day / sub
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f.name).symlink_to(f)
    return root


@pytest.fixture
def train_data_root(tmp_path) -> pathlib.Path:
    """A DATA_ROOT for supervised_learning.train: 2021 = every 2021 fixture game but day VAL_FROM's, and
    2022/20220101 = VAL_FROM's games."""
    root = tmp_path / "data"
    games = sorted(GATE_GAMES.glob("2021/*/*/*.log"))
    link_games([f for f in games if f.parent.parent.name != VAL_FROM], root / "2021")
    link_games([f for f in games if f.parent.parent.name == VAL_FROM], root / "2022", {VAL_FROM: "20220101"})
    return root


@pytest.fixture
def small_roots(tmp_path):
    """Two loader roots of the 15 smallest 2021 fixture games (11 accepted; 3 'Auto-played seat', 1 'BYE'), split
    by day: before / from April."""
    games = sorted(GATE_GAMES.glob("2021/*/*/*.log"), key=lambda f: (f.stat().st_size, f.name))[:15]
    a = link_games([f for f in games if f.parent.parent.name < "20210401"], tmp_path / "A")
    b = link_games([f for f in games if f.parent.parent.name >= "20210401"], tmp_path / "B")
    return a, b


def make_wide_val_root(root: pathlib.Path) -> pathlib.Path:
    """A DATA_ROOT whose validation day has many games (T-047 E1 / S1): 2022/20220101 = the 25 games of 2021-01-01
    (some of them expected skips), 2021 = every other 2021 fixture game."""
    games = sorted(GATE_GAMES.glob("2021/*/*/*.log"))
    link_games([f for f in games if f.parent.parent.name != "20210101"], root / "2021")
    link_games([f for f in games if f.parent.parent.name == "20210101"], root / "2022", {"20210101": "20220101"})
    return root


@pytest.fixture
def wide_val_root(tmp_path) -> pathlib.Path:
    return make_wide_val_root(tmp_path / "data")


def pytest_sessionfinish(session, exitstatus):
    """A skipped test fails the run (T-050 G4: every test, the compile tests included, is mandatory; the CI runs
    this suite): exit status 1 and the count, unless the run already failed."""
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = len(reporter.stats.get("skipped", [])) if reporter is not None else 0
    if skipped and session.exitstatus == 0:
        reporter.write_line(f"FAIL: {skipped} test(s) skipped; every test is mandatory (conftest.py)", red=True)
        session.exitstatus = 1
