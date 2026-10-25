"""Data roots built from the committed fixture games (tests/data/gate_games), linked file by file into tmp_path: the
loader's os.walk does not enter linked directories.

Under pytest-xdist (CI: -n 4, T-063) each worker process sets its torch default to 1 thread (tests that need a count
set it and restore it; SL.train sets its own), runs the gates self-test first (the longest test: it starts at t=0),
and a pool of >= 8 data workers waits for big_pool_slot (one such pool at a time across workers). A serial run keeps
its order and thread default."""
import contextlib
import hashlib
import pathlib

import pytest

SELF_TEST = "test_gates.py::test_gates_self_test"
BIG_POOL = 8  # a pool of this many data workers or more takes big_pool_slot


def _xdist_worker(config) -> bool:
    return hasattr(config, "workerinput")


def pytest_configure(config):
    if _xdist_worker(config):
        import torch
        torch.set_num_threads(1)  # 4 workers x the default (cores) threads would oversubscribe the runner


def pytest_collection_modifyitems(config, items):
    """Under xdist, the self-test first (every worker applies the same order, as xdist requires)."""
    if _xdist_worker(config):
        items.sort(key=lambda item: not item.nodeid.endswith(SELF_TEST))  # stable: the rest keep their order


@pytest.fixture(scope="session")
def big_pool_slot(tmp_path_factory, request):
    """slot(n): a context that, for n >= BIG_POOL, holds a file lock shared by the xdist workers (in the parent of
    their basetemps), so at most one pool of >= 8 torch-importing children runs at a time; a serial run locks a file
    of its own basetemp (never waits). filelock is a torch dependency."""
    from filelock import FileLock
    base = tmp_path_factory.getbasetemp()
    lock = FileLock(str((base.parent if _xdist_worker(request.config) else base) / "big_pool.lock"))

    def slot(n):
        return lock if n >= BIG_POOL else contextlib.nullcontext()
    return slot

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


def build_train_data_root(root: pathlib.Path) -> pathlib.Path:
    """A DATA_ROOT for supervised_learning.train: 2021 = every 2021 fixture game but day VAL_FROM's, and
    2022/20220101 = VAL_FROM's games."""
    games = sorted(GATE_GAMES.glob("2021/*/*/*.log"))
    link_games([f for f in games if f.parent.parent.name != VAL_FROM], root / "2021")
    link_games([f for f in games if f.parent.parent.name == VAL_FROM], root / "2022", {VAL_FROM: "20220101"})
    return root


@pytest.fixture
def train_data_root(tmp_path) -> pathlib.Path:
    return build_train_data_root(tmp_path / "data")


@pytest.fixture(scope="module")
def module_data_root(tmp_path_factory) -> pathlib.Path:
    """One train data root per test module (per xdist worker) for the runs a module shares (T-063): a shared run and
    every run compared with or resumed from it read the same root, as they did inside one test's tmp_path."""
    return build_train_data_root(tmp_path_factory.mktemp("module") / "data")


def run_log(run_dir: pathlib.Path) -> pathlib.Path:
    """The log a test's trainer run writes next to its run dir: <parent>/<name>.log."""
    return run_dir.parent / f"{run_dir.name}.log"


def tree_digest(*paths: pathlib.Path) -> dict:
    """{path relative to its parent: sha256} of every file under the given dirs / files."""
    out = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(f for f in path.rglob("*") if f.is_file())
        for f in files:
            out[str(f.relative_to(path.parent))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


@contextlib.contextmanager
def read_only_run(run_dir: pathlib.Path):
    """Yield run_dir (built once, shared by a module's tests); on exit assert no file of it or of its log changed: a
    test that wrote into the shared run would corrupt every later comparison (T-063)."""
    before = tree_digest(run_dir, run_log(run_dir))
    yield run_dir
    after = tree_digest(run_dir, run_log(run_dir))
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"a test changed the shared run {run_dir}: {', '.join(changed)}"


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
