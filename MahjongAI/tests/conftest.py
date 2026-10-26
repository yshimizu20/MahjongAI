"""Data roots built from the committed fixture games (tests/data/gate_games), linked file by file into tmp_path: the
loader's os.walk does not enter linked directories.

Under pytest-xdist (CI: -n 4, T-063) each worker process sets its torch default to 1 thread (tests that need a count
set it and restore it; SL.train sets its own), each part of the gates self-test (the longest test) starts a worker's
first chunk at t=0, and a pool of >= 8 data workers waits for big_pool_slot (one such pool at a time across workers).
A serial run keeps its order and thread default.

Pool sizes of the test session (T-065, serial and xdist alike): every trainer run builds its validation set and prior
in the trainer (PRIOR_WORKERS 0) and trains with 2 data workers (DATA_WORKERS 2). Each spawned worker imports torch,
and CI is CPU-bound: the spawns were 71 % of the suite's CPU. Neither size is a run-identity key or changes a batch,
row or checkpoint (test_startup_pool, test_prior_workers, test_worker_count, test_worker_sweep prove it, with sizes of
their own). A test that claims the production sizes takes `production_pools`."""
import contextlib
import hashlib
import pathlib

import pytest

SELF_TEST = "test_gates.py::test_gates_self_test"  # its parts: SELF_TEST[0], SELF_TEST[1], ...
BIG_POOL = 8  # a pool of this many data workers or more takes big_pool_slot
TEST_POOLS = {"PRIOR_WORKERS": 0, "DATA_WORKERS": 2}  # the test session's supervised_learning pool sizes (T-065)


def _xdist_worker(config) -> bool:
    return hasattr(config, "workerinput")


def pytest_configure(config):
    if _xdist_worker(config):
        import torch
        torch.set_num_threads(1)  # 4 workers x the default (cores) threads would oversubscribe the runner


def shared_tmp(tmp_path_factory, config) -> pathlib.Path:
    """A folder of this pytest run that every xdist worker sees (the parent of the workers' basetemps); a serial run's
    own basetemp. Both are new for each run."""
    base = tmp_path_factory.getbasetemp()
    return base.parent if _xdist_worker(config) else base


def worksteal_starts(n_items: int, n_workers: int) -> list:
    """The index of each worker's first item in xdist worksteal's first distribution: idle workers get the pending
    items in contiguous chunks of len(pending) // workers_left (xdist 3.x WorkStealingScheduling.check_schedule)."""
    starts, start, left = [], 0, n_items
    for i in range(n_workers):
        starts.append(start)
        take = left // (n_workers - i)
        start, left = start + take, left - take
    return starts


@pytest.hookimpl(trylast=True)  # after the -m / -k deselection: the positions index the items xdist will schedule
def pytest_collection_modifyitems(config, items):
    """Under xdist, part k of the self-test is the first item of first chunk k % workers (after the earlier parts of
    that chunk), the rest keep their order (every worker applies the same order, as xdist requires)."""
    if not _xdist_worker(config):
        return
    parts = [item for item in items if item.nodeid.split("[")[0].endswith(SELF_TEST)]
    rest = [item for item in items if not item.nodeid.split("[")[0].endswith(SELF_TEST)]
    n = config.workerinput["workercount"]
    starts = worksteal_starts(len(items), n)
    at = {starts[k % n] + k // n: item for k, item in enumerate(parts)}
    fill = iter(rest)
    items[:] = [at[i] if i in at else next(fill) for i in range(len(items))]


@pytest.fixture(scope="session", autouse=True)
def session_pool_sizes():
    """TEST_POOLS on supervised_learning for the whole session, set before every module fixture (session scope); a
    test's own monkeypatch overrides it for that test. Yields the production sizes it replaced. (Read here, not at
    import: pytest imports this file as `conftest` and the tests import it as `MahjongAI.tests.conftest`, two
    modules.)"""
    from MahjongAI import supervised_learning as SL
    production = {k: getattr(SL, k) for k in TEST_POOLS}
    with pytest.MonkeyPatch.context() as m:
        for k, v in TEST_POOLS.items():
            m.setattr(SL, k, v)
        yield production


@pytest.fixture
def production_pools(monkeypatch, session_pool_sizes):
    """The production pool sizes for one test (a module fixture uses session_pool_sizes itself)."""
    from MahjongAI import supervised_learning as SL
    for k, v in session_pool_sizes.items():
        monkeypatch.setattr(SL, k, v)
    return session_pool_sizes


@pytest.fixture(scope="session")
def big_pool_slot(tmp_path_factory, request):
    """slot(n): a context that, for n >= BIG_POOL, holds a file lock shared by the xdist workers (in the parent of
    their basetemps), so at most one pool of >= 8 torch-importing children runs at a time; a serial run locks a file
    of its own basetemp (never waits). filelock is a torch dependency."""
    from filelock import FileLock
    lock = FileLock(str(shared_tmp(tmp_path_factory, request.config) / "big_pool.lock"))

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
