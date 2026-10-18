import hashlib
import os
import random
from collections import Counter

from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.utils.game_workers import RemoteError


class UnexpectedSkip(RuntimeError):
    """A game skipped for a reason outside the loader's `expected_skips`. Raised after the loader has moved past
    the game, so a run that saves its state and resumes does not meet it again."""


def epoch_order(files: list, seed, epoch: int) -> list:
    """The read order of `files` (pass them sorted) in epoch `epoch`: shuffled by random.Random(seed) in epoch 0
    (the pre-T-045 order) and by random.Random(f"{seed}/epoch{epoch}") from epoch 1 on. A str seed is hashed with
    SHA-512, so the order is the same in every process. seed=None keeps the sorted, date-ordered list."""
    files = list(files)
    if seed is not None:
        random.Random(seed if epoch == 0 else f"{seed}/epoch{epoch}").shuffle(files)
    return files


def expected_accepted_left(left: int, accepted: int, skipped: int) -> float:
    """A loader's expected accepted games left: its files left times its run acceptance rate, smoothed so a loader
    that has read nothing yet counts at 0.5. The cluster draws each batch's loader with this weight."""
    return left * (accepted + 1) / (accepted + skipped + 2)


def draw_index(weights: list, seed, k: int) -> int:
    """Draw k: an index drawn with probability weights[i] / sum(weights), from random.Random(f"{seed}/draw{k}").
    Stateless, so a resume continues the sequence from k alone. A zero weight is never drawn."""
    total = sum(weights)
    assert total > 0, f"no positive weight in {weights}"
    r = random.Random(f"{seed}/draw{k}").random() * total
    cumulative, last = 0.0, None
    for i, w in enumerate(weights):
        if w > 0:
            cumulative += w
            last = i
            if r < cumulative:
                return i
    return last  # r rounded up to the total


class DataLoader:
    """Yields batches of parsed games from every `*.log` under `path`.

    Layout assumed: <path>/<day>/<day>_<nn>/<game>.log, so a game's day is its second parent directory.
    days / exclude_days: keep only, or drop, the games of those day directories (the train/val split).
    seed: the read order of each epoch, `epoch_order` (None keeps the sorted, date-ordered list).
    file_list: the current epoch's read order (`file_list[current_index - 1]` is the last file read).
    epoch: epochs started before this one; `next_epoch()` starts the next one with a new order of the same files.
    accepted: games returned in batches, whole run (with `skipped`, the cluster's estimate of the acceptance rate).
    Games that fail to parse are counted in `skipped` (key "invalid:<reason>" or the exception's type name).
    expected_skips: None counts every skip and continues; a set of keys raises UnexpectedSkip on any other key.
    logged_skips: keys whose skipped file is also reported through `log_func` (a parse error always is).
    last_batch_files: the files of the games in the batch last returned, in batch order.
    pending: accepted files of a batch cut short by UnexpectedSkip; the next batch reads them first (kept in the state),
    so a stop loses no good game while the position stays past the offending one.
    workers: a GameWorkers (T-011 D1) parses and builds the games in worker processes; a game is then a PrebuiltGame
    in the halfturns slot (tokens slot None). The loader submits its next files ahead and stays the only authority on
    order, batches, skips and state. A fatal worker error (or a stop) while fetching rolls the loader back to its
    state before the batch, so a checkpoint saved on that stop resumes exactly where an unstopped run would go on.
    """

    def __init__(self, path: str, batch_size: int = 1, log_func: callable = print,
                 days=None, exclude_days=None, seed=None, expected_skips=None, logged_skips=(), workers=None):
        self.path = path
        self.workers = workers
        self.file_list = sorted(
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(self.path)
            for fname in fnames
            if fname.endswith(".log")
        )
        if days is not None:
            keep = set(days)
            self.file_list = [f for f in self.file_list if self.day_of(f) in keep]
        if exclude_days is not None:
            drop = set(exclude_days)
            self.file_list = [f for f in self.file_list if self.day_of(f) not in drop]
        self.file_list = epoch_order(self.file_list, seed, 0)
        self.seed = seed
        self.epoch = 0
        self.accepted = 0
        self.current_index = 0
        self.batch_size = batch_size
        self.batch_count = 0
        self.log_func = log_func
        self.skipped = Counter()
        self.expected_skips = None if expected_skips is None else frozenset(expected_skips)
        self.logged_skips = frozenset(logged_skips)
        self.last_batch_files = []
        self.pending = []

    @staticmethod
    def day_of(file_path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(file_path)))

    def __iter__(self):
        return self

    def _snapshot(self):
        """Everything a batch changes (file_list is replaced, never mutated, by next_epoch)."""
        return (self.current_index, list(self.pending), self.skipped.copy(), self.accepted, self.epoch, self.file_list,
                self.batch_count, self.last_batch_files)

    def _restore(self, snapshot):
        (self.current_index, self.pending, self.skipped, self.accepted, self.epoch, self.file_list,
         self.batch_count, self.last_batch_files) = snapshot
        self.pending = list(self.pending)
        self.skipped = self.skipped.copy()

    def __next__(self):
        snapshot = self._snapshot() if self.workers is not None else None
        self.batch_count += 1

        if self.current_index >= len(self.file_list) and not self.pending:
            raise StopIteration

        all_halfturns_list, all_encoding_tokens_list, files = [], [], []
        queue, self.pending = self.pending, []

        while (queue or self.current_index < len(self.file_list)) and len(all_halfturns_list) < self.batch_size:
            if queue:
                filename = queue.pop(0)
            else:
                filename = self.file_list[self.current_index]
                self.current_index += 1

            if self.workers is not None:
                try:  # outside the skip handler: a dead worker is no skip
                    self.workers.prefetch(queue + self.file_list[self.current_index:self.current_index + self.workers.window])
                    fetched = self.workers.fetch(filename)
                except BaseException:
                    self._restore(snapshot)  # a dead worker or a stop: as if this batch was never asked for
                    raise
            try:
                all_halfturns, all_encoding_tokens = process(filename) if self.workers is None else fetched.game()
            except Exception as e:
                remote = isinstance(e, RemoteError)  # a worker's exception that did not pickle, as a record
                if isinstance(e, InvalidGameException) or (remote and e.invalid_game):
                    key = "invalid:" + e.msg
                else:  # a parser bug on one game: report it, keep going unless told otherwise
                    key = e.type_name if remote else type(e).__name__
                    self.log_func(f"skipped {filename}: {key}: {str(e)[:200]}")
                try:
                    self._skip(filename, key, e)
                except UnexpectedSkip:
                    self.pending = files + queue  # the good games already read in this batch are not lost
                    raise
                continue

            all_halfturns_list.append(all_halfturns)
            all_encoding_tokens_list.append(all_encoding_tokens)
            files.append(filename)

        self.last_batch_files = files
        self.accepted += len(files)
        if self.workers is not None:  # the next batch's files are parsed while the trainer runs this one
            self.workers.prefetch(self.pending + self.file_list[self.current_index:self.current_index + self.workers.window])
        return all_halfturns_list, all_encoding_tokens_list

    def _skip(self, filename: str, key: str, error: Exception):
        self.skipped[key] += 1
        if key in self.logged_skips:
            self.log_func(f"skipped {filename}: {key}")
        if self.expected_skips is not None and key not in self.expected_skips:
            raise UnexpectedSkip(f"unexpected skip {key!r} of {filename}: {str(error)[:200]}") from error

    def left(self) -> int:
        """Files not read yet in this epoch, plus the pending good games of a cut batch."""
        return len(self.file_list) - self.current_index + len(self.pending)

    def next_epoch(self):
        """Starts the next epoch: the same files (re-sorted, no new walk of the disk) in that epoch's order."""
        assert not self.pending, f"next_epoch() with pending games {self.pending}: they would be lost"
        self.epoch += 1
        self.current_index = 0
        self.file_list = epoch_order(sorted(self.file_list), self.seed, self.epoch)

    def order_sha256(self) -> str:
        """Hash of the current read order (paths relative to `path`)."""
        h = hashlib.sha256()
        for f in self.file_list:
            h.update(os.path.relpath(f, self.path).encode() + b"\n")
        return h.hexdigest()

    def state_dict(self) -> dict:
        return {"current_index": self.current_index, "seed": self.seed, "n_files": len(self.file_list),
                "skipped": dict(self.skipped), "pending": [os.path.relpath(f, self.path) for f in self.pending],
                "epoch": self.epoch, "accepted": self.accepted, "order_sha256": self.order_sha256()}

    def load_state_dict(self, state: dict):
        """Restores a state_dict() onto a loader built with the same arguments. Epoch 0 keeps this loader's order
        (a caller may have set file_list); a later epoch rebuilds that epoch's order. A state written before T-045
        (no "epoch" / "accepted" / "order_sha256") loads as epoch 0, accepted 0, with no order check."""
        assert state["seed"] == self.seed and state["n_files"] == len(self.file_list), (
            f"checkpoint loader state {state} does not match this loader (seed={self.seed}, n_files={len(self.file_list)})"
        )
        epoch = state.get("epoch", 0)
        if epoch >= 1:
            self.file_list = epoch_order(sorted(self.file_list), self.seed, epoch)
        if "order_sha256" in state:
            assert state["order_sha256"] == self.order_sha256(), (
                f"checkpoint loader order {state['order_sha256']} != this loader's epoch-{epoch} order "
                f"{self.order_sha256()}: the files or their order changed"
            )
        self.epoch = epoch
        self.accepted = state.get("accepted", 0)
        self.current_index = state["current_index"]
        self.skipped = Counter(state.get("skipped", {}))  # counts of the games before the checkpoint
        self.pending = [os.path.join(self.path, f) for f in state.get("pending", [])]

    def __repr__(self):
        return (f"DataLoader(path={self.path}, batch_size={self.batch_size}, n_files={len(self.file_list)}, "
                f"seed={self.seed}, epoch={self.epoch})")


class DataLoaderCluster:
    """Yields batches from a list of DataLoaders (one per year) indefinitely, the years mixed.

    Each batch comes from one loader, drawn by `draw_index` with weight `expected_accepted_left`, so every
    loader's share of the accepted games left stays about constant and the loaders finish an epoch together.
    A loader with nothing left is never drawn. With one loader no draw is made.
    When every loader has nothing left, every loader starts its next epoch (a new order) and so does the cluster.
    If every file list is empty, __next__ raises RuntimeError instead of spinning.
    A batch can be empty (every game of it skipped); it is returned as is.
    idx: the loader of the batch last returned. draws: draws made so far (the counter of the next draw's seed).
    """

    def __init__(self, dataloaders: list, seed=0):
        assert dataloaders, "DataLoaderCluster needs at least one DataLoader"
        assert len({dl.batch_size for dl in dataloaders}) == 1, (
            f"loaders of one cluster need one batch_size: {[dl.batch_size for dl in dataloaders]}")
        self.dataloaders = dataloaders
        self.seed = seed
        self.idx = 0
        self.draws = 0
        self.epoch = 0
        # Warn early if any loader found no files.
        for dl in dataloaders:
            if not dl.file_list:
                print(f"WARNING: DataLoader for path '{dl.path}' found no .log files.")

    def __iter__(self):
        return self

    def state_dict(self) -> dict:
        return {"seed": self.seed, "idx": self.idx, "draws": self.draws, "epoch": self.epoch,
                "loaders": [dl.state_dict() for dl in self.dataloaders]}

    def load_state_dict(self, state: dict):
        """A state written before T-045 (no "draws") loads only for one loader: its position keeps its meaning."""
        assert len(state["loaders"]) == len(self.dataloaders), (
            f"checkpoint has {len(state['loaders'])} loaders, this cluster {len(self.dataloaders)}")
        if "draws" in state:
            assert state["seed"] == self.seed, f"checkpoint cluster seed {state['seed']} != {self.seed}"
            self.draws, self.epoch = state["draws"], state["epoch"]
        else:
            assert len(self.dataloaders) == 1, (
                "a pre-T-045 cluster state of several loaders (year by year) has no position in the mixed order")
            self.draws, self.epoch = 0, 0
        self.idx = state["idx"]
        for dl, s in zip(self.dataloaders, state["loaders"]):
            dl.load_state_dict(s)

    def skipped(self) -> Counter:
        return sum((dl.skipped for dl in self.dataloaders), Counter())

    @property
    def last_batch_files(self) -> list:
        """Files of the batch last returned (it came from the loader at `idx`)."""
        return self.dataloaders[self.idx].last_batch_files

    def __next__(self):
        """With data workers, a batch that fails with anything but UnexpectedSkip (a dead worker, a stop) leaves the
        cluster and every loader as they were before it (the draw and an epoch change included)."""
        if all(getattr(dl, "workers", None) is None for dl in self.dataloaders):
            return self._next()
        snapshot = (self.idx, self.draws, self.epoch, [dl._snapshot() for dl in self.dataloaders])
        try:
            return self._next()
        except UnexpectedSkip:
            raise
        except BaseException:
            self.idx, self.draws, self.epoch, loaders = snapshot
            for dl, s in zip(self.dataloaders, loaders):
                dl._restore(s)
            raise

    def _next(self):
        if not any(dl.file_list for dl in self.dataloaders):
            raise RuntimeError(
                "DataLoaderCluster: all dataloaders are empty. "
                "Check that your data paths exist and contain .log files."
            )
        if all(dl.left() == 0 for dl in self.dataloaders):
            for dl in self.dataloaders:
                dl.next_epoch()
            self.epoch += 1
        if len(self.dataloaders) == 1:
            self.idx = 0
        else:
            weights = [expected_accepted_left(dl.left(), dl.accepted, sum(dl.skipped.values()))
                       for dl in self.dataloaders]
            self.idx = draw_index(weights, self.seed, self.draws)
            self.draws += 1
        return next(self.dataloaders[self.idx])
