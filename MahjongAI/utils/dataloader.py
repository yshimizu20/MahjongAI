import os
import random
from collections import Counter

from MahjongAI.process_xml import process, InvalidGameException


class UnexpectedSkip(RuntimeError):
    """A game skipped for a reason outside the loader's `expected_skips`. Raised after the loader has moved past
    the game, so a run that saves its state and resumes does not meet it again."""


class DataLoader:
    """Yields batches of parsed games from every `*.log` under `path`.

    Layout assumed: <path>/<day>/<day>_<nn>/<game>.log, so a game's day is its second parent directory.
    days / exclude_days: keep only, or drop, the games of those day directories (the train/val split).
    seed: shuffle the file list once with this seed (None keeps the sorted, date-ordered list).
    Games that fail to parse are counted in `skipped` (key "invalid:<reason>" or the exception's type name).
    expected_skips: None counts every skip and continues; a set of keys raises UnexpectedSkip on any other key.
    last_batch_files: the files of the games in the batch last returned, in batch order.
    """

    def __init__(self, path: str, batch_size: int = 1, log_func: callable = print,
                 days=None, exclude_days=None, seed=None, expected_skips=None):
        self.path = path
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
        if seed is not None:
            random.Random(seed).shuffle(self.file_list)
        self.seed = seed
        self.current_index = 0
        self.batch_size = batch_size
        self.batch_count = 0
        self.log_func = log_func
        self.skipped = Counter()
        self.expected_skips = None if expected_skips is None else frozenset(expected_skips)
        self.last_batch_files = []

    @staticmethod
    def day_of(file_path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(file_path)))

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_count += 1

        if self.current_index >= len(self.file_list):
            raise StopIteration

        all_halfturns_list, all_encoding_tokens_list, files = [], [], []

        while self.current_index < len(self.file_list) and len(all_halfturns_list) < self.batch_size:
            filename = self.file_list[self.current_index]
            self.current_index += 1

            try:
                all_halfturns, all_encoding_tokens = process(filename)
            except InvalidGameException as e:
                self._skip(filename, "invalid:" + e.msg, e)
                continue
            except Exception as e:  # a parser bug on one game: report it, keep going unless told otherwise
                self.log_func(f"skipped {filename}: {type(e).__name__}: {str(e)[:200]}")
                self._skip(filename, type(e).__name__, e)
                continue

            all_halfturns_list.append(all_halfturns)
            all_encoding_tokens_list.append(all_encoding_tokens)
            files.append(filename)

        self.last_batch_files = files
        return all_halfturns_list, all_encoding_tokens_list

    def _skip(self, filename: str, key: str, error: Exception):
        self.skipped[key] += 1
        if self.expected_skips is not None and key not in self.expected_skips:
            raise UnexpectedSkip(f"unexpected skip {key!r} of {filename}: {str(error)[:200]}") from error

    def reset(self):
        """Resets the dataloader's current index for re-iteration."""
        self.current_index = 0

    def state_dict(self) -> dict:
        return {"current_index": self.current_index, "seed": self.seed, "n_files": len(self.file_list),
                "skipped": dict(self.skipped)}

    def load_state_dict(self, state: dict):
        assert state["seed"] == self.seed and state["n_files"] == len(self.file_list), (
            f"checkpoint loader state {state} does not match this loader (seed={self.seed}, n_files={len(self.file_list)})"
        )
        self.current_index = state["current_index"]
        self.skipped = Counter(state.get("skipped", {}))  # counts of the games before the checkpoint

    def __repr__(self):
        return f"DataLoader(path={self.path}, batch_size={self.batch_size}, n_files={len(self.file_list)}, seed={self.seed})"


class DataLoaderCluster:
    """Cycles through a list of DataLoaders indefinitely.

    When a DataLoader is exhausted, it is reset and the cluster advances
    to the next one. When all DataLoaders are exhausted, it wraps back
    to the first, so __next__ never raises StopIteration — unless every
    single DataLoader is empty, in which case a RuntimeError is raised
    immediately to avoid an infinite spin.
    """

    def __init__(self, dataloaders: list):
        self.dataloaders = dataloaders
        self.idx = 0
        # Warn early if any loader found no files.
        for dl in dataloaders:
            if not dl.file_list:
                print(f"WARNING: DataLoader for path '{dl.path}' found no .log files.")

    def __iter__(self):
        return self

    def state_dict(self) -> dict:
        return {"idx": self.idx, "loaders": [dl.state_dict() for dl in self.dataloaders]}

    def load_state_dict(self, state: dict):
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
        # An exhausted loader is reset and the NEXT loader is tried; with one loader that is the same
        # loader again, so allow len + 1 attempts. The cluster is empty only if every loader raises
        # StopIteration on a fresh pass.
        attempts = 0
        while attempts <= len(self.dataloaders):
            try:
                return next(self.dataloaders[self.idx])
            except StopIteration:
                self.dataloaders[self.idx].reset()
                self.idx = (self.idx + 1) % len(self.dataloaders)
                attempts += 1
        raise RuntimeError(
            "DataLoaderCluster: all dataloaders are empty. "
            "Check that your data paths exist and contain .log files."
        )
