import os
import sys
import torch

from MahjongAI.process_xml import process, InvalidGameException


class DataLoader:
    def __init__(self, path: str, batch_size: int = 1, log_func: callable = print):
        self.path = path
        self.file_list = sorted(
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(self.path)
            for fname in fnames
            if fname.endswith(".log")
        )
        self.current_index = 0
        self.batch_size = batch_size
        self.batch_count = 0
        self.log_func = log_func

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_count += 1

        if self.current_index >= len(self.file_list):
            raise StopIteration

        all_halfturns_list, all_encoding_tokens_list = [], []

        while self.current_index < len(self.file_list) and len(all_halfturns_list) < self.batch_size:
            filename = self.file_list[self.current_index]
            # print(f"\nProcessing file: {filename}")
            self.current_index += 1

            try:
                all_halfturns, all_encoding_tokens = process(filename)
            except InvalidGameException as e:
                # print(f"Invalid game detected: {e.msg}")
                continue  # Skip to the next file
            except AssertionError as e:
                print(f"Assertion error: {e} happened in {filename}")
                sys.exit(1) # TODO: remove this line
            except Exception as e:
                print(f"Unexpected error: {e} happened in {filename}")
                continue

            # return all_halfturns, all_encoding_tokens
            all_halfturns_list.append(all_halfturns)
            all_encoding_tokens_list.append(all_encoding_tokens)

        # self.log_func(f"{self.__repr__()} - Batch {self.batch_count}")

        return all_halfturns_list, all_encoding_tokens_list

    def reset(self):
        """Resets the dataloader's current index for re-iteration."""
        self.current_index = 0

    def __repr__(self):
        return f"DataLoader(path={self.path}, batch_size={self.batch_size})"


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

    def __next__(self):
        # Allow at most one full cycle through all loaders without returning data.
        # If every loader raises StopIteration on the first call, all are empty.
        attempts = 0
        while attempts < len(self.dataloaders):
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
