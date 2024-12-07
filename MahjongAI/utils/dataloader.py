import os
import sys
import torch

from MahjongAI.process_xml import process, InvalidGameException


class DataLoader:
    def __init__(self, path: str, batch_size: int = 1, log_func: callable = print):
        self.path = path
        self.file_list = os.listdir(self.path)
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
                all_halfturns, all_encoding_tokens = process(
                    os.path.join(self.path, filename)
                )
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

        self.log_func(f"{self.__repr__()} - Batch {self.batch_count}")

        return all_halfturns_list, all_encoding_tokens_list

    def reset(self):
        """Resets the dataloader's current index for re-iteration."""
        self.current_index = 0

    def __repr__(self):
        return f"DataLoader(path={self.path}, batch_size={self.batch_size})"
