import os
import torch

from MahjongAI.process_xml import process, InvalidGameException

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, path: str, batch_size: int = 1):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.current_index = 0
        self.batch_size = 1

    def __iter__(self):
        return self

    def __next__(self):
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

            # return all_halfturns, all_encoding_tokens
            all_halfturns_list.append(all_halfturns)
            all_encoding_tokens_list.append(all_encoding_tokens)

        return all_halfturns_list, all_encoding_tokens_list

    def reset(self):
        """Resets the dataloader's current index for re-iteration."""
        self.current_index = 0
