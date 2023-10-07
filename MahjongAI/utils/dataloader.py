import os

from MahjongAI.process_xml import process, InvalidGameException


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            for filename in os.listdir(self.path):
                print(filename)
                try:
                    turns = process(os.path.join(self.path, filename))
                except InvalidGameException as e:
                    print(e.msg)
                    continue

                yield turns
