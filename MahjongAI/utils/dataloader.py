import os

from MahjongAI.process_xml import process, InvalidGameException


class DataLoader:
    def __init__(self, path: str):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == len(self.file_list):
            self.current_index = 0

        filename = self.file_list[self.current_index]
        print(filename)
        self.current_index += 1

        try:
            turns = process(os.path.join(self.path, filename))
            return turns
        except InvalidGameException as e:
            print(e.msg)
            return self.__next__()
