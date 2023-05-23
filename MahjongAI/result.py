from typing import List


class RoundResult:
    def __init__(self, sc: List[int]):
        self.sc = sc
        # TODO: define reward based on sc


class AgariResult(RoundResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"AgariResult: sc={self.sc}"


class RyukyokuResult(RoundResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"RyukyokuResult: sc={self.sc}"
