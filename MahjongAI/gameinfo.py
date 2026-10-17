class GameInfo:
    def __init__(self, go: int = 161):
        self.go = go

    def against_human(self):
        return self.go & 0x1

    def no_red(self):
        return (self.go & 0x2) >> 1

    def kansaki(self):
        return (self.go & 0x4) >> 2

    def tonnan(self):
        return (self.go & 0x8) >> 3

    def three_players(self):
        return (self.go & 0x10) >> 4

    def fast(self):
        return (self.go & 0x40) >> 6

    def level(self):
        return (self.go & 0xA0) >> 5

    def n_rounds(self):
        return 4 * (self.tonnan() + 1)

    def rounds_remaining(self, curr_round: int) -> int:
        """The regular rounds left, the current one included; 0 in an extension round (the bakaze one-hot, T-038,
        marks the extension). T-039: never negative."""
        return max(0, self.n_rounds() - curr_round)

    def __repr__(self):
        return f"GameInfo: against_human={self.against_human()}, no_red={self.no_red()}, kansaki={self.kansaki()}, tonnan={self.tonnan()}, three_players={self.three_players()}, fast={self.fast()}, level={self.level()}"
