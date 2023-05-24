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

    def __repr__(self):
        return f"GameInfo: against_human={self.against_human()}, no_red={self.no_red()}, kansaki={self.kansaki()}, tonnan={self.tonnan()}, three_players={self.three_players()}, fast={self.fast()}, level={self.level()}"


class RoundInfo:
    def __init__(self, gameinfo: GameInfo, curr_round: int):
        self.gameinfo = gameinfo
        N_ROUNDS = 4 * (gameinfo.tonnan() + 1)
        self.remaining_rounds = N_ROUNDS - curr_round
        self.parent_rounds_remaining = [
            (self.remaining_rounds + i) // 4 for i in range(4)
        ]
        self.player_wind = curr_round % 4
        self.round_wind = curr_round // 4

    def __repr__(self):
        return f"RoundInfo: gameinfo={self.gameinfo}, remaining_rounds={self.remaining_rounds}, parent_rounds_remaining={self.parent_rounds_remaining}"
