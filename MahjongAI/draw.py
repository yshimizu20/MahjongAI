class Draw:
    pass


# Chi that uses the caller's OWN red five: (number, which) -> slot 0..5 within the colour.
# number = lowest rank of the chi (0-based), which = position of the acquired tile.
# number + which == 4 means the acquired tile is the five itself, which is not a choice of
# the caller and therefore uses the plain chi index instead.
RED_CHI_VARIANT = {(2, 0): 0, (2, 1): 1, (3, 0): 2, (3, 2): 3, (4, 1): 4, (4, 2): 5}


class Naki(Draw):
    """One Tenhou meld code (`<N m="...">`).

    A LOG code (`Naki(m)` from the XML) names the real 136-id tiles. An OFFERED code (built by a `from_*_info`
    constructor or `kakan_of` for a call the parser offers, `offered=True`) is exact in what the model and the gates
    read: the kind, the action index, from_who, and whether a red five is taken from the hand or called (a chi also
    keeps its called position; a pon's called position bits can differ from the log's). It does NOT name which plain
    copy of a kind leaves the hand (the 37-count hand tensors cannot say); only a kakan built by `kakan_of` from a
    log pon is the exact log code. So an offered code is never applied to a hand: `get_exposed` raises on it; apply
    the executed call's log code instead.
    """

    @staticmethod
    def from_chi_info(color: int, number: int, which: int, has_red: bool):
        pattern = (color * 7 + number) * 3 + which
        naki_code = 0x3 | 0x4 | 0x8 | 0x20 | 0x80 | (pattern << 10)
        if has_red:
            bit = 3 + (4 - number) * 2
            naki_code ^= 1 << bit

        return Naki(naki_code, offered=True)

    @staticmethod
    def from_pon_info(tile: int, which: int, has_red: bool, from_who: int):
        # which = 1 if red was in the hand else 0
        assert which in [0, 1]
        pattern = tile * 3 + which
        naki_code = (from_who & 3) | 0x8 | (pattern << 9)
        if has_red:
            naki_code |= 3 << 5

        return Naki(naki_code, offered=True)

    @staticmethod
    def kakan_of(pon: "Naki") -> "Naki":
        """The kakan that extends `pon`: Tenhou keeps the pon's kind, called position, unused-copy bits (now the
        added copy) and from_who, and swaps the pon bit (0x8) for the kakan bit (0x10). Exact (== the log code)
        when `pon` is a log code; a synthetic pon gives a kakan with the same synthetic copies."""
        assert pon.is_pon(), f"kakan_of needs a pon, got {pon!r}"
        return Naki((pon.naki_code & ~0x8) | 0x10, offered=True)

    @staticmethod
    def from_minkan_info(tile: int, from_who: int, which: int):
        # which = 0 if red was acquired else 1
        assert which in [0, 1]
        pattern = tile * 4 + which
        naki_code = (from_who & 3) | (pattern << 8)

        return Naki(naki_code, offered=True)

    @staticmethod
    def from_ankan_info(tile: int):
        pattern = tile * 4
        naki_code = pattern << 8

        return Naki(naki_code, offered=True)

    def __init__(self, naki_code: int, offered: bool = False):
        """offered: True for a code the parser builds for an offered call (see the class docstring)."""
        self.naki_code = naki_code
        self.offered = offered

    def from_who(self):
        return self.naki_code & 3

    def is_chi(self):
        return bool(self.naki_code & 0x4)

    def is_pon(self):
        return not self.is_chi() and (self.naki_code & 0x8) != 0

    def is_kakan(self):
        return not self.is_chi() and (self.naki_code & 0x10) != 0

    def is_minkan(self):
        return self.naki_code & 0b111100 == 0 and self.from_who()

    def is_ankan(self):
        return self.naki_code & 0b111100 == 0 and not self.from_who()

    def pattern_chi(self):
        pattern = (self.naki_code & 0xFC00) >> 10
        which = pattern % 3
        pattern //= 3
        color = pattern // 7
        number = pattern % 7
        has_red = False
        codes = [
            (self.naki_code & mask) >> shift
            for mask, shift in zip([0x0018, 0x0060, 0x0180], [3, 5, 7])
        ]
        exposed = [(9 * color + number + i) * 4 + code for i, code in enumerate(codes)]
        has_red = color * 36 + 16 in exposed
        acquired = exposed.pop(which)
        return (color, number, which, has_red, exposed, acquired)

    def pattern_pon(self):
        pattern = (self.naki_code & 0xFE00) >> 9
        which = pattern % 3
        pattern //= 3
        color = pattern // 9
        number = pattern % 9
        code = (self.naki_code & 0x0060) >> 5
        exposed = [(9 * color + number) * 4 + c for c in range(4) if c != code]
        has_red = color < 3 and color * 36 + 16 in exposed
        acquired = exposed.pop(which)
        return (color, number, which, has_red, exposed, acquired)

    def pattern_kakan(self):
        pattern = (self.naki_code & 0xFE00) >> 9
        which = pattern % 3
        pattern //= 3
        color = pattern // 9
        number = pattern % 9
        has_red = number == 4 and color != 3
        code = (self.naki_code & 0x0060) >> 5
        exposed = [(9 * color + number) * 4 + code]
        acquired = None
        return (color, number, which, has_red, exposed, acquired)

    def pattern_minkan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = number == 4 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        acquired = exposed.pop(which)
        return (color, number, which, has_red, exposed, acquired)

    def pattern_ankan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = number == 4 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        acquired = None
        return (color, number, which, has_red, exposed, acquired)

    def get_exposed(self):
        """The 136-ids this meld takes from the hand and the called one. Log codes only (the class docstring)."""
        if self.offered:
            raise ValueError(f"{self!r} is an offered call: canonical tile copies, not the hand's; "
                             "apply the executed log code")
        exposed, acquired = None, None

        if self.is_chi():
            _, _, _, _, exposed, acquired = self.pattern_chi()

        elif self.is_pon():
            _, _, _, _, exposed, acquired = self.pattern_pon()

        elif self.is_kakan():
            _, _, _, _, exposed, acquired = self.pattern_kakan()

        elif self.is_minkan():
            _, _, _, _, exposed, acquired = self.pattern_minkan()

        elif self.is_ankan():
            _, _, _, _, exposed, acquired = self.pattern_ankan()

        else:
            raise ValueError("Invalid naki code")

        return exposed, acquired  # 0-135

    def get_during_turn_filter_idx(self):
        """
        Returns the index of the filter that should be applied during the turn
        Outputs should land within the range [0, 70]
        """
        ANKAN_START = 3  # 0: pass, 1: agari (tsumo), 2: riichi
        KAKAN_START = 37

        if self.is_ankan():  # 34 patterns
            color, number, _, _, _, _ = self.pattern_ankan()
            return ANKAN_START + 9 * color + number

        elif self.is_kakan():  # 34 patterns
            color, number, _, _, _, _ = self.pattern_kakan()
            return KAKAN_START + 9 * color + number

        raise ZeroDivisionError("Invalid naki code")

    def get_post_turn_filter_idx(self):
        """
        Returns the index of the filter that should be applied after the turn
        Outputs should land within the range [0, 153]
        """
        CHI_START = 2  # 0: pass, 1: agari (ron)
        PON_START = 83
        MINKAN_START = 120

        if self.is_chi():
            color, number, which, has_red, _, _ = self.pattern_chi()
            if has_red and number + which != 4:
                # caller used its own red five -> 18 dedicated slots 65..82, injective by table
                return CHI_START + 63 + 6 * color + RED_CHI_VARIANT[(number, which)]
            else:
                return CHI_START + color * 21 + number * 3 + which

        elif self.is_pon():
            color, number, which, has_red, _, acquired = self.pattern_pon()
            if color == 3:
                return PON_START + 30 + number
            elif has_red and acquired % 4 != 0:
                # red five is in the meld and is not the acquired tile -> caller used its own red
                return PON_START + 27 + color
            else:
                return PON_START + 9 * color + number

        elif self.is_minkan():
            color, number, _, _, _, _ = self.pattern_minkan()
            return MINKAN_START + 9 * color + number

        raise ZeroDivisionError("Invalid naki code")
    
    def __repr__(self):
        if self.is_chi():
            return f"Chi: {self.get_post_turn_filter_idx()}"
        elif self.is_pon():
            return f"Pon: {self.get_post_turn_filter_idx()}"
        elif self.is_kakan():
            return f"Kakan: {self.get_during_turn_filter_idx()}"
        elif self.is_minkan():
            return f"Minkan: {self.get_post_turn_filter_idx()}"
        elif self.is_ankan():
            return f"Ankan: {self.get_during_turn_filter_idx()}"
        else:
            return "Invalid naki code"


class Tsumo(Draw):
    def __init__(self, tile: int):
        self.tile = tile  # 0-135
