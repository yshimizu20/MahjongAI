class Draw:
    pass


class Naki(Draw):
    @staticmethod
    def from_chi_info(
        color: int, number: int, which: int, has_red: bool, from_who: int
    ):
        pattern = (color * 7 + number) * 3 + which
        naki_code = (from_who & 3) | 0x4 | 0x8 | 0x20 | 0x80 | (pattern << 10)
        if has_red:
            bit = 3 + (4 - number) * 2
            naki_code ^= 1 << bit

        return Naki(naki_code, clean=False)

    @staticmethod
    def from_pon_info(tile: int, which: int, has_red: bool, from_who: int):
        pattern = tile * 3 + which
        naki_code = (from_who & 3) | 0x8 | (pattern << 9)
        if has_red:
            naki_code |= 1 << 5

        return Naki(naki_code, clean=False)

    @staticmethod
    def from_kakan_info(tile: int):
        pattern = tile * 3
        naki_code = 0x10 | (pattern << 9)

        return Naki(naki_code, clean=False)

    @staticmethod
    def from_minkan_info(tile: int, from_who: int):
        pattern = tile * 4
        naki_code = (from_who & 3) | (pattern << 8)

        return Naki(naki_code, clean=False)

    @staticmethod
    def from_ankan_info(tile: int):
        pattern = tile * 4
        naki_code = pattern << 8

        return Naki(naki_code, clean=False)

    def __init__(self, naki_code: int, clean: bool = True):
        self.naki_code = naki_code
        self.convenient_naki_code = naki_code
        if clean:
            self.convenient_naki_code = self._clean()

    def _clean(self):
        if self.is_chi():
            return self._clean_chi()
        elif self.is_pon():
            return self._clean_pon()
        elif self.is_kakan():
            return self._clean_kakan()
        elif self.is_minkan():
            return self._clean_minkan()
        elif self.is_ankan():
            return self._clean_ankan()
        else:
            raise ValueError("Invalid naki code")

    def _clean_chi(self):
        _, number, _, red, _, _ = self.pattern_chi()
        code = self.naki_code & ~(0x18 | 0x60 | 0x180)
        code |= 0x8 | 0x20 | 0x80
        if red:
            bit = 3 + (4 - number) * 2
            code ^= 1 << bit
        return code

    def _clean_pon(self):
        _, _, _, red, *_ = self.pattern_pon()
        code = self.naki_code & ~(0x60)
        if not red:
            code |= 0x20
        return code

    def _clean_kakan(self):
        _, _, _, red, *_ = self.pattern_kakan()
        code = self.naki_code & ~(0x60)
        if not red:
            code |= 0x20
        return code

    def _clean_minkan(self):
        _, _, which, red, *_ = self.pattern_minkan()
        code = self.naki_code & ~(3 << 8)
        if not red or which != 0:
            code |= 1 << 8
        return code

    def _clean_ankan(self):
        _, _, which, red, *_ = self.pattern_ankan()
        code = self.naki_code & ~(3 << 8)
        if not red or which != 0:
            code |= 1 << 8
        return code

    def from_who(self):
        return self.naki_code & 3

    def is_chi(self):
        return (self.naki_code & 0x4) >> 2

    def is_pon(self):
        return not self.is_chi() and (self.naki_code & 0x8)

    def is_kakan(self):
        return not self.is_chi() and (self.naki_code & 0x10)

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
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_pon(self):
        pattern = (self.naki_code & 0xFE00) >> 9
        which = pattern % 3
        pattern //= 3
        color = pattern // 9
        number = pattern % 9
        code = (self.naki_code & 0x0060) >> 5
        exposed = [(9 * color + number) * 4 + c for c in range(4) if c != code]
        has_red = color < 3 and color * 36 + 16 in exposed
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_kakan(self):
        pattern = (self.naki_code & 0xFE00) >> 9
        which = pattern % 3
        pattern //= 3
        color = pattern // 9
        number = pattern % 9
        has_red = number == 5 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_minkan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = number == 5 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_ankan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = number == 5 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        obtained = None
        return (color, number, which, has_red, exposed, obtained)

    def get_exposed(self):
        exposed, obtained = None, None

        if self.is_chi():
            _, _, _, _, exposed, obtained = self.pattern_chi()

        elif self.is_pon():
            _, _, _, _, exposed, obtained = self.pattern_pon()

        elif self.is_kakan():
            _, _, _, _, exposed, obtained = self.pattern_kakan()

        elif self.is_minkan():
            _, _, _, _, exposed, obtained = self.pattern_minkan()

        elif self.is_ankan():
            _, _, _, _, exposed, obtained = self.pattern_ankan()

        else:
            raise ValueError("Invalid naki code")

        return exposed, obtained


class Tsumo(Draw):
    def __init__(self, tile: int):
        self.tile = tile  # 0-136
