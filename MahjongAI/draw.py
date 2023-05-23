class Draw:
    pass


class Naki(Draw):
    def __init__(self, naki_code: int):
        self.naki_code = naki_code

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
        has_red = self.number == 5 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_minkan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = self.number == 5 and color != 3
        exposed = [(9 * color + number) * 4 + c for c in range(4)]
        obtained = exposed.pop(which)
        return (color, number, which, has_red, exposed, obtained)

    def pattern_ankan(self):
        pattern = (self.naki_code & 0xFF00) >> 8
        which = pattern % 4
        pattern //= 4
        color = pattern // 9
        number = pattern % 9
        has_red = self.number == 5 and color != 3
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
