"""Tenhou meld codes the engine writes (`<N m="...">`): the inverse of utils/legal_oracles.decode_meld.

Bit layout (Tenhou, 4-player): the low 2 bits are the discarder's seat relative to the caller (0 for an ankan);
chi: bit 2 set, m >> 10 = (7 * suit + lowest rank) * 3 + called position, the three copy offsets at bits 3-4 /
5-6 / 7-8; pon (bit 3) / kakan (bit 4): m >> 9 = kind * 3 + called position, the copy NOT in the pon (a kakan's
added copy) at bits 5-6; daiminkan / ankan: m >> 8 = a 136-id of the kan, the called one for a daiminkan.

An ankan's id bits name no tile that matters, so four codes decode alike (which = id % 4); the engine writes
Tenhou's usual copy 0 (17027 of 17039 ankans in the 2021 4p logs), the canonical alias of the four.
"""
from typing import Sequence

from MahjongAI.utils.legal_oracles import Meld, decode_meld

CHI_FROM_REL = 3  # a chi is always called from kamicha


def encode_chi(ids: Sequence[int], called: int) -> int:
    """ids: the three 136-ids of the run (two from the hand, `called` from the discard), any order."""
    ids = sorted(ids, key=lambda i: i // 4)
    kinds = [i // 4 for i in ids]
    assert kinds[2] < 27 and kinds[0] // 9 == kinds[2] // 9 and kinds[1] == kinds[0] + 1 and kinds[2] == kinds[0] + 2, ids
    assert called in ids, (ids, called)
    which = ids.index(called)
    suit, rank = divmod(kinds[0], 9)
    t = (7 * suit + rank) * 3 + which
    copies = (ids[0] % 4) << 3 | (ids[1] % 4) << 5 | (ids[2] % 4) << 7
    return t << 10 | copies | 0x4 | CHI_FROM_REL


def encode_pon(ids: Sequence[int], called: int, from_rel: int) -> int:
    """ids: the three 136-ids of the pon (two from the hand, `called` from the discard)."""
    ids = sorted(ids)
    kind = ids[0] // 4
    assert len(ids) == 3 and all(i // 4 == kind for i in ids) and called in ids, (ids, called)
    assert from_rel in (1, 2, 3), from_rel
    unused = next(c for c in range(4) if kind * 4 + c not in ids)
    return (kind * 3 + ids.index(called)) << 9 | unused << 5 | 0x8 | from_rel


def encode_kakan(pon_code: int) -> int:
    """The kakan that extends a pon: Tenhou keeps every bit of the pon and swaps the pon bit for the kakan bit
    (the pon's unused copy is the added one)."""
    assert pon_code & 0x8 and not pon_code & 0x4, f"{pon_code} is not a pon code"
    return (pon_code & ~0x8) | 0x10


def encode_minkan(called: int, from_rel: int) -> int:
    assert from_rel in (1, 2, 3), from_rel
    return called << 8 | from_rel


def encode_ankan(kind: int) -> int:
    assert 0 <= kind < 34, kind
    return (kind * 4) << 8


def encode_meld(meld: Meld) -> int:
    """The code of a decoded meld (legal_oracles.decode_meld): its own code, or the canonical alias of an ankan."""
    if meld.kind == "chi":
        return encode_chi(meld.ids, meld.called)
    if meld.kind == "pon":
        return encode_pon(meld.ids, meld.called, meld.from_rel)
    if meld.kind == "kakan":
        pon_ids = tuple(i for i in meld.ids if i != meld.added)
        return encode_kakan(encode_pon(pon_ids, meld.called, meld.from_rel))
    if meld.kind == "minkan":
        return encode_minkan(meld.called, meld.from_rel)
    if meld.kind == "ankan":
        return encode_ankan(meld.ids[0] // 4)
    raise ValueError(meld.kind)


def canonical(m: int) -> int:
    """m itself, or for an ankan its canonical alias."""
    return encode_meld(decode_meld(m))


def every_4p_code():
    """Every 4-player meld code as Tenhou writes it (the same set tests/test_naki._every_4p_code builds): chi 63 runs
    x 3 called positions x 4^3 copy offsets, pon / kakan 34 kinds x 3 called positions x 4 unused copies x 3
    from_rel, every hai0 x from_rel 0 (ankan) / 1-3 (daiminkan): 4032 + 1224 + 1224 + 136 + 408 = 7024 codes."""
    for t in range(63):
        for c in range(64):
            yield t << 10 | (c >> 4) << 7 | ((c >> 2) & 3) << 5 | (c & 3) << 3 | 0x4 | CHI_FROM_REL
    for bit in (0x8, 0x10):
        for t in range(102):
            for unused in range(4):
                for rel in (1, 2, 3):
                    yield t << 9 | unused << 5 | bit | rel
    for hai0 in range(136):
        for rel in range(4):
            yield hai0 << 8 | rel


def every_alias():
    """Every 16-bit value decode_meld accepts under the 4-player seat rule (a chi from kamicha, a pon / kakan /
    daiminkan from another seat, an ankan from nobody): the 7024 codes plus the values that differ only in bits
    the decoder ignores."""
    for m in range(1 << 16):
        try:
            meld = decode_meld(m)
        except AssertionError:
            continue
        if meld.kind == "chi" and meld.from_rel != CHI_FROM_REL:
            continue
        if meld.kind in ("pon", "kakan", "minkan") and meld.from_rel == 0:
            continue
        if meld.kind == "ankan" and meld.from_rel != 0:
            continue
        yield m
