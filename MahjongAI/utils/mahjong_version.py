"""Version guard for the `mahjong` library: utils/agari.py and utils/win_oracles.py are written for 2.x."""
from importlib.metadata import version


def require_mahjong_2():
    """Raise ImportError unless mahjong 2.x is installed (requirements.txt pins 2.0.0).

    mahjong 1.x names the wind yaku YakuhaiOfPlace / YakuhaiOfRound (2.x: SeatWind* / RoundWind*, one class per
    direction) and carries Tenhou ids as Yaku.tenhou_id (2.x: YAKU_ID_TO_TENHOU_ID), so the yaku names and Tenhou
    ids would be wrong under 1.x. A later major version may reassign them again."""
    found = version("mahjong")
    if not found.startswith("2."):
        raise ImportError(
            f"MahjongAI needs mahjong 2.x (pinned 2.0.0), found {found}: 1.x uses the old wind yaku classes "
            "(YakuhaiOfPlace / YakuhaiOfRound) and Yaku.tenhou_id"
        )
