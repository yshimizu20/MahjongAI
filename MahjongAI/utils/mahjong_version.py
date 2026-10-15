"""Version guard for the `mahjong` library: utils/agari.py and utils/win_oracles.py are written for 1.x."""
from importlib.metadata import version


def require_mahjong_1():
    """Raise ImportError unless mahjong 1.x is installed (requirements.txt pins 1.4.0).

    mahjong 2.x renames the wind yaku (YakuhaiOfPlace / YakuhaiOfRound -> SeatWind* / RoundWind*) and removes
    Yaku.tenhou_id, so yaku names and Tenhou ids would change silently. Porting to 2.x is task T-036."""
    found = version("mahjong")
    if not found.startswith("1."):
        raise ImportError(
            f"MahjongAI needs mahjong 1.x (pinned 1.4.0), found {found}: 2.x renames the wind yaku and removes "
            "Yaku.tenhou_id (port: T-036)"
        )
