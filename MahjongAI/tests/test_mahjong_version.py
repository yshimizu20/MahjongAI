"""The mahjong version guard (T-036): the code reads 2.x yaku names and ids, so any other major version must fail at
import, never mislabel a yaku."""
import pytest

from MahjongAI.utils import mahjong_version


def test_guard_passes_on_the_installed_version():
    assert mahjong_version.version("mahjong").startswith("2.")
    mahjong_version.require_mahjong_2()


@pytest.mark.parametrize("found", ["1.4.0", "3.0.0"])
def test_guard_raises_on_another_major_version(monkeypatch, found):
    monkeypatch.setattr(mahjong_version, "version", lambda name: found)
    with pytest.raises(ImportError, match=f"needs mahjong 2.x \\(pinned 2.0.0\\), found {found}"):
        mahjong_version.require_mahjong_2()
