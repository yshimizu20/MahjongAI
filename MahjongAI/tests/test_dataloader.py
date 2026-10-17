"""DataLoader skip handling (plan T-005): an expected exclusion is counted (and logged with its file when asked); any
other skip raises UnexpectedSkip after the loader has moved past the game; skip counts survive state_dict."""
import os

import pytest

from MahjongAI.supervised_learning import EXPECTED_SKIPS, LOGGED_SKIPS
from MahjongAI.utils import gates
from MahjongAI.utils.dataloader import DataLoader, UnexpectedSkip

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
KEY = "invalid:Own-turn action after own discard"
DAY, NAME, _ = gates.AFTER_OWN_DISCARD_LOGS[0]
DAY_DIR = os.path.join(GATE_GAMES, DAY[:4], DAY)
PATH = os.path.join(DAY_DIR, f"{DAY}_00", NAME)


def _drain(loader):
    for _ in loader:
        pass


def test_expected_skip_is_counted_and_logged():
    lines = []
    loader = DataLoader(DAY_DIR, 1, lines.append, expected_skips=EXPECTED_SKIPS, logged_skips=LOGGED_SKIPS)
    assert PATH in loader.file_list
    _drain(loader)
    assert loader.skipped[KEY] >= 1
    assert f"skipped {PATH}: {KEY}" in lines
    restored = DataLoader(DAY_DIR, 1, lines.append)
    restored.load_state_dict(loader.state_dict())
    assert restored.skipped == loader.skipped and restored.current_index == loader.current_index


def test_unexpected_skip_raises_past_the_game():
    loader = DataLoader(DAY_DIR, 1, lambda *_: None, expected_skips=EXPECTED_SKIPS - {KEY})
    with pytest.raises(UnexpectedSkip, match="Own-turn action after own discard"):
        _drain(loader)
    assert os.path.basename(loader.file_list[loader.current_index - 1]) in {g[1] for g in gates.AFTER_OWN_DISCARD_LOGS}
    assert loader.skipped[KEY] == 1
