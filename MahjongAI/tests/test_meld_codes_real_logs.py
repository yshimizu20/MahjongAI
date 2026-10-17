"""T-026: the meld-code layer against REAL Tenhou logs, and T-008: the offered kakan is the log code.

test_log_melds_replay replays the committed fixtures with the standard library only (no parser): 136-id hands from
INIT, draws and discards, and checks every <N m="..."> against the tiles the seat really holds and the discard it
really calls. tests/test_gates.py::test_decode_meld_exhaustive checks draw.Naki against legal_oracles.decode_meld;
both could share one wrong reading of Tenhou's bits, which only a real log exposes.
"""
import collections
import glob
import os
import re

from MahjongAI.draw import Naki
from MahjongAI.process_xml import process
from MahjongAI.turn import DuringTurn
from MahjongAI.utils import legal_oracles
from MahjongAI.utils.constants import DECISION_NAKI_IDX

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
EVENT = re.compile(r'<(INIT|N|[D-G]\d+|[T-W]\d+)\b([^>]*)/?>')
# T-046: seat 1 kakans the red 5s (88) it discarded earlier (a malformed log the parser rejects)
MALFORMED_KAKAN_LOG = "2022010616gm-00a9-0000-d4d83e3e.log"
KAKAN_LOGS = ["2021010105gm-00a9-0000-0e0fb57b.log",  # gates.OPEN_KAN_LOG
              "2021010214gm-00a9-0000-7ed48ecb.log"]  # gates.PENDING_FLIP_TSUMO_LOG


def _fixture(name):
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
    assert len(found) == 1, (name, found)
    return found[0]


def _attr(attrs, name):
    return re.search(rf'\b{name}="([^"]*)"', attrs).group(1)


def replay_melds(raw):
    """Every meld of one log, checked against the replayed 136-id hands. Returns (kind counts, anomalies)."""
    kinds, anomalies = collections.Counter(), []
    hands = pons = last = None  # last = (discarder, discarded id) of the latest discard, None after a draw

    def anomaly(name, who, m):
        anomalies.append((name, who, m))

    for tag, attrs in EVENT.findall(raw):
        if tag == "INIT":
            hands = [set(int(i) for i in _attr(attrs, f"hai{p}").split(",")) for p in range(4)]
            pons, last = [{} for _ in range(4)], None
        elif tag[0] in "TUVW":
            hands["TUVW".index(tag[0])].add(int(tag[1:]))
            last = None
        elif tag[0] in "DEFG":
            p, tile = "DEFG".index(tag[0]), int(tag[1:])
            if tile not in hands[p]:
                anomaly("discard_not_held", p, tile)
            hands[p].discard(tile)
            last = (p, tile)
        else:
            who, m = int(_attr(attrs, "who")), int(_attr(attrs, "m"))
            meld, nk = legal_oracles.decode_meld(m), Naki(m)
            kind = next(k for k, f in (("chi", nk.is_chi), ("pon", nk.is_pon), ("kakan", nk.is_kakan),
                                       ("minkan", nk.is_minkan), ("ankan", nk.is_ankan)) if f())
            kinds[kind] += 1
            if kind != meld.kind:
                anomaly("kind_mismatch", who, m)
                continue
            exposed, _ = nk.get_exposed()
            taken = legal_oracles.taken_ids(meld)
            if sorted(exposed) != sorted(taken):
                anomaly("exposed_ne_taken", who, m)
            k = legal_oracles.meld_kind(meld)
            if kind in ("chi", "pon", "minkan"):
                if last is None or meld.called != last[1]:
                    anomaly("called_ne_discard", who, m)
                if last is None or (who + meld.from_rel) % 4 != last[0]:
                    anomaly("from_ne_discarder", who, m)
                if kind == "chi" and meld.from_rel != 3:
                    anomaly("chi_not_from_left", who, m)
                if kind == "pon":
                    pons[who][k] = m
            elif kind == "kakan":
                pon = pons[who].pop(k, None)
                if pon is None:
                    anomaly("kakan_without_pon", who, m)
                elif Naki.kakan_of(Naki(pon)).naki_code != m:
                    anomaly("kakan_ne_kakan_of_pon", who, m)
            if not set(taken) <= hands[who]:
                anomaly(f"{kind}_{'added' if kind == 'kakan' else 'taken'}_not_held", who, m)
            hands[who] -= set(taken)
            last = None
    return kinds, anomalies


def test_log_melds_replay():
    files = sorted(glob.glob(os.path.join(GATE_GAMES, "**", "*.log"), recursive=True))
    assert len(files) == 47, len(files)  # 41 + the 6 T-043 / T-044 fixtures
    kinds, anomalies = collections.Counter(), {}
    for f in files:
        k, a = replay_melds(open(f, "rb").read().decode("utf-8", "ignore"))
        kinds.update(k)
        if a:
            anomalies[os.path.basename(f)] = a
    assert all(kinds[k] > 0 for k in ("chi", "pon", "kakan", "minkan", "ankan")), kinds
    # the one malformed fixture must be caught: the replay can fail on a real log
    assert set(anomalies) == {MALFORMED_KAKAN_LOG}, anomalies
    assert [name for name, *_ in anomalies[MALFORMED_KAKAN_LOG]] == ["kakan_added_not_held"], anomalies


def test_executed_kakan_offer_equals_log_code():
    for name in KAKAN_LOGS:
        path = _fixture(name)
        raw = open(path, "rb").read().decode("utf-8", "ignore")
        log_kakan = [int(m) for m in re.findall(r'<N who="\d" m="(\d+)"', raw)
                     if legal_oracles.decode_meld(int(m)).kind == "kakan"]
        H, _ = process(path)
        offered = [d.naki.naki_code for halfturns in H for t in halfturns if isinstance(t, DuringTurn)
                   for d in t.decisions[DECISION_NAKI_IDX] if d.executed and d.naki.is_kakan()]
        assert len(log_kakan) == 2 and len(offered) == 2, (name, log_kakan, offered)
        assert offered == log_kakan, (name, offered, log_kakan)
