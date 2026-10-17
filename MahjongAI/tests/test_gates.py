"""pytest entry for the pre-training gates: the same regression as `python -m MahjongAI.utils.gates --test`.

Runs on the committed fixture games in tests/data/gate_games/ (same <yyyy>/<day>/<day>_00/ layout as
data/processed/), so it needs no local data. They are exactly the processed/ files the self-test reads, so its
sorted picks are unchanged. When self_test() starts reading a new game, re-trace the set against the full tree:
wrap builtins.open, run gates.self_test() with the default DATA_ROOT, and copy every opened processed/ file here."""
import ast
import collections
import os
import random
from unittest import mock

import numpy as np
import pytest

from MahjongAI import process_xml
from MahjongAI.draw import Naki
from MahjongAI.process_xml import process
from MahjongAI.utils import agari, gates, legal_oracles, win_oracles
from MahjongAI.utils.legal_oracles import Meld

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")


def test_gates_self_test(monkeypatch):
    monkeypatch.setattr(gates, "DATA_ROOT", GATE_GAMES)
    gates.self_test()


# ---- T-040: the oracles share no code with the parser

PARSER_MODULES = {"MahjongAI.process_xml", "MahjongAI.decision", "MahjongAI.draw", "MahjongAI.gameinfo",
                  "MahjongAI.utils.legal", "MahjongAI.utils.agari"}
TOKEN_BUILDERS = {"decision_mask", "naki2idx", "discard2idx", "reach2idx", "new_dora2idx"}
BANNED_IN_ORACLES = {"Naki", "legal", "agari", "gameinfo"} | TOKEN_BUILDERS
# gates.py functions that read the PARSER's objects or patch the parser: the only ones allowed the banned names
PARSER_SIDE = {"index_checks", "_option_fixtures", "self_test"}
# the runner helpers parse a game: the parser's entry point is the one parser name they may hold
RUNNER = {"run", "_count", "_first_valid_game", "_game_with", "_parsed_fixture", "main"}
RUNNER_ORIGINS = {"MahjongAI.process_xml.process", "MahjongAI.process_xml.InvalidGameException"}
# Naki's own decoders: an oracle decodes the raw naki_code with legal_oracles.decode_meld instead
NAKI_DECODERS = {"get_exposed", "pattern_chi", "pattern_pon", "pattern_kakan", "pattern_minkan", "pattern_ankan"}
ORACLES = {"xml_rounds", "check_state_xml", "during_options", "post_options", "post_rule_hits", "answered_token",
           "check_wins", "check_win_point", "check_win_anchors", "win_flags", "win_routes", "history_facts",
           "check_actions", "tok_discard", "tok_call", "tok_reach", "tok_dora", "kept_xml", "auto_play_runs"}


def _tree(module):
    return ast.parse(open(module.__file__).read())


def _origins(tree, module_name):
    """{local name: dotted origin} of every import in the tree (aliases and relative imports resolved), plus every
    origin imported anywhere, as (origin, lineno) pairs."""
    package = module_name.rsplit(".", 1)[0].split(".")
    bound, imported = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                imported.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            for a in node.names:
                bound[a.asname or a.name] = f"{mod}.{a.name}"
                imported.append((f"{mod}.{a.name}", node.lineno))
    for node in tree.body:  # a module-level rebinding (X = legal, Y = process_xml.naki2idx) keeps its origin
        if isinstance(node, ast.Assign):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in bound:
                origin = bound[root.id] + "".join(f".{a}" for a in reversed(_attr_chain(node.value)))
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = origin
    return bound, imported


def _attr_chain(node):
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    return out


def _parser_origin(origin):
    return (any(origin == m or origin.startswith(m + ".") for m in PARSER_MODULES)
            or origin.rsplit(".", 1)[-1] in TOKEN_BUILDERS)


def _dynamic_imports(tree):
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Name) and n.func.id == "__import__")
        or (isinstance(n.func, ast.Attribute) and n.func.attr == "import_module"))]


def test_oracle_independence():
    for module in (legal_oracles, win_oracles):
        tree = _tree(module)
        _, imported = _origins(tree, module.__name__)
        hits = [(o, line) for o, line in imported if _parser_origin(o)]
        assert not hits, f"{module.__name__} imports parser code: {hits}"
        assert not _dynamic_imports(tree), f"{module.__name__} imports dynamically at lines {_dynamic_imports(tree)}"
    tree = _tree(gates)
    bound, imported = _origins(tree, gates.__name__)
    hits = [(o, line) for o, line in imported if o.rsplit(".", 1)[-1] in TOKEN_BUILDERS]
    assert not hits, f"gates.py imports a parser builder: {hits}"
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = (ORACLES | PARSER_SIDE) - set(functions)
    assert not missing, f"gates.py lost the functions {sorted(missing)}: update this test"
    # every function outside the parser side: no banned name, no name bound to parser code under any alias or
    # rebinding (the runner helpers: the parse entry point only), no token builder or Naki decoder attribute
    hits = {}
    for name, fn in functions.items():
        if name in PARSER_SIDE:
            continue
        allowed = RUNNER_ORIGINS if name in RUNNER else set()
        names = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name)}
        bad = {n for n in names if n in BANNED_IN_ORACLES}
        bad |= {n for n in names if _parser_origin(bound.get(n, "")) and bound[n] not in allowed}
        bad |= {x.attr for x in ast.walk(fn) if isinstance(x, ast.Attribute) and x.attr in TOKEN_BUILDERS | NAKI_DECODERS}
        if bad:
            hits[name] = sorted(bad)
    assert not hits, f"gate functions outside {sorted(PARSER_SIDE)} reach parser code: {hits}"
    assert not _dynamic_imports(tree), f"gates.py imports dynamically at lines {_dynamic_imports(tree)}"


def _every_meld_code():
    """Every valid four-player Tenhou <N m> code: chi (from the kamicha only), pon, kakan, daiminkan, ankan."""
    for t in range(21 * 3):
        for c0 in range(4):
            for c1 in range(4):
                for c2 in range(4):
                    yield t << 10 | c2 << 7 | c1 << 5 | c0 << 3 | 0x4 | 3
    for bit in (0x8, 0x10):
        for t in range(34 * 3):
            for unused in range(4):
                for rel in (1, 2, 3):
                    yield t << 9 | unused << 5 | bit | rel
    for hai0 in range(136):
        for rel in (0, 1, 2, 3):
            yield hai0 << 8 | rel


def test_decode_meld_exhaustive():
    codes = list(_every_meld_code())
    assert len(codes) == len(set(codes)) == 4032 + 2 * 1224 + 136 * 4
    kinds = collections.Counter()
    for m in codes:
        meld, nk = legal_oracles.decode_meld(m), Naki(m)
        kind = next(k for k, f in (("chi", nk.is_chi), ("pon", nk.is_pon), ("kakan", nk.is_kakan),
                                   ("minkan", nk.is_minkan), ("ankan", nk.is_ankan)) if f())
        kinds[kind] += 1
        exposed, acquired = nk.get_exposed()
        assert meld.kind == kind and meld.from_rel == nk.from_who() and meld.code == m, (m, meld)
        assert sorted(legal_oracles.taken_ids(meld)) == sorted(exposed), (m, meld, exposed)
        if kind == "kakan":
            k = meld.added // 4
            assert acquired is None and meld.added == exposed[0] and meld.ids == tuple(range(4 * k, 4 * k + 4)), (m, meld)
            pon_code = (m & ~0x10) | 0x8  # the pon this kakan extends: same kind, called position and unused copy
            pon, pon_acquired = legal_oracles.decode_meld(pon_code), Naki(pon_code).get_exposed()[1]
            assert meld.called == pon.called == pon_acquired, (m, meld, pon)
            assert set(pon.ids) | {meld.added} == set(meld.ids) and meld.added not in pon.ids, (m, meld, pon)
        else:
            assert meld.called == acquired and meld.added is None, (m, meld, acquired)
            assert sorted(meld.ids) == sorted(exposed + ([acquired] if acquired is not None else [])), (m, meld)
        if kind in ("ankan", "kakan"):
            assert legal_oracles.during_index(meld) == nk.get_during_turn_filter_idx(), (m, meld)
        else:
            assert legal_oracles.post_index(meld) == nk.get_post_turn_filter_idx(), (m, meld)
    assert kinds == {"chi": 4032, "pon": 1224, "kakan": 1224, "minkan": 408, "ankan": 136}, kinds


# ---- T-040: has_yaku (hand-written) == HandCalculator (library) on complete hands

E, S, W, N = 27, 28, 29, 30
PINFU, MENZEN_TSUMO = win_oracles.TENHOU_YAKU_ID["Pinfu"], win_oracles.TENHOU_YAKU_ID["Tsumo"]
SANANKOU = win_oracles.TENHOU_YAKU_ID["Sanankou"]


def _kinds(text):
    """'123m 55p EE' -> tile kinds (z: E S W N haku hatsu chun as E S W N P F C)."""
    out = []
    for group in text.split():
        if group[-1] in "mps":
            out += [9 * "mps".index(group[-1]) + int(d) - 1 for d in group[:-1]]
        else:
            out += [27 + "ESWNPFC".index(ch) for ch in group]
    return out


def _flags(tsumo, seat=E, rnd=E):
    return win_oracles.WinFlags(tsumo=tsumo, riichi=False, double_riichi=False, ippatsu=False, rinshan=False,
                                chankan=False, haitei=False, houtei=False, tenhou=False, chiihou=False,
                                seat_wind=seat, round_wind=rnd)


def _both(concealed, win, tsumo, pons=(), seat=E, rnd=E):
    """(has_yaku, library ids) of a closed hand `concealed` (with the win tile) plus open pons of the given kinds."""
    used = collections.Counter()

    def take(k):
        used[k] += 1
        assert used[k] <= 4
        return 4 * k + used[k] - 1 if k not in (4, 13, 22) else 4 * k + used[k] % 4  # never a red five
    win_id = None
    ids = []
    for k in _kinds(concealed):
        ids.append(take(k))
        if k == win and win_id is None:
            win_id = ids[-1]
    melds = []
    for k in pons:
        mids = tuple(take(k) for _ in range(3))
        melds.append(Meld("pon", mids, mids[0], None, 1, 0))
    c34 = np.bincount([i // 4 for i in ids], minlength=34)
    flags = _flags(tsumo, seat, rnd)
    return win_oracles.has_yaku(c34, melds, win, flags), win_oracles.yaku_ids_by_library(ids, melds, win_id, flags)


@pytest.mark.parametrize("name, concealed, win, tsumo, pons, seat, rnd, expected", [
    ("pinfu, non-yakuhai wind pair", "23m 4m 567m 345p 678s WW", 3, False, (), E, E, {PINFU}),
    ("round-wind pair, ron: no yaku", "23m 4m 567m 345p 678s EE", 3, False, (), S, E, set()),
    ("round-wind pair, tsumo: menzen tsumo only", "23m 4m 567m 345p 678s EE", 3, True, (), S, E, {MENZEN_TSUMO}),
    ("penchan ron: no yaku", "12m 3m 456p 345s 789s 55p", 2, False, (), E, E, set()),
    ("5m in 456567m: pinfu by the 567 reading", "4566m 7m 5m 123p 678s 99s", 4, False, (), E, E, {PINFU}),
    ("shanpon ron on 111m 222p 333s: no yaku", "111m 222p 33s 3s 456s 77p", 20, False, (), E, E, set()),
    ("shanpon tsumo on 111m 222p 333s: sanankou", "111m 222p 33s 3s 456s 77p", 20, True, (), E, E,
     {MENZEN_TSUMO, SANANKOU}),
    ("three of a guest wind (open): no yaku", "23m 1m 789p 456s 99s", 0, False, (N,), E, E, set()),
    ("123m 789m 55p 123s 789s win 1m: pinfu, not chanta", "23m 1m 789m 55p 123s 789s", 0, False, (), E, E, {PINFU}),
])
def test_has_yaku_probes(name, concealed, win, tsumo, pons, seat, rnd, expected):
    exists, lib = _both(concealed, win, tsumo, pons, seat, rnd)
    assert lib == expected, (name, sorted(lib))
    assert exists == bool(expected), (name, exists)


def _random_win(rng):
    """One random complete hand with real ids: 4 blocks + a pair (or chiitoitsu / kokushi), random open melds
    (chi / pon / minkan / ankan / kakan), the win tile on a random concealed tile, random winds and flags. None when
    the draw ran out of copies."""
    suit = rng.randrange(3)
    honor_bias = rng.random() < .3
    blocks = []
    for _ in range(4):
        if rng.random() < .35:
            k = rng.choice(range(27, 34)) if honor_bias and rng.random() < .5 else (
                9 * suit + rng.randrange(9) if rng.random() < .5 else rng.randrange(34))
            blocks.append(("trip", k))
        else:
            blocks.append(("run", 9 * (suit if rng.random() < .6 else rng.randrange(3)) + rng.randrange(7)))
    pair = rng.randrange(34) if rng.random() < .5 else 9 * suit + rng.randrange(9)
    free = {k: rng.sample(range(4 * k, 4 * k + 4), 4) for k in range(34)}
    melds, concealed = [], []
    n_open = rng.choice([0, 0, 1, 2, 3])
    for bi in rng.sample(range(4), 4):
        s, k = blocks[bi]
        kinds = [k] * 3 if s == "trip" else [k, k + 1, k + 2]
        if len(melds) < n_open:
            kind = "chi" if s == "run" else rng.choice(["pon", "pon", "minkan", "ankan", "kakan"])
            kinds = kinds + [k] if kind in ("minkan", "ankan", "kakan") else kinds
            if any(not free[x] for x in kinds) or sum(1 for x in kinds if x == k) > len(free[k]):
                return None
            ids = tuple(sorted(free[x].pop() for x in kinds))
            melds.append(Meld(kind, ids, None if kind == "ankan" else ids[0], ids[3] if kind == "kakan" else None,
                              3 if kind == "chi" else 0 if kind == "ankan" else 2, 0))
        else:
            if sum(1 for x in kinds if x == k) > len(free[k]) or any(not free[x] for x in kinds):
                return None
            concealed += [free[x].pop() for x in kinds]
    if len(free[pair]) < 2:
        return None
    concealed += [free[pair].pop(), free[pair].pop()]
    if not melds and rng.random() < .15:
        if rng.random() < .5:
            kinds = rng.sample(range(34) if rng.random() < .5 else range(9 * suit, 9 * suit + 9), 7)
            concealed = [4 * k + j for k in kinds for j in (0, 1)]
        else:
            concealed = [4 * k for k in win_oracles.ORPHANS] + [4 * rng.choice(win_oracles.ORPHANS) + 1]
    win_id = rng.choice(concealed)
    tsumo, situ = rng.random() < .5, rng.random() < .1
    menzen = all(m.kind == "ankan" for m in melds)
    flags = win_oracles.WinFlags(
        tsumo=tsumo, riichi=situ and menzen and rng.random() < .5, double_riichi=False, ippatsu=False,
        rinshan=situ and tsumo and rng.random() < .3, chankan=False, haitei=False,
        houtei=situ and not tsumo and rng.random() < .3, tenhou=False, chiihou=False,
        seat_wind=rng.choice(range(27, 31)), round_wind=rng.choice(range(27, 30)))
    return concealed, melds, win_id, flags


def test_has_yaku_vs_library():
    rng = random.Random(40)
    n = yes = 0
    bad = []
    while n < 10000:
        case = _random_win(rng)
        if case is None:
            continue
        concealed, melds, win_id, flags = case
        n += 1
        exists = win_oracles.has_yaku(np.bincount([i // 4 for i in concealed], minlength=34), melds, win_id // 4, flags)
        lib = win_oracles.yaku_ids_by_library(concealed, melds, win_id, flags)
        yes += bool(lib)
        if exists != bool(lib):
            bad.append((sorted(concealed), melds, win_id, flags, exists, sorted(lib)))
    assert not bad, f"{len(bad)} / {n} hands disagree, e.g. {bad[:3]}"
    assert 0.2 < yes / n < 0.95, f"{yes} / {n} hands with a yaku: the sample lost one side"


# ---- T-036: Tenhou's yaku ids under mahjong 2.x, pinned by hand (a library reassignment must fail here)

TENHOU_IDS = {
    "Tsumo": 0, "Riichi": 1, "Ippatsu": 2, "Chankan": 3, "Rinshan": 4, "Haitei": 5, "Houtei": 6, "Pinfu": 7,
    "Tanyao": 8, "Iipeiko": 9,
    "SeatWind:E": 10, "SeatWind:S": 11, "SeatWind:W": 12, "SeatWind:N": 13,
    "RoundWind:E": 14, "RoundWind:S": 15, "RoundWind:W": 16,
    "Haku": 18, "Hatsu": 19, "Chun": 20, "DaburuRiichi": 21, "Chiitoitsu": 22, "Chantai": 23, "Ittsu": 24,
    "Sanshoku": 25, "SanshokuDoukou": 26, "SanKantsu": 27, "Toitoi": 28, "Sanankou": 29, "Shosangen": 30,
    "Honroto": 31, "Ryanpeikou": 32, "Junchan": 33, "Honitsu": 34, "Chinitsu": 35, "Renhou": 36, "Tenhou": 37,
    "Chiihou": 38, "Daisangen": 39, "Suuankou": 40, "SuuankouTanki": 41, "Tsuuiisou": 42, "Ryuuiisou": 43,
    "Chinroutou": 44, "ChuurenPoutou": 45, "DaburuChuurenPoutou": 46, "KokushiMusou": 47, "DaburuKokushiMusou": 48,
    "DaiSuushii": 49, "Shousuushii": 50, "Suukantsu": 51,
}


def test_tenhou_yaku_ids_pinned():
    raw = [n for n in win_oracles.TENHOU_YAKU_ID if n.startswith(("SeatWind", "RoundWind", "Yakuhai")) and ":" not in n]
    assert not raw, f"library wind class names must never map to an id: {raw}"
    leaked = {"Dora", "AkaDora", "UraDora", "Paarenchan"} & set(win_oracles.TENHOU_YAKU_ID)
    assert not leaked, f"bonus han / Paarenchan must never map to an id: {leaked}"
    assert win_oracles.TENHOU_YAKU_ID == TENHOU_IDS


@pytest.mark.parametrize("seat, rnd, expected", [
    (W, S, {12}),  # seat wind West in the South round
    (S, W, {16}),  # round wind West (西入), seat South
    (W, W, {12, 16}),  # both
])
def test_wind_ids_by_library(seat, rnd, expected):  # pon West + 123m 789p 456s 99s, ron on 1m
    exists, lib = _both("23m 1m 789p 456s 99s", 0, False, (W,), seat, rnd)
    assert lib == expected and exists


def test_every_wind_id_by_library():  # every SeatWind* / RoundWind* class through the library's own id map
    bad = []
    for pon_wind in (E, S, W, N):
        for seat in (E, S, W, N):
            for rnd in (E, S, W, N):  # North round: never played on Tenhou, but the library maps it to 17
                expected = ({10 + seat - E} if pon_wind == seat else set()) | ({14 + rnd - E} if pon_wind == rnd else set())
                exists, lib = _both("23m 1m 789p 456s 99s", 0, False, (pon_wind,), seat, rnd)
                if lib != expected or exists != bool(expected):
                    bad.append((pon_wind, seat, rnd, sorted(lib), exists))
    assert not bad, bad


def test_chinroutou_through_both_adapters():  # Tenhou id 44: no hand in the library differential hit it
    concealed = np.bincount(_kinds("111p 999p 11s"), minlength=34)
    melds = [Naki.from_pon_info(0, 0, False, 1), Naki.from_pon_info(8, 0, False, 1)]  # pon 1m, pon 9m
    ctx = agari.WinContext(is_tsumo=False, seat_wind=S, round_wind=E)
    assert agari.win_yaku(concealed, melds, 18, ctx) == ("Chinroutou",)
    assert win_oracles.parser_yaku_ids(("Chinroutou",)) == {44}
    exists, lib = _both("111p 999p 11s", 18, False, (0, 8), S, E)
    assert lib == {44} and exists


# ---- T-041: every gate name is read off failed_gates and proven able to fail

def test_gate_names(monkeypatch):
    names = gates.gate_names()
    assert set(gates.ZERO_GATES) <= names
    for k in gates.LABELLED:
        assert {f"lab_{k}", f"tensor_{k}", f"action_xml_coverage_{k}"} <= names, k
    for h in gates.HEADS:
        assert {f"{h}_samples", f"{h}_label_outside_mask"} <= names, h
    assert {"games", "dora_draw_to_discard_checked", "win_yaku_coverage", "agari_coverage"} <= names
    assert set(gates.GATE_ONLY) <= names and not gates.UNTESTED_GATES
    monkeypatch.setattr(gates, "ZERO_GATES", gates.ZERO_GATES + ["t041_new_gate"])
    assert "t041_new_gate" in gates.gate_names()


def _contract_violations(fn):
    """Compare nodes of a function AST that break the failed_gates contract: an operator other than == / != / in /
    not in, a numeric literal other than 0 as a side, or a non-integer / >= 500 coefficient inside a side."""
    bad = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        if not all(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            bad.append((node.lineno, "operator"))
        for side in [node.left] + node.comparators:
            for sub in ast.walk(side):
                v = getattr(sub, "value", None) if isinstance(sub, ast.Constant) else None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if sub is side and v != 0:
                        bad.append((node.lineno, f"literal {v}"))
                    elif sub is not side and not (isinstance(v, int) and abs(v) < 500):
                        bad.append((node.lineno, f"coefficient {v}"))
    return bad


def test_failed_gates_contract():
    fn = next(n for n in _tree(gates).body if isinstance(n, ast.FunctionDef) and n.name == "failed_gates")
    assert not _contract_violations(fn)
    for dirty in ('if C["x"] == 1: pass', 'if C["x"] > 0: pass', 'if C["x"] != 600 * C["y"]: pass'):
        assert _contract_violations(ast.parse(dirty)), dirty  # the check can fail


def test_gate_coverage_gaps():
    names, gate_only = {"a", "b", "g"}, {"g": "oracle only"}
    full = {"a": {"parser"}, "b": {"parser", "gate"}, "g": {"gate"}}
    assert gates.gate_coverage_gaps(names, full, gate_only, {}) == []
    assert gates.gate_coverage_gaps(names | {"new"}, full, gate_only, {}) == [
        "new: parser gate with no parser-side corruption"]
    assert gates.gate_coverage_gaps(names, {**full, "a": {"gate"}}, gate_only, {}) == [
        "a: parser gate with no parser-side corruption (gate-code corruptions only)"]
    assert gates.gate_coverage_gaps(names, {"a": {"parser"}, "b": {"parser"}}, gate_only, {}) == [
        "g: gate-only check with no corruption"]
    assert gates.gate_coverage_gaps(names | {"u"}, full, gate_only, {"u": "cannot fail"}) == []
    assert gates.gate_coverage_gaps(names, full, gate_only, {"a": "cannot fail"}) == [
        "a: in UNTESTED_GATES but a corruption fails it (drop the entry)"]
    assert gates.gate_coverage_gaps(names, full, {**gate_only, "old": "gone"}, {"gone": "x"}) == [
        "gone: stale UNTESTED_GATES entry (failed_gates cannot return it)",
        "old: stale GATE_ONLY entry (failed_gates cannot return it)"]
    assert "g: on both GATE_ONLY and UNTESTED_GATES" in gates.gate_coverage_gaps(names, full, gate_only, {"g": "x"})


def test_gate_fingerprint():
    f0 = gates.gate_fingerprint()
    with mock.patch.object(gates, "snap_kans", lambda snap: (4, 1)):
        assert gates.gate_fingerprint() != f0
    with mock.patch.dict(win_oracles.TENHOU_YAKU_ID, {"Suuankou": 41, "SuuankouTanki": 40}):
        assert gates.gate_fingerprint() != f0
    assert gates.gate_fingerprint() == f0
    with mock.patch.object(process_xml, "_is_houtei", lambda *a: False):
        assert gates.gate_fingerprint() == f0
    with mock.patch.object(Naki, "get_post_turn_filter_idx", lambda self: 0):
        assert gates.gate_fingerprint() == f0
    H, T = process(gates.SAMPLE_LOG)
    H[0][0].stateObj.honba, T[0][0] = 99, T[0][0] ^ 1
    assert gates.gate_fingerprint() == f0


def test_watch_gate_code(monkeypatch):
    calls = []
    monkeypatch.setattr(gates, "count_game", lambda *a: calls.append(a))
    dummy = gates.count_game
    with gates.watch_gate_code() as (check, touched):
        gates.count_game(1)
        assert not touched  # nothing patched: a parser-side corruption
        with mock.patch.object(gates, "xml_rounds", lambda raw: []):
            gates.count_game(2)  # the spy compares at every count call
        assert touched
    assert gates.count_game is dummy and calls == [(1,), (2,)]
    with gates.watch_gate_code() as (check, touched):
        with mock.patch.object(win_oracles, "has_yaku", lambda *a: False):
            check()  # no count call: the on-demand comparison
    assert touched


# ---- T-042: per-year default days; the CLI path on the fixture tree

def test_resolve_days(monkeypatch):
    val_week = [f"202201{d:02d}" for d in range(1, 8)]
    assert gates.resolve_days("2022", None, 0, GATE_GAMES) == val_week  # the T-042 crash: --year 2022, no --days
    every = gates.resolve_days(None, None, 0, GATE_GAMES)
    assert every == gates.DEFAULT_POOL["2021"] + val_week and {"20211230", "20211231"} <= set(every)
    assert gates.resolve_days("2021", ["20210101", "20210102"], 0, GATE_GAMES) == ["20210101", "20210102"]
    with pytest.raises(SystemExit, match="20220101"):
        gates.resolve_days("2021", ["20220101"], 0, GATE_GAMES)
    with pytest.raises(SystemExit, match="2021011"):
        gates.resolve_days(None, ["2021011"], 0, GATE_GAMES)
    assert gates.resolve_days("2020", None, 0, GATE_GAMES) == ["20200719"] == gates.resolve_days("2020", None, 5, GATE_GAMES)
    with pytest.raises(SystemExit, match="2019"):
        gates.resolve_days("2019", None, 0, GATE_GAMES)
    monkeypatch.setattr(gates, "DEFAULT_POOL", {"2022": val_week})  # 2021 outside the pool: 21 fixture day dirs
    days = sorted(os.listdir(os.path.join(GATE_GAMES, "2021")))
    got = gates.resolve_days("2021", None, 3, GATE_GAMES)
    assert len(days) > gates.OUTSIDE_POOL_DAYS and len(got) == gates.OUTSIDE_POOL_DAYS == len(set(got))
    assert set(got) <= set(days) and got == sorted(got) == gates.resolve_days("2021", None, 3, GATE_GAMES)


def test_select_files():
    files = gates.select_files(gates.resolve_days("2022", None, 0, GATE_GAMES), GATE_GAMES)
    assert files and all(os.path.relpath(f, GATE_GAMES).split(os.sep)[1] == "20220106" for f in files)
    files = gates.select_files(gates.resolve_days(None, None, 0, GATE_GAMES), GATE_GAMES)
    assert {os.path.relpath(f, GATE_GAMES).split(os.sep)[0] for f in files} == {"2021", "2022"}  # both pool years
    with pytest.raises(SystemExit, match="20190101"):
        gates.select_files(["20190101"], GATE_GAMES)


def test_default_pool_2022_is_the_val_week():
    from MahjongAI.supervised_learning import VAL_DAYS
    assert gates.DEFAULT_POOL["2022"] == list(VAL_DAYS)


def test_main_cli(monkeypatch, capsys):
    monkeypatch.setattr(gates, "DATA_ROOT", GATE_GAMES)
    gates.main(["--year", "2021", "--days", "20210101", "--games", "2"])
    assert "GATES OK" in capsys.readouterr().out
