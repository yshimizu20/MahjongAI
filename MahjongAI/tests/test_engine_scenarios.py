"""T-053 engine: every scripted scenario (kan emission rows, terminal tags, settlement, pao, riichi yakuman) with the
sidecar check on each, and the real-tag check."""
import pytest

from MahjongAI.engine import engine_lib


@pytest.mark.parametrize("make", engine_lib.SCENARIOS, ids=lambda m: m.__name__)
def test_scenario(make):
    sc = make()
    played = engine_lib.play_scenario(sc)
    counts = engine_lib.check_scenario(sc, played)  # the scenario's expectations + rounds, tags, kans, the sidecar
    assert counts["decision coverage"] > 0 and counts["decision coverage"] == len(played.round.decisions)


def test_real_tags_reproduced():
    counts = engine_lib.check_real_tags(engine_lib.gate_logs())
    assert sum(v for k, v in counts.items() if k.startswith("AGARI")) == 30
    assert sum(v for k, v in counts.items() if k.startswith("RYUUKYOKU")) == 10


def test_kan_emission_clean_on_real_logs():
    counts = engine_lib.collections.Counter()
    for f in engine_lib.gate_logs():
        counts += engine_lib.check_kan_emission(open(f, encoding="utf-8", errors="ignore").read(), f)
    assert counts["ankan: N DORA T"] > 0 and counts["open_kan: N T DORA D"] > 0


def test_kan_emission_catches_swapped_order():
    played = engine_lib.play_scenario(engine_lib.sc_ankan_chained())
    bad = engine_lib.re.sub(r"(<N [^>]*/>)(<DORA [^>]*/>)(<T\d+/>)", r"\1\3\2", played.xml, count=1)
    with pytest.raises(engine_lib.EngineCheckError):
        engine_lib.check_kan_emission(bad)
