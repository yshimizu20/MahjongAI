"""T-053 engine, the stage-5 review's rule findings against REAL Tenhou logs (tests/data/engine_fixtures), with the
expectations taken from the logs' own tags: S1 West-round dealer continuation, S2 pao over the whole hand, S3
owari, S5 riichi yakuman ura; plus the rule oracles (round transitions, tenpai / nagashi reconstruction, owari) on
every complete gate log."""
from MahjongAI.engine import engine_lib, scoring
from MahjongAI.engine.game import next_round
from MahjongAI.engine.round import RoundOutcome, RoundSpec


def test_real_fixtures_reproduced():
    counts = engine_lib.check_real_fixtures()
    assert counts["S1 West continuation after dealer tenpai with a 30,000 non-dealer"] == 1
    assert sum(v for k, v in counts.items() if k.startswith("S2 pao")) == 4
    assert counts["real fixture: every transition, running score and owari reproduced"] == 6
    assert counts["S5 real riichi yakuman tags listing doraHaiUra"] >= 1


def test_s1_west_dealer_tenpai_with_another_seat_at_30000_continues():
    spec = RoundSpec(8, 1, 0, (243, 173, 295, 289))  # 2021081622gm-00a9-0000-17dc8b4d, West 1 honba 1
    out = RoundOutcome("exhaustive", [243, 183, 305, 259], 1, dealer_tenpai=True)
    assert next_round(spec, out) == RoundSpec(8, 2, 1, (243, 183, 305, 259))
    # the dealer first with 30,000 at a dealer win / tenpai still ends it (agari-yame), an abort never does
    assert next_round(RoundSpec(8, 0, 0, (290, 240, 240, 230)),
                      RoundOutcome("agari", [320, 240, 240, 200], 0, dealer_won=True, winners=[0])) is None
    assert next_round(RoundSpec(8, 0, 0, (290, 240, 240, 230)),
                      RoundOutcome("abort", [320, 240, 240, 200], 0, abort_type="kan4")) == \
        RoundSpec(8, 1, 0, (320, 240, 240, 200))


def test_s2_pao_covers_the_whole_hand():
    compound = scoring.HandScore(26, 50, 64000, 5, (), (39, 42), 0, 320)
    assert scoring.agari_deltas(compound, 0, 0, 0, 3, 0, {39: 2}) == [649, 0, -649, 0]  # 2019103023...f90e0fe9
    single = scoring.HandScore(13, 60, 48000, 5, (), (39,), 0, 160)
    assert scoring.agari_deltas(single, 3, 0, 3, 2, 0, {39: 1}) == [-240, -246, 0, 486]  # 2020040823...44be7808


def test_s3_owari_rounds_raw_scores_and_balances_first_place():
    assert scoring.owari_points([265, 448, -8, 295]) == [-13.0, 54.0, -51.0, 10.0]  # 2021081109...fc96a2e9
    assert scoring.owari_points([305, 295, 205, 195]) == [39.0, 10.0, -19.0, -30.0]
    assert sum(scoring.owari_points([199, 345, 64, 392])) == 0.0
    # from 2022-01-22 the raw scores keep 0.1 precision: 2022051222gm-00a9-0000-b02c0edb
    assert scoring.owari_points([-5, 423, 269, 313], exact=True) == [-50.5, 52.3, -13.1, 11.3]
    assert scoring.OWARI_EXACT_FROM == "20220122"


def test_rule_oracles_agree_with_every_complete_gate_log():
    counts = engine_lib.check_corpus(engine_lib.gate_logs())
    assert counts["disagreements"] == 0
    assert counts["games"] >= 30 and counts["owari"] == counts["games"]
    assert sum(v for k, v in counts.items() if k.startswith("skipped: GO type")) >= 1  # the tonpuu gate log
    assert sum(v for k, v in counts.items() if k.startswith("RYUUKYOKU exhaustive")) > 0
