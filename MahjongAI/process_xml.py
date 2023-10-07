import sys
import xml.etree.ElementTree as ET
import numpy as np
import copy
from mahjong.shanten import Shanten

sys.path.append("..")

from MahjongAI.gameinfo import GameInfo, RoundInfo
from MahjongAI.utils.constants import *
from MahjongAI.utils.agari import evaluate_ron, evaluate_tsumo
from MahjongAI.draw import Naki
from MahjongAI.state import StateObject
from MahjongAI.decision import (
    NakiDecision,
    ReachDecision,
    decision_mask,
)
from MahjongAI.turn import DuringTurn, DiscardTurn, PostTurn
from MahjongAI.result import AgariResult, RyukyokuResult


shanten_solver = Shanten()


class InvalidGameException(Exception):
    def __init__(self, msg):
        self.msg = msg


def process(file_path: str, verbose: bool = False):
    gameinfo, rounds = _get_rounds(file_path)

    if gameinfo.against_human() != 1:
        raise InvalidGameException("Against human")
    if gameinfo.no_red() == 1:
        raise InvalidGameException("Red tiles")
    if gameinfo.kansaki() == 1:
        raise InvalidGameException("Kansaki")
    if gameinfo.three_players() == 1:
        raise InvalidGameException("Three players")

    all_halfturns = []

    for kyoku_events in rounds:
        # get imitial state
        kyoku_init_state = kyoku_events[0]
        assert kyoku_init_state["event"] == "INIT"

        scores = [int(score) for score in kyoku_init_state["attr"]["ten"].split(",")]
        parent = kyoku_init_state["attr"]["oya"]
        curr_round, honba, kyotaku, _, _, dora = list(
            map(int, kyoku_init_state["attr"]["seed"].split(","))
        )
        doras = [dora]
        remaining_rounds = gameinfo.n_rounds() - int(curr_round)
        parent_rounds_remaining = [(remaining_rounds + i) // 4 for i in range(4)]
        roundinfo = RoundInfo(gameinfo, curr_round)

        # create state object tensors
        hand_indices = [
            list(map(int, kyoku_init_state["attr"][f"hai{player}"].split(",")))
            for player in range(4)
        ]  # 0 - 135
        remaining_tiles = REMAINING_TILES.copy()
        remaining_tiles_pov = [REMAINING_TILES.copy() for _ in range(4)]
        remaining_tsumo = REMAINING_TSUMO
        hand_tensors = [np.zeros(37, dtype=np.float32) for _ in range(4)]  # 0-36
        sutehai_tensor = np.zeros((4, 37), dtype=np.float32)  # to check furiten

        for hi, ht, pov in zip(hand_indices, hand_tensors, remaining_tiles_pov):
            for tile in hi:
                tile_idx = TILE2IDX[tile]
                ht[tile_idx] += 1.0
                pov[tile_idx] -= 1.0
                remaining_tiles[tile_idx] -= 1.0

        remaining_tiles[TILE2IDX[int(dora)]] -= 1
        for pov in remaining_tiles_pov:
            pov[TILE2IDX[int(dora)]] -= 1

        hand_tensors_full = copy.deepcopy(
            hand_tensors
        )  # hand info including all melded tiles

        hands = np.zeros((4, 136), dtype=np.float32)
        player_indices = np.arange(4)[:, np.newaxis]
        hands[player_indices, hand_indices] = 1.0

        assert np.allclose(hands.sum(axis=1), [13.0, 13.0, 13.0, 13.0])
        assert np.max(hands) == 1.0
        assert sum(remaining_tiles[:34]) == 83

        # more non-tensor state objects
        double_reaches = [0] * 4
        reaches = [0] * 4
        ippatsu = [0] * 4
        melds = [[] for _ in range(4)]
        turns = []
        halfturns = []
        is_menzen = [True] * 4
        curr_halfturn = None
        cycles = 0
        encoding_tokens = []
        acquired = None
        result_idx = None

        if verbose:
            print(f"Scores: {scores}")
            print(f"Parent: {parent}")
            print(f"Current round: {curr_round}")
            print(f"Honba: {honba}")
            print(f"Kyotaku: {kyotaku}")
            print(f"Remaining rounds: {remaining_rounds}")
            print(f"Parent rounds remaining: {parent_rounds_remaining}")
            print("\n" + "=" * 20 + "\n")

        for event_idx, event in enumerate(kyoku_events[1:]):
            # print(event)
            eventtype = event["event"]

            if eventtype == "DORA":
                tile = int(event["attr"]["hai"])
                tile_idx = TILE2IDX[tile][0]
                doras = doras[:] + [tile]
                remaining_tiles[tile_idx] -= 1
                for pov in remaining_tiles_pov:
                    pov[tile_idx] -= 1
                remaining_tsumo -= 1
                encoding_tokens.append(new_dora2idx(tile_idx))

            elif eventtype == "REACH":
                assert curr_halfturn is not None
                player = int(event["attr"]["who"])

                if event["attr"]["step"] == "1":
                    assert isinstance(curr_halfturn, DuringTurn)
                    assert acquired is not None

                    for decision in curr_halfturn.decisions:
                        if isinstance(decision, ReachDecision):
                            decision.executed = True
                            break

                else:  # event["attr"]["step"] == 2
                    assert isinstance(curr_halfturn, PostTurn)
                    reaches = reaches[:]
                    reaches[player] = 1
                    kyotaku += 1
                    scores = scores.copy()
                    scores[player] -= 10
                    if cycles <= 4 and min(is_menzen):
                        double_reaches = double_reaches[:]
                        double_reaches[player] = 1
                    encoding_tokens.append(reach2idx(player))

                ippatsu = ippatsu[:]
                ippatsu[player] = 1

            elif eventtype == "N":
                assert curr_halfturn is not None
                acquired = None

                naki = Naki(int(event["attr"]["m"]))
                player = int(event["attr"]["who"])
                from_who = naki.from_who()
                cycles += (player - from_who) % 4
                melds[player] = melds[player][:] + [naki]

                # change executed to True
                if naki.is_ankan() or naki.is_kakan():
                    assert isinstance(curr_halfturn, DuringTurn)
                    for lst in curr_halfturn.decisions:
                        for decision in lst:
                            if (
                                isinstance(decision, NakiDecision)
                                and decision.naki.convenient_naki_code
                                == naki.convenient_naki_code
                            ):
                                decision.executed = True
                                break

                else:
                    assert isinstance(curr_halfturn, PostTurn)
                    # TODO: exclude decision that would not have been picked up anyways
                    for lst in curr_halfturn.decisions:
                        for decision in lst:
                            if (
                                isinstance(decision, NakiDecision)
                                and decision.naki.convenient_naki_code
                                == naki.convenient_naki_code
                            ):
                                decision.executed = True
                                break

                if not naki.is_chi() and not naki.is_pon():
                    remaining_tsumo -= 1

                exposed, acquired = naki.get_exposed()
                exposed_idx = TILE2IDX[exposed]

                for i, pov in enumerate(remaining_tiles_pov):
                    if i == player:
                        continue
                    for e in exposed_idx:
                        for ee in e:
                            pov[ee] -= 1.0

                for e in exposed:
                    assert hands[player, e] == 1.0
                    hands[player, e] -= 1.0
                    hand_tensors[player][TILE2IDX[e]] -= 1.0

                if acquired is not None:
                    hand_tensors_full[player][TILE2IDX[acquired]] += 1.0

                stateObj = StateObject(
                    remaining_turns=remaining_tsumo,
                    hand_tensor=hand_tensors[player],
                    remaining_tiles=remaining_tiles,
                    remaining_tiles_pov=remaining_tiles_pov[player],
                    sutehai_tensor=sutehai_tensor[player],
                    reaches=reaches,  # share same reach list
                    melds=melds,
                    scores=scores,
                    kyotaku=kyotaku,
                    honba=honba,
                    dora=doras,  # share same dora list
                )

                if naki.from_who() != 0:
                    is_menzen[player] = False
                if sum(ippatsu):
                    ippatsu = [0] * 4

                if naki.is_kakan():
                    # evaluate chankan ron
                    curr_halfturn.decisions += evaluate_ron(
                        player=player,
                        hand_tensors_full=hand_tensors_full,
                        naki_list=melds,
                        sutehai_tensor=sutehai_tensor,
                        discarded_tile=exposed[0],
                        doras=doras,
                        reaches=reaches,
                        ippatsu=ippatsu,
                        is_chankan=True,
                        is_haitei=False,
                        double_reaches=double_reaches,
                        is_renhou=[False] * 4,
                        player_wind=roundinfo.player_wind,
                        round_wind=roundinfo.round_wind,
                        kyotaku=kyotaku,
                        honba=honba,
                    )

                    # remove corresponding pon from melds
                    for i, meld in enumerate(melds[player]):
                        if meld.is_pon():
                            color, number, *_ = meld.pattern_pon()
                            if color * 9 + number == exposed[0] // 4:
                                melds[player].pop(i)
                                break

                # TODO: add logic for ankan kokushi ron

                curr_halfturn = None
                encoding_tokens.append(naki2idx(player, naki))

                if naki.is_minkan() or naki.is_ankan() or naki.is_kakan():
                    cycles += 1
                    acquired = None

            elif eventtype[0] in ["T", "U", "V", "W"]:
                assert acquired is None
                curr_halfturn = None

                player = ["T", "U", "V", "W"].index(eventtype[0])
                cycles += 1
                acquired = int(eventtype[1:])
                tile_idx = TILE2IDX[acquired]
                remaining_tiles[tile_idx] -= 1
                remaining_tiles_pov[player][tile_idx] -= 1
                remaining_tsumo -= 1
                assert hands[player, acquired] == 0.0

                hands[player, acquired] = 1.0
                hand_tensors[player][tile_idx] += 1.0
                hand_tensors_full[player][tile_idx] += 1.0
                assert int(hand_tensors[player][:34].sum()) % 3 == 2

                decisions = [[] for _ in range(4)]
                during_decisions = decisions[player]

                # check if tsumo is possible
                during_decisions += evaluate_tsumo(
                    player=player,
                    hand_tensors_full=hand_tensors_full,
                    naki_list=melds,
                    tsumo_tile=acquired,
                    doras=doras,
                    reaches=reaches,
                    ippatsu=ippatsu,
                    is_haitei=(remaining_tsumo == 0),
                    is_rinshan=False,
                    double_reaches=double_reaches,
                    is_tenhou=(cycles == 1),
                    is_chiihou=(2 <= cycles <= 4 and min(is_menzen)),
                    player_wind=roundinfo.player_wind,
                    round_wind=roundinfo.round_wind,
                    kyotaku=kyotaku,
                    honba=honba,
                )

                # check if ankan is possible
                for i in np.where(hand_tensors[player] == 4.0)[0]:
                    during_decisions.append(
                        NakiDecision(player, Naki.from_ankan_info(i), executed=False)
                    )

                # TODO: make it cleaner
                # check if kakan is possible
                for meld in melds[player]:
                    if meld.is_pon():
                        color, number, *_ = meld.pattern_pon()
                        if hands[player, 9 * color + number] == 1.0:
                            during_decisions.append(
                                NakiDecision(
                                    player,
                                    Naki.from_kakan_info(
                                        9 * color + number, 0
                                    ),  # TODO get accurate "which" value
                                    executed=False,
                                )
                            )

                # check if reach is possible
                if not reaches[player] and is_menzen[player]:
                    shanten = shanten_solver.calculate_shanten(
                        hand_tensors[player][:34]
                    )
                    if shanten <= 0:
                        during_decisions.append(ReachDecision(player, executed=False))

                stateObj = StateObject(
                    remaining_turns=remaining_tsumo,
                    hand_tensor=hand_tensors[player],
                    remaining_tiles=remaining_tiles,
                    remaining_tiles_pov=remaining_tiles_pov[player],
                    sutehai_tensor=sutehai_tensor[player],
                    reaches=reaches,  # share same reach list
                    melds=melds,
                    scores=scores,
                    kyotaku=kyotaku,
                    honba=honba,
                    dora=doras,  # share same dora list
                )

                halfturn = DuringTurn(
                    player=player,
                    stateObj=stateObj,
                    decisions=decisions,
                    encoding_tokens=encoding_tokens[:],
                )

                halfturns.append(halfturn)
                curr_halfturn = halfturn  # carry over
                if sum(ippatsu):
                    ippatsu = [0] * 4

            elif eventtype[0] in ["D", "E", "F", "G"]:
                assert acquired is not None

                # the tile was obtained through tsumo
                if curr_halfturn is not None:
                    assert isinstance(curr_halfturn, DuringTurn)
                curr_halfturn = None

                player = ["D", "E", "F", "G"].index(eventtype[0])

                tile = int(eventtype[1:])
                tile_idx = TILE2IDX[tile]
                assert hands[player, tile] == 1.0

                half_turn = DiscardTurn(
                    player=player,
                    stateObj=stateObj,
                    discarded_tile=tile,
                    encoding_tokens=encoding_tokens[:],
                )
                halfturns.append(half_turn)

                hands[player, tile] = 0.0
                hand_tensors[player][tile_idx] -= 1.0
                hand_tensors_full[player][tile_idx] -= 1.0
                assert int(hand_tensors[player][:34].sum()) % 3 == 1, hand_tensors[
                    player
                ].sum()
                sutehai_tensor[player, tile_idx] = 1.0

                for pov in remaining_tiles_pov:
                    pov[tile_idx] -= 1.0
                remaining_tiles_pov[player][tile_idx] += 1.0

                # post_decisions will contain three lists of different decisions: ron, pon/kan, chi
                post_decisions = []

                rons = evaluate_ron(
                    player=player,
                    hand_tensors_full=hand_tensors_full,
                    naki_list=melds,
                    sutehai_tensor=sutehai_tensor,
                    discarded_tile=tile,
                    doras=doras,
                    reaches=reaches,
                    ippatsu=ippatsu,
                    is_chankan=False,
                    is_haitei=(remaining_tsumo == 0),
                    double_reaches=double_reaches,
                    is_renhou=[
                        (cycles <= 3 and player >= cycles and min(is_menzen))
                        for _ in range(4)
                    ],
                    player_wind=roundinfo.player_wind,
                    round_wind=roundinfo.round_wind,
                    kyotaku=kyotaku,
                    honba=honba,
                )
                post_decisions.append(rons)

                pon_chi_decisions = decision_mask(
                    player, hand_tensors, tile_idx[0], len(tile_idx) == 2
                )
                post_decisions += pon_chi_decisions

                half_turn = PostTurn(
                    player=player,
                    stateObj=stateObj,
                    decisions=post_decisions,
                    encoding_tokens=encoding_tokens[:],
                )
                halfturns.append(half_turn)
                curr_halfturn = half_turn  # carry over

                encoding_tokens.append(
                    discard2idx(player, tile_idx[0], tile == acquired)
                )
                acquired = None

            elif eventtype == "AGARI" or eventtype == "RYUUKYOKU":
                result_idx = event_idx + 1
                break

            else:
                raise ValueError(f"Invalid event type: {eventtype}")

        result_events = kyoku_events[result_idx:]
        result = []

        for event in result_events:
            assert event["event"] in ["AGARI", "RYUUKYOKU"]

            if event["event"] == "AGARI":
                result.append(
                    AgariResult(list(map(int, event["attr"]["sc"].split(","))))
                )
                # TODO: make decision executed = True

            elif event["event"] == "RYUUKYOKU":
                result.append(
                    RyukyokuResult(list(map(int, event["attr"]["sc"].split(","))))
                )

        if verbose:
            print(f"len(turns): {len(turns)}")
            print(result)
            print(f"remaining_tiles: {remaining_tiles}")
            print(f"player hands: {hand_tensors}")

        assert sum([1 for r in remaining_tiles if r < 0]) == 0
        assert sum([1 for rem in remaining_tiles_pov for r in rem if r < 0]) == 0

        all_halfturns.append(halfturns)

    return all_halfturns


def _get_rounds(file_path: str):
    with open(file_path, "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    events = []

    def traverse(element: ET.Element):
        item = {"event": element.tag, "attr": element.attrib}
        events.append(item)
        for child in element:
            traverse(child)

    traverse(root)

    # get game config
    gametype = root.findall("GO")
    if len(gametype) != 1:
        raise ValueError("Invalid number of game type elements")
    try:
        gameinfo = GameInfo(int(gametype[0].attrib["type"]))
    except:
        raise ValueError("Invalid game type element")

    # group events into each round
    rounds = []
    curr_round_events = None

    for event in events:
        if event["event"] in ["mjloggm", "SHUFFLE", "UN", "GO", "TAIKYOKU"]:
            continue
        elif event["event"] == "BYE":
            raise InvalidGameException("BYE")

        if event["event"] == "INIT":
            if curr_round_events is not None:
                rounds.append(curr_round_events)
            curr_round_events = []

        curr_round_events.append(event)

    if curr_round_events is not None:
        rounds.append(curr_round_events)

    return gameinfo, rounds


NAKI_IDX_START = 37 * 4 * 2
CHI_IDX_START = NAKI_IDX_START
PON_IDX_START = CHI_IDX_START + 4 * 3 * 3 * 7 * 3 * 2
KAKAN_IDX_START = PON_IDX_START + 4 * 3 * 4 * 9 * 2
MINKAN_IDX_START = KAKAN_IDX_START + 4 * 4 * 9 * 2
ANKAN_IDX_START = MINKAN_IDX_START + 4 * 3 * 4 * 9 * 2
REACH_IDX_START = ANKAN_IDX_START + 4 * 4 * 9 * 2
NEW_DORA_IDX_START = REACH_IDX_START + 4  # 4116


def discard2idx(who: int, tile_idx: int, is_tsumogiri: bool):
    return (who * 37 + tile_idx) * 2 + int(is_tsumogiri)


def naki2idx(who: int, naki: Naki):
    from_who = naki.from_who()
    if naki.is_chi():
        color, number, which, has_red, *_ = naki.pattern_chi()
        return NAKI_IDX_START + (
            ((((who * 3 + from_who - 1) * 3 + color) * 7 + number) * 3 + which) * 2
            + int(has_red)
        )
    elif naki.is_pon():
        color, number, _, has_red, *_ = naki.pattern_pon()
        return PON_IDX_START + (
            (((who * 3 + from_who - 1) * 4 + color) * 9 + number) * 2 + int(has_red)
        )
    elif naki.is_kakan():
        color, number, _, has_red, *_ = naki.pattern_kakan()
        return KAKAN_IDX_START + (((who * 4 + color) * 9 + number) * 2 + int(has_red))
    elif naki.is_minkan():
        color, number, _, has_red, *_ = naki.pattern_minkan()
        return MINKAN_IDX_START + (
            (((who * 3 + from_who - 1) * 4 + color) * 9 + number) * 2 + int(has_red)
        )
    elif naki.is_ankan():
        color, number, _, has_red, *_ = naki.pattern_ankan()
        return ANKAN_IDX_START + (((who * 4 + color) * 9 + number) * 2 + int(has_red))
    else:
        raise ValueError("Invalid naki code")


def reach2idx(who: int):
    return REACH_IDX_START + who


def new_dora2idx(tile_idx: int):
    return NEW_DORA_IDX_START + tile_idx


if __name__ == "__main__":
    # import cProfile

    file_path = "data/sample/sample_haifu.xml"
    # cProfile.run("process(file_path, verbose=True)", sort="tottime")
    all_halfturns = process(file_path, verbose=True)

    print(list(map(len, all_halfturns)))
