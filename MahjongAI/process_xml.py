import sys
import xml.etree.ElementTree as ET
import numpy as np
import copy
from mahjong.shanten import Shanten

sys.path.append("..")

from MahjongAI.gameinfo import GameInfo, RoundInfo
from MahjongAI.utils.constants import *
from MahjongAI.utils.agari import evaluate_ron, evaluate_tsumo
from MahjongAI.draw import Naki, Tsumo
from MahjongAI.discard import Discard
from MahjongAI.state import StateObject
from MahjongAI.decision import (
    NakiDecision,
    ReachDecision,
    AgariDecision,
    PassDecision,
    decision_mask,
)
from MahjongAI.turn import TsumoTurn, NakiTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.result import AgariResult, RyukyokuResult


shanten_solver = Shanten()


def get_rounds(file_path: str):
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
    curr_round_events = []

    for event in events:
        if event["event"] in ["mjloggm", "SHUFFLE", "UN", "GO", "TAIKYOKU"]:
            continue

        curr_round_events.append(event)

        if event["event"] in ["AGARI", "RYUUKYOKU"]:
            rounds.append(curr_round_events)
            curr_round_events = []

    return gameinfo, rounds


def process(file_path: str, verbose: bool = False):
    gameinfo, rounds = get_rounds(file_path)

    assert gameinfo.against_human() == 1
    assert gameinfo.no_red() == 0
    assert gameinfo.kansaki() == 0
    assert gameinfo.three_players() == 0

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
        hand_tensors = [np.zeros(37, dtype=np.float32) for _ in range(4)]  # 0 - 36
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
        half_turns = []
        is_menzen = [True] * 4
        curr_turn = None
        enc_indices = []
        cycles = 0

        if verbose:
            print(f"Scores: {scores}")
            print(f"Parent: {parent}")
            print(f"Current round: {curr_round}")
            print(f"Honba: {honba}")
            print(f"Kyotaku: {kyotaku}")
            print(f"Remaining rounds: {remaining_rounds}")
            print(f"Parent rounds remaining: {parent_rounds_remaining}")
            print("\n" + "=" * 20 + "\n")

        for event in kyoku_events[1:-1]:
            eventtype = event["event"]

            if eventtype == "DORA":
                tile = int(event["attr"]["hai"])
                tile_idx = TILE2IDX[tile]
                doras = doras[:] + [tile]
                remaining_tiles[tile_idx] -= 1
                for pov in remaining_tiles_pov:
                    pov[tile_idx] -= 1
                remaining_tsumo -= 1

                enc_idx = new_dora2idx(tile_idx)
                enc_indices.append(enc_idx)

            elif eventtype == "REACH":
                player = int(event["attr"]["who"])

                if event["attr"]["step"] == "1":
                    assert curr_turn is not None
                    for decision in curr_turn.pre_decisions:
                        if isinstance(decision, ReachDecision):
                            decision.executed = True
                            break
                else:  # step == 2
                    reaches = reaches[:]
                    reaches[player] = 1
                    kyotaku += 1
                    scores = scores.copy()
                    scores[player] -= 1000
                    if cycles <= 4 and min(is_menzen):
                        double_reaches = double_reaches[:]
                        double_reaches[player] = 1

                ippatsu = ippatsu[:]
                ippatsu[player] = 1

                enc_idx = reach2idx(player)
                enc_indices.append(enc_idx)

            elif eventtype == "N":
                naki = Naki(int(event["attr"]["m"]))
                player = int(event["attr"]["who"])
                from_who = naki.from_who()
                cycles += (player - from_who) % 4
                melds[player] = melds[player][:] + [naki]

                # change executed to True
                assert len(turns) > 0
                turn = turns[-1]
                if naki.is_ankan():
                    for lst in turn.pre_decisions:
                        for decision in lst:
                            if (
                                decision.naki.convenient_naki_code
                                == naki.convenient_naki_code
                            ):
                                decision.executed = True
                                break
                else:
                    for lst in turn.post_decisions:
                        for decision in lst:
                            if (
                                decision.naki.convenient_naki_code
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
                    dora=doras,  # share same reach list
                )
                curr_turn = NakiTurn(player=player, naki=naki, stateObj=stateObj)
                if naki.from_who() != 0:
                    is_menzen[player] = False
                if sum(ippatsu):
                    ippatsu = [0] * 4

                if naki.is_kakan():
                    curr_turn.post_decisions += evaluate_ron(
                        player=player,
                        hand_tensors=hand_tensors,
                        naki_list=melds,
                        sutehai_tensor=sutehai_tensor,
                        discarded_tile=acquired,
                        doras=doras,
                        reaches=reaches,
                        ippatsu=ippatsu,
                        is_chankan=True,
                        is_haitei=False,
                        double_reaches=double_reaches,
                        is_renhou=False,
                        player_wind=roundinfo.player_wind,
                        round_wind=roundinfo.round_wind,
                        kyotaku=kyotaku,
                        honba=honba,
                    )

                enc_idx = naki2idx(player, naki)
                enc_indices.append(enc_idx)

            elif eventtype[0] in ["T", "U", "V", "W"]:
                player = ["T", "U", "V", "W"].index(eventtype[0])
                cycles += 1
                tile = int(eventtype[1:])
                tile_idx = TILE2IDX[tile]
                remaining_tiles[tile_idx] -= 1
                remaining_tiles_pov[player][tile_idx] -= 1
                remaining_tsumo -= 1
                assert hands[player, tile] == 0.0
                hands[player, tile] = 1.0
                hand_tensors[player][tile_idx] += 1.0
                hand_tensors_full[player][tile_idx] += 1.0
                assert int(hand_tensors[player][:34].sum()) % 3 == 2

                pre_decisions = [PassDecision(player, executed=False)]
                # check if tsumo is possible
                pre_decisions += evaluate_tsumo(
                    player=player,
                    hand_tensors_full=hand_tensors_full,
                    naki_list=melds,
                    tsumo_tile=tile,
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
                    pre_decisions.append(
                        NakiDecision(player, Naki.from_ankan_info(i), executed=False)
                    )
                # TODO: make it cleaner
                # check if kakan is possible
                for meld in melds[player]:
                    if meld.is_pon():
                        color, number, *_ = meld.pattern_pon()
                        if hands[player, 9 * color + number] == 1.0:
                            pre_decisions.append(
                                NakiDecision(
                                    player, Naki(9 * color + number), executed=False
                                )
                            )
                # check if reach is possible
                if not reaches[player] and is_menzen[player]:
                    shanten = shanten_solver.calculate_shanten(
                        hand_tensors[player][:34]
                    )
                    if shanten <= 0:
                        pre_decisions.append(ReachDecision(player, executed=False))

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
                curr_turn = TsumoTurn(
                    player=player, draw=Tsumo(tile), stateObj=stateObj
                )
                curr_turn.pre_decisions = pre_decisions
                half_turn = DuringTurn(
                    player=player,
                    enc_indices=enc_indices,
                    stateObj=stateObj,
                    decisions=pre_decisions,
                )
                half_turns.append(half_turn)
                if sum(ippatsu):
                    ippatsu = [0] * 4

            elif eventtype[0] in ["D", "E", "F", "G"]:
                player = ["D", "E", "F", "G"].index(eventtype[0])
                assert curr_turn.player == player

                tile = int(eventtype[1:])
                tile_idx = TILE2IDX[tile]
                assert hands[player, tile] == 1.0

                half_turn = DiscardTurn(
                    player=player,
                    enc_indices=enc_indices,
                    stateObj=curr_turn.stateObj,
                    discarded_tile=tile,
                )
                half_turns.append(half_turn)

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
                curr_turn.discard = Discard(tile)
                post_decisions = decision_mask(
                    player, hand_tensors, tile_idx[0], len(tile_idx) == 2
                )
                num_naki = sum(len(m) for m in melds)
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
                        remaining_tsumo + i - 68 >= 0 and num_naki == 0
                        for i in range(4)
                    ],
                    player_wind=roundinfo.player_wind,
                    round_wind=roundinfo.round_wind,
                    kyotaku=kyotaku,
                    honba=honba,
                )
                for ron, lst in zip(rons, post_decisions):
                    if ron is not None:
                        lst.append(ron)
                curr_turn.post_decisions = post_decisions

                half_turn = PostTurn(
                    player=player,
                    enc_indices=enc_indices,
                    stateObj=stateObj,
                    decisions=post_decisions,
                )
                half_turns.append(half_turn)

                enc_idx = discard2idx(player, tile_idx[-1], curr_turn.is_tsumogiri())
                enc_indices.append(enc_idx)

                turns.append(curr_turn)
                curr_turn = None

            else:
                raise ValueError(f"Invalid event type: {eventtype}")

        event = kyoku_events[-1]
        assert event["event"] in ["AGARI", "RYUUKYOKU"]

        if event["event"] == "AGARI":
            result = AgariResult(list(map(int, event["attr"]["sc"].split(","))))

        elif event["event"] == "RYUUKYOKU":
            result = RyukyokuResult(list(map(int, event["attr"]["sc"].split(","))))

        if verbose:
            print(f"len(turns): {len(turns)}")
            print(result)
            print(f"remaining_tiles: {remaining_tiles}")
            print(f"player hands: {hand_tensors}")

        assert sum([1 for r in remaining_tiles if r < 0]) == 0
        assert sum([1 for rem in remaining_tiles_pov for r in rem if r < 0]) == 0


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
    process(file_path, verbose=True)
