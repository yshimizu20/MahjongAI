import sys
import xml.etree.ElementTree as ET
import numpy as np
from mahjong.shanten import Shanten

sys.path.append("..")

from MahjongAI.gameinfo import GameInfo
from MahjongAI.utils.constants import *
from MahjongAI.utils.agari import FuritenTracker, WinContext, evaluate_ron, evaluate_tsumo, round_wind, seat_wind
from MahjongAI.utils import legal
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
KIND_TO_RED = {4: 34, 13: 35, 22: 36}  # 5m, 5p, 5s -> their red slot in the 37-index


def _mark_safe(row: np.ndarray, kind: int) -> None:
    """Mark tile KIND `kind` (0-33) safe in a (37,) safe_tiles row. Safety is per kind: a five sets both its
    plain slot and its red slot 34-36, whichever copy was discarded."""
    row[kind] = 1.0
    if kind in KIND_TO_RED:
        row[KIND_TO_RED[kind]] = 1.0


def _double_riichi(first_go_around_uninterrupted: bool, own_discards: int, cycles: int, is_menzen: list) -> bool:
    """A riichi is a double riichi iff it is declared on the player's own first discard and nobody has called
    anything yet (chi, pon, daiminkan, ankan or kakan). cycles / is_menzen are unused: they are the old test
    (`cycles <= 4 and min(is_menzen)`, true after an early ankan), which the gates self-test reinstates."""
    return first_go_around_uninterrupted and own_discards == 0


def _is_haitei(remaining_tsumo: int, is_rinshan: bool) -> bool:
    """Haitei: a tsumo on the last wall draw. A rinshan draw is never haitei, also when it is the 70th draw (Tenhou:
    haitei and rinshan never combine)."""
    return remaining_tsumo == 0 and not is_rinshan


def _is_houtei(remaining_tsumo: int, is_chankan: bool, after_rinshan: bool) -> bool:
    """Houtei: a ron on the discard after the last wall draw, also when that draw was a rinshan draw (75/75 such
    rons in the 2021 logs carry houtei). A chankan is never houtei. after_rinshan is unused: it lets the gates
    self-test reinstate the rule that drops houtei after a rinshan draw."""
    return remaining_tsumo == 0 and not is_chankan


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
    all_encoding_tokens = []

    for kyoku_events in rounds:
        # get imitial state
        kyoku_init_state = kyoku_events[0]
        assert kyoku_init_state["event"] == "INIT"

        scores = [int(score) for score in kyoku_init_state["attr"]["ten"].split(",")]
        parent = kyoku_init_state["attr"]["oya"]
        oya = int(parent)
        parent_tensor = np.zeros(4, dtype=np.float32)
        parent_tensor[int(parent)] = 1.0
        curr_round, honba, kyotaku, _, _, dora = list(map(int, kyoku_init_state["attr"]["seed"].split(",")))
        doras = [dora]
        rounds_remaining = gameinfo.n_rounds() - int(curr_round)
        parent_rounds_remaining = [(rounds_remaining + i) // 4 for i in range(4)]
        round_wind_ = round_wind(curr_round)

        # create state object tensors
        hand_indices = [
            list(map(int, kyoku_init_state["attr"][f"hai{player}"].split(","))) for player in range(4)
        ]  # 0 - 135
        remaining_tiles = REMAINING_TILES.copy()
        remaining_tiles_pov = [REMAINING_TILES.copy() for _ in range(4)]
        remaining_tsumo = REMAINING_TSUMO
        hand_tensors = [np.zeros(37, dtype=np.float32) for _ in range(4)]  # 0-36
        sutehai_tensors = np.zeros((4, 37), dtype=np.float32)  # to check furiten

        for hi, ht, pov in zip(hand_indices, hand_tensors, remaining_tiles_pov):
            for tile in hi:
                tile_idx = TILE2IDX[tile]
                ht[tile_idx] += 1.0
                pov[tile_idx] -= 1.0
                remaining_tiles[tile_idx] -= 1.0

        remaining_tiles[TILE2IDX[int(dora)]] -= 1
        for pov in remaining_tiles_pov:
            pov[TILE2IDX[int(dora)]] -= 1

        hands = np.zeros((4, 136), dtype=np.float32)
        player_indices = np.arange(4)[:, np.newaxis]
        hands[player_indices, hand_indices] = 1.0

        assert np.allclose(hands.sum(axis=1), [13.0, 13.0, 13.0, 13.0])
        assert np.max(hands) == 1.0
        assert sum(remaining_tiles[:34]) == 83

        halfturns = []
        # the initial dora indicator is public from the first decision on (kan doras arrive as DORA events)
        encoding_tokens = [new_dora2idx(TILE2IDX[int(dora)][0])]

        # more non-tensor state objects
        double_reaches = [0] * 4
        reaches = [0] * 4
        ippatsu = [0] * 4
        melds = [[] for _ in range(4)]
        turns = []
        is_menzen = [1] * 4
        curr_halfturn = None
        cycles = 0
        acquired = None
        result_idx = None
        last_draw = None  # (player, TILE2IDX[tile]) of the latest draw; None after a chi/pon/kan until the next draw
        riichi_declaring = False  # True from REACH step 1 until the declaration discard
        pending_call = None  # the chi / pon whose discard comes next (kuikae applies to it); None otherwise
        first_go_around_uninterrupted = True  # False after the first call of the round (any chi / pon / kan)
        own_discards = [0] * 4
        safe_tiles = np.zeros((4, 37), dtype=np.float32)  # safe_tiles[q]: tiles that cannot deal into q's riichi
        rinshan_pending = False  # True from a kan until the rinshan draw that follows it
        last_draw_rinshan = False  # the latest draw was a rinshan draw
        furiten = FuritenTracker()
        for p in range(4):
            furiten.set_hand(p, hand_tensors[p][:34])

        def ron_contexts(discarder: int, is_chankan: bool) -> dict:
            """Each responder's WinContext for a ron on the tile just discarded (or added by a kakan)."""
            return {
                p: WinContext(
                    is_tsumo=False,
                    seat_wind=seat_wind(p, oya),
                    round_wind=round_wind_,
                    is_riichi=bool(reaches[p]),
                    is_double_riichi=bool(double_reaches[p]),
                    is_ippatsu=bool(ippatsu[p]),
                    is_chankan=is_chankan,
                    is_houtei=_is_houtei(remaining_tsumo, is_chankan, last_draw_rinshan),
                )
                for p in range(4) if p != discarder
            }

        def make_state(p: int) -> StateObject:
            """Snapshot of the round as player p sees it right now.

            Every per-player array is COPIED (hand, remaining_tiles_pov, sutehai, melds): the round loop
            mutates these buffers in place, and tensors are built only after the whole round is parsed,
            so a view would show end-of-round values to every sample. Lists that the loop rebinds on
            change (reaches, ippatsu, double_reaches, scores, doras) are safe to share.
            """
            return StateObject(
                remaining_turns=remaining_tsumo,
                hand_tensor=hand_tensors[p].copy(),
                remaining_tiles=remaining_tiles.copy(),
                remaining_tiles_pov=remaining_tiles_pov[p].copy(),
                sutehai_tensors=np.copy(sutehai_tensors),
                is_menzen=is_menzen[:],
                double_reaches=double_reaches,
                reaches=reaches,
                ippatsu=ippatsu,
                melds=[m[:] for m in melds],
                scores=scores,
                kyotaku=kyotaku,
                honba=honba,
                dora=doras,
                parent_tensor=parent_tensor,
                rounds_remaining=rounds_remaining,
                parent_rounds_remaining=parent_rounds_remaining,
                remaining_tsumo=remaining_tsumo,
                player=p,
                drawn_tile=list(last_draw[1]) if last_draw is not None and last_draw[0] == p else None,
                safe_tiles=safe_tiles.copy(),
            )

        if verbose:
            print(f"Scores: {scores}")
            print(f"Parent: {parent}")
            print(f"Current round: {curr_round}")
            print(f"Honba: {honba}")
            print(f"Kyotaku: {kyotaku}")
            print(f"Remaining rounds: {rounds_remaining}")
            print(f"Parent rounds remaining: {parent_rounds_remaining}")
            print("\n" + "=" * 20 + "\n")

        for event_idx, event in enumerate(kyoku_events[1:]):
            eventtype = event["event"]

            if eventtype == "DORA":
                tile = int(event["attr"]["hai"])
                tile_idx = TILE2IDX[tile]  # a red five decrements its kind AND its red slot, as at INIT
                doras = doras[:] + [tile]
                remaining_tiles[tile_idx] -= 1
                for pov in remaining_tiles_pov:
                    pov[tile_idx] -= 1
                encoding_tokens.append(new_dora2idx(tile_idx[0]))  # the token stays on the kind index

            elif eventtype == "REACH":
                assert curr_halfturn is not None
                player = int(event["attr"]["who"])

                if event["attr"]["step"] == "1":
                    assert isinstance(curr_halfturn, DuringTurn) and curr_halfturn.player == player
                    assert acquired is not None

                    offered = curr_halfturn.decisions[DECISION_REACH_IDX]
                    assert len(offered) == 1 and isinstance(offered[0], ReachDecision), (
                        f"riichi by player {player} matched {len(offered)} offered riichi decisions"
                    )
                    offered[0].executed = True

                    # The declaration is public BEFORE the declaration discard: the player picks that
                    # tile knowing it is in riichi, and every responder sees the riichi token.
                    reaches = reaches[:]
                    reaches[player] = 1
                    if _double_riichi(first_go_around_uninterrupted, own_discards[player], cycles, is_menzen):
                        double_reaches = double_reaches[:]  # set with the declaration, like reaches
                        double_reaches[player] = 1
                    for kind in np.flatnonzero(sutehai_tensors[player, :34]):  # his own discards: furiten
                        _mark_safe(safe_tiles[player], kind)
                    encoding_tokens.append(reach2idx(player))
                    riichi_declaring = True

                else:  # step 2: the riichi stands (the declaration discard was not ronned); pay the stick
                    assert isinstance(curr_halfturn, PostTurn)
                    kyotaku += 1
                    scores = scores.copy()
                    scores[player] -= 10

                ippatsu = ippatsu[:]
                ippatsu[player] = 1

            elif eventtype == "N":
                assert curr_halfturn is not None
                acquired = None
                last_draw = None

                naki = Naki(int(event["attr"]["m"]))
                player = int(event["attr"]["who"])
                first_go_around_uninterrupted = False  # any call, an ankan included, ends double-riichi eligibility
                from_who = naki.from_who()
                cycles += (player - from_who) % 4
                melds[player] = melds[player][:] + [naki]
                if naki.is_chi() or naki.is_pon():
                    pending_call = naki  # the caller's discard follows directly: no draw in between
                else:
                    rinshan_pending = True  # any kan: the caller's next draw is the rinshan draw

                encoding_tokens.append(naki2idx(player, naki))

                # Label the executed call. Match on the ACTION INDEX the model predicts, never on the
                # raw Tenhou meld code: its `which`/unused-copy bits depend on tile ids the mask cannot know.
                if naki.is_ankan() or naki.is_kakan():
                    assert isinstance(curr_halfturn, DuringTurn) and curr_halfturn.player == player
                    target_idx = naki.get_during_turn_filter_idx()
                    offered = [
                        d for d in curr_halfturn.decisions[DECISION_NAKI_IDX]
                        if d.naki.get_during_turn_filter_idx() == target_idx
                    ]
                else:
                    assert isinstance(curr_halfturn, PostTurn)
                    target_idx = naki.get_post_turn_filter_idx()
                    offered = [
                        d for d in curr_halfturn.decisions[player][DECISION_NAKI_IDX]
                        if d.naki.get_post_turn_filter_idx() == target_idx
                    ]
                assert len(offered) == 1, (
                    f"call {naki!r} (idx {target_idx}) by player {player} matched {len(offered)} offered decisions"
                )
                offered[0].executed = True

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

                # update open-hand / ippatsu flags BEFORE the snapshot so the discard that follows
                # this call sees is_menzen[player] = 0 and ippatsu cleared. A kakan clears ippatsu only
                # AFTER its chankan PostTurn below: ippatsu still counts for a chankan ron.
                if naki.from_who() != 0:
                    is_menzen[player] = 0
                if sum(ippatsu) and not naki.is_kakan():
                    ippatsu = [0] * 4

                stateObj = make_state(player)

                # in case of Kakan, create new post turn to train on chankan ron decision
                if naki.is_kakan():
                    # remove corresponding pon from melds
                    for i, meld in enumerate(melds[player]):
                        if meld.is_pon():
                            color, number, *_ = meld.pattern_pon()
                            if color * 9 + number == exposed[0] // 4:
                                melds[player].pop(i)
                                break

                    new_post_decisions = [[[] for _ in range(3)] for _ in range(4)]

                    # evaluate chankan ron
                    kakan_kind = exposed[0] // 4
                    decisions, ron_blocked = evaluate_ron(
                        player, kakan_kind, hand_tensors, melds, furiten, ron_contexts(player, is_chankan=True)
                    )

                    for decision in decisions:
                        if decision is not None:
                            new_post_decisions[decision.player][DECISION_AGARI_IDX].append(decision)

                    curr_halfturn = PostTurn(
                        player=player,
                        stateObj=stateObj,
                        decisions=new_post_decisions,
                        encoding_idx=len(encoding_tokens),
                        responder_states={p: make_state(p) for p in range(4) if p != player},
                        ron_blocked=ron_blocked,
                    )

                    halfturns.append(curr_halfturn)
                    # the round goes on past the window only if every responder passed (a passed chankan counts)
                    furiten.on_offer_passed(kakan_kind, [p for p in range(4) if p != player], reaches)

                    if sum(ippatsu):  # the chankan window is closed: the kakan breaks every ippatsu
                        ippatsu = [0] * 4

                # no PostTurn after an ankan: tenhou.net/man 「他家が暗槓した牌では国士無双はあがれない」

                curr_halfturn = None

                if naki.is_minkan() or naki.is_ankan() or naki.is_kakan():
                    cycles += 1
                    acquired = None

            elif eventtype[0] in ["T", "U", "V", "W"]:
                assert acquired is None
                assert pending_call is None, f"a draw before the discard that follows {pending_call!r}"
                curr_halfturn = None

                player = ["T", "U", "V", "W"].index(eventtype[0])
                cycles += 1
                acquired = int(eventtype[1:])
                tile_idx = TILE2IDX[acquired]
                last_draw = (player, tile_idx)
                remaining_tiles[tile_idx] -= 1
                remaining_tiles_pov[player][tile_idx] -= 1
                remaining_tsumo -= 1  # the ONLY decrement: 70 draws per round, kans included (rinshan draw is a T event)
                assert remaining_tsumo >= 0, remaining_tsumo
                assert hands[player, acquired] == 0.0

                hands[player, acquired] = 1.0
                hand_tensors[player][tile_idx] += 1.0
                assert int(hand_tensors[player][:34].sum()) % 3 == 2

                # check if tsumo is possible
                is_rinshan = last_draw_rinshan = rinshan_pending
                rinshan_pending = False
                first_draw = first_go_around_uninterrupted and own_discards[player] == 0  # no call of any kind yet
                agari_decisions, tsumo_blocked = evaluate_tsumo(
                    player,
                    hand_tensors[player],
                    melds[player],
                    tile_idx[0],
                    WinContext(
                        is_tsumo=True,
                        seat_wind=seat_wind(player, oya),
                        round_wind=round_wind_,
                        is_riichi=bool(reaches[player]),
                        is_double_riichi=bool(double_reaches[player]),
                        is_ippatsu=bool(ippatsu[player]),
                        is_rinshan=is_rinshan,
                        is_haitei=_is_haitei(remaining_tsumo, is_rinshan),
                        is_tenhou=first_draw and player == oya,
                        is_chiihou=first_draw and player != oya,
                    ),
                )

                # check if reach is possible
                n_kans, kan_owners = legal.kans_on_table(melds)
                reach_decisions = []
                if (not reaches[player] and is_menzen[player]
                        and legal.riichi_allowed(scores[player], remaining_tsumo, n_kans, kan_owners)):
                    shanten = shanten_solver.calculate_shanten(hand_tensors[player][:34])
                    if shanten <= 0:
                        reach_decisions.append(ReachDecision(player, executed=False))

                # check if naki is possible
                naki_decisions = []
                kan_ok = legal.kan_allowed(remaining_tsumo, n_kans)
                # check if ankan is possible
                for i in np.where(hand_tensors[player] == 4.0)[0] if kan_ok else []:
                    if reaches[player] and not legal.riichi_ankan_allowed(
                        hand_tensors[player], tile_idx[0], int(i), melds[player]
                    ):
                        continue
                    naki_decisions.append(NakiDecision(player, Naki.from_ankan_info(i), executed=False))

                # check if kakan is possible
                for meld in melds[player] if kan_ok else []:
                    if meld.is_pon():
                        assert not reaches[player], f"player {player} in riichi holds a pon"
                        color, number, *_ = meld.pattern_pon()
                        if hand_tensors[player][9 * color + number] == 1.0:
                            naki_decisions.append(
                                NakiDecision(
                                    player,
                                    Naki.from_kakan_info(9 * color + number, 0),  # TODO get accurate "which" value
                                    executed=False,
                                )
                            )

                during_decisions = [
                    agari_decisions,
                    reach_decisions,
                    naki_decisions,
                ]

                stateObj = make_state(player)

                halfturn = DuringTurn(
                    player=player,
                    stateObj=stateObj,
                    decisions=during_decisions,
                    encoding_idx=len(encoding_tokens),
                    tsumo_blocked=tsumo_blocked,
                )

                halfturns.append(halfturn)
                curr_halfturn = halfturn  # carry over

                # the first draw after riichi ends the ippatsu window AFTER its tsumo decision and snapshot:
                # an ippatsu tsumo sees ippatsu = 1, the discard that follows sees 0
                if ippatsu[player]:
                    ippatsu = ippatsu[:]
                    ippatsu[player] = 0

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
                    stateObj=make_state(player),  # fresh: a DORA (after a kan) or REACH event can sit between the draw and this discard
                    discarded_tile=tile,
                    encoding_idx=len(encoding_tokens),
                    is_riichi_declaration=riichi_declaring,
                    after_call=pending_call,
                )
                riichi_declaring = False
                pending_call = None
                halfturns.append(half_turn)

                hands[player, tile] = 0.0
                hand_tensors[player][tile_idx] -= 1.0
                assert int(hand_tensors[player][:34].sum()) % 3 == 1, hand_tensors[player].sum()
                sutehai_tensors[player, tile_idx] = 1.0
                own_discards[player] += 1
                furiten.on_discard(player, tile_idx[0])
                furiten.set_hand(player, hand_tensors[player][:34])  # new waits; the own discard ends same-turn furiten
                if reaches[player]:  # his own discard is safe against himself at once: he cannot ron it
                    _mark_safe(safe_tiles[player], tile_idx[0])

                for pov in remaining_tiles_pov:
                    pov[tile_idx] -= 1.0
                remaining_tiles_pov[player][tile_idx] += 1.0

                # the discard is now public: responders must see it in their history
                encoding_tokens.append(discard2idx(player, tile_idx[-1], tile == acquired))  # red five -> slot 34-36

                # post_decisions will contain three lists of different decisions: ron, pon/kan, chi
                post_decisions = [[[] for _ in range(3)] for _ in range(4)]

                rons, ron_blocked = evaluate_ron(
                    player, tile_idx[0], hand_tensors, melds, furiten, ron_contexts(player, is_chankan=False)
                )

                for ron in rons:
                    if ron is not None:
                        post_decisions[ron.player][DECISION_AGARI_IDX].append(ron)

                naki_decisions = decision_mask(player, hand_tensors, tile_idx[0], len(tile_idx) == 2)
                n_kans, kan_owners = legal.kans_on_table(melds)
                calls_ok = legal.calls_allowed(remaining_tsumo, n_kans, kan_owners)

                for p, decision_lst in enumerate(naki_decisions):
                    if p == player:
                        assert len(decision_lst) == 0
                        continue
                    if reaches[p] or not calls_ok:  # no call by a riichi player, on the last discard, or with suukaikan pending
                        continue
                    post_decisions[p][DECISION_NAKI_IDX] = [
                        d for d in decision_lst
                        if (not d.naki.is_minkan() or legal.kan_allowed(remaining_tsumo, n_kans))
                        and (d.naki.is_minkan() or legal.call_leaves_legal_discard(hand_tensors[p], d.naki))
                    ]

                half_turn = PostTurn(
                    player=player,
                    stateObj=stateObj,
                    decisions=post_decisions,
                    encoding_idx=len(encoding_tokens),  # includes the discard token appended above
                    responder_states={p: make_state(p) for p in range(4) if p != player},
                    ron_blocked=ron_blocked,
                )
                halfturns.append(half_turn)
                curr_halfturn = half_turn  # carry over
                # the round goes on past the window only if every responder passed the ron
                furiten.on_offer_passed(tile_idx[0], [p for p in range(4) if p != player], reaches)

                # Only now, with the responder snapshots built, is the discard safe against the OTHER riichi
                # players (riichi furiten): a responder asked to ron this tile must not see it marked safe.
                # A responder who rons it ends the round, so marking every riichi player is exact.
                for q in range(4):
                    if reaches[q] and q != player:
                        _mark_safe(safe_tiles[q], tile_idx[0])

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
                result.append(AgariResult(list(map(int, event["attr"]["sc"].split(",")))))
                _mark_agari_executed(halfturns, int(event["attr"]["who"]), int(event["attr"]["fromWho"]))

            elif event["event"] == "RYUUKYOKU":
                result.append(RyukyokuResult(list(map(int, event["attr"]["sc"].split(",")))))
                if event["attr"].get("type") == "ron3":
                    _mark_ron3_executed(halfturns)

        if verbose:
            print(f"len(turns): {len(turns)}")
            print(result)
            print(f"remaining_tiles: {remaining_tiles}")
            print(f"player hands: {hand_tensors}")

        assert sum([1 for r in remaining_tiles if r < 0]) == 0
        assert sum([1 for rem in remaining_tiles_pov for r in rem if r < 0]) == 0

        all_halfturns.append(halfturns)
        all_encoding_tokens.append(encoding_tokens)

    return all_halfturns, all_encoding_tokens


def _mark_agari_executed(halfturns: list, who: int, from_who: int) -> None:
    """Label the win that an AGARI event records.

    Tsumo (who == fromWho): the last halfturn is the winner's DuringTurn (also after a rinshan draw).
    Ron (who != fromWho): the last halfturn is a PostTurn whose player is the loser; this covers
    chankan (the PostTurn built in the kakan branch) and double ron (two AGARI events, same PostTurn).
    """
    assert halfturns, "AGARI event with no halfturns"
    last = halfturns[-1]
    if who == from_who:
        assert isinstance(last, DuringTurn) and last.player == who, f"tsumo by {who}, last halfturn: {last!r}"
        agari = last.decisions[DECISION_AGARI_IDX]
    else:
        assert isinstance(last, PostTurn) and last.player == from_who, f"ron by {who} from {from_who}, last halfturn: {last!r}"
        agari = last.decisions[who][DECISION_AGARI_IDX]
    assert len(agari) == 1, f"win by player {who} was not offered by evaluate_ron/evaluate_tsumo: {last!r}"
    agari[0].executed = True


def _mark_ron3_executed(halfturns: list) -> None:
    """Label the three rons of a triple-ron abort (<RYUUKYOKU type="ron3">): no AGARI event follows, and the
    round ends on the tile the three players declared, so the FINAL halfturn is that PostTurn and all three
    responders must hold a ron offer there."""
    assert halfturns, "ron3 with no halfturns"
    last = halfturns[-1]
    assert isinstance(last, PostTurn), f"ron3, last halfturn: {last!r}"
    marked = 0
    for p in range(4):
        if p != last.player and last.decisions[p][DECISION_AGARI_IDX]:
            last.decisions[p][DECISION_AGARI_IDX][0].executed = True
            marked += 1
    assert marked == 3, f"ron3 marked {marked} rons on {last!r}"


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


def discard2idx(who: int, tile_idx: int, is_tsumogiri: bool):
    assert 0 <= tile_idx < 37
    return EventStateTypes.DISCARD << 13 | EventTypes.DISCARD << 10 | who << 8 | int(is_tsumogiri) << 7 | tile_idx


def naki2idx(who: int, naki: Naki):
    assert 0 <= who < 4
    embd_token = None

    if naki.is_chi():
        post_turn_filter_idx = naki.get_post_turn_filter_idx()
        embd_token = EventStateTypes.POST << 13 | EventTypes.NAKI << 10 | who << 8 | post_turn_filter_idx
    elif naki.is_pon():
        post_turn_filter_idx = naki.get_post_turn_filter_idx()
        embd_token = EventStateTypes.POST << 13 | EventTypes.NAKI << 10 | who << 8 | post_turn_filter_idx
    elif naki.is_kakan():
        during_turn_filter_idx = naki.get_during_turn_filter_idx()
        embd_token = EventStateTypes.DURING << 13 | EventTypes.NAKI << 10 | who << 8 | during_turn_filter_idx
    elif naki.is_minkan():
        post_turn_filter_idx = naki.get_post_turn_filter_idx()
        embd_token = EventStateTypes.POST << 13 | EventTypes.NAKI << 10 | who << 8 | post_turn_filter_idx
    elif naki.is_ankan():
        during_turn_filter_idx = naki.get_during_turn_filter_idx()
        embd_token = EventStateTypes.DURING << 13 | EventTypes.NAKI << 10 | who << 8 | during_turn_filter_idx
    else:
        raise ValueError("Invalid naki code")

    assert embd_token <= (1 << 17)
    return embd_token


def reach2idx(who: int):
    assert 0 <= who < 4
    embd_token = EventStateTypes.DURING << 13 | EventTypes.RIICHI << 10 | who << 8 | 2
    assert embd_token <= (1 << 17)
    return embd_token


def new_dora2idx(tile_idx: int):  # tile_idx should be 0-36
    assert 0 <= tile_idx < 37
    embd_token = EventStateTypes.NEW_DORA << 13 | EventTypes.NEW_DORA << 10 | tile_idx
    assert embd_token <= (1 << 17)
    return embd_token


if __name__ == "__main__":
    # import cProfile

    file_path = "data/sample/sample_haifu.xml"
    # cProfile.run("process(file_path, verbose=True)", sort="tottime")
    all_halfturns, all_encoding_tokens = process(file_path, verbose=True)

    print(list(map(len, all_halfturns)))
