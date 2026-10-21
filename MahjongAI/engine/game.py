"""One hanchan: rounds in sequence, the round-transition table, and the game's file set.

Round-transition table [C8] (Tenhou hanchan, tenhou.net/man), in order, applied by `next_round`:
  1  a result ends the round
  2  dealer continuation (dealer win, dealer tenpai at an exhaustive draw, any abort): same dealer, honba + 1;
     else the deal passes; honba + 1 after a draw, 0 after a win
  3  tobi: a score below 0 ends the game (0 continues)
  4  South 4 or later: the deal passes -> the game ends when the leader has >= 30,000, or after West 4; else
     an extension round (West 1 ...)
  5  agari-yame / tenpai-yame in South 4 and in every West round: a dealer win or dealer tenpai ends the game
     only when the dealer is first with >= 30,000 (an abort never ends it); a dealer continuation with another
     seat at >= 30,000 goes on (real log 2021081622gm-00a9-0000-17dc8b4d: West 1 continues after dealer tenpai
     with scores [243,183,305,259]); engine_lib check_corpus: 58,063 real round transitions, 0 disagreements
  6  placement ties: the initial seat order
  7  unclaimed riichi sticks: to the first place (they appear in `owari`)
The log is `<mjloggm ver="2.3">` with GO type 169 (4-player red-five kuitan hanchan: the training data's rule
set), as the parser needs. Saved as `<out>/game.log` + `decisions.json` (the sidecar [C11]) + `meta.json`.
"""
import json
import os
import random
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Sequence

from MahjongAI.gameinfo import GameInfo
from MahjongAI.engine import scoring
from MahjongAI.engine.round import Round, RoundOutcome, RoundSpec
from MahjongAI.engine.wall import Wall

GO_TYPE = 169
START_SCORE = 250
END_SCORE = 300  # 30,000: the hanchan ends past South 4 once the leader (or, in an extension, anyone) has it
ALL_LAST = 7  # South 4
LAST_ROUND = 11  # West 4
RULES = {"players": 4, "length": "hanchan", "red_fives": True, "kuitan": True, "start": 25000, "return": 30000,
         "uma": [20, 10, -10, -20], "tobi": "below zero", "extension": "West, until someone has 30000, max West 4",
         "pao": "daisangen, daisuushii", "nagashi_mangan": True, "double_yakuman": False, "kazoe_yakuman": True}


def next_round(spec: RoundSpec, outcome: RoundOutcome) -> Optional[RoundSpec]:
    """The next round's spec, or None when the game ends (the transition table above)."""
    scores = list(outcome.scores)
    if min(scores) < 0:
        return None
    renchan = outcome.renchan
    honba = spec.honba + 1 if (renchan or outcome.kind != "agari") else 0
    leader = scoring.rank_order(scores)[0]
    if spec.index >= ALL_LAST:
        if renchan:
            if outcome.kind != "abort" and leader == spec.oya and scores[leader] >= END_SCORE:
                return None
        elif scores[leader] >= END_SCORE or spec.index == LAST_ROUND:
            return None
    return RoundSpec(spec.index if renchan else spec.index + 1, honba, outcome.kyotaku, tuple(scores))


def final_scores(scores: Sequence[int], kyotaku: int) -> List[int]:
    final = list(scores)
    final[scoring.rank_order(final)[0]] += scoring.RIICHI_STICK * kyotaku
    return final


def placements(final: Sequence[int]) -> List[int]:
    """placements[seat] = 1-4."""
    out = [0] * 4
    for rank, q in enumerate(scoring.rank_order(final)):
        out[q] = rank + 1
    return out


def header(labels: Sequence[str], seed: int) -> str:
    names = " ".join(f'n{q}="{urllib.parse.quote(labels[q], safe="")}"' for q in range(4))
    return (f'<mjloggm ver="2.3"><SHUFFLE seed="python-random,{seed}" ref=""/><GO type="{GO_TYPE}" lobby="0"/>'
            f'<UN {names} dan="0,0,0,0" rate="1500.00,1500.00,1500.00,1500.00" sx="M,M,M,M"/><TAIKYOKU oya="0"/>')


def git_head(path: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class GameRecord:
    xml: str
    decisions: List[dict]
    meta: dict
    outcomes: List[RoundOutcome]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "game.log"), "w", encoding="utf-8") as f:
            f.write(self.xml)
        with open(os.path.join(out_dir, "decisions.json"), "w", encoding="utf-8") as f:
            json.dump(self.decisions, f)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)


class Game:
    """players[seat] plays seat `seat` for the whole game; `seed` fixes the walls (one random.Random stream) and
    is also what the players were reseeded with by the caller. meta.json carries `seats`: the complete player
    specification per seat (players.py `describe`: kind, checkpoint path + sha256, temperature, seed), which the
    replay audit rebuilds exactly (players.from_description)."""

    def __init__(self, players: Sequence, labels: Sequence[str], seed: int, temperature: float = 0.0):
        assert len(players) == 4 and len(labels) == 4
        self.players = list(players)
        self.labels = list(labels)
        self.seed = int(seed)
        self.temperature = temperature
        self.gameinfo = GameInfo(GO_TYPE)

    def play(self) -> GameRecord:
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        spec: Optional[RoundSpec] = RoundSpec(0, 0, 0, (START_SCORE,) * 4)
        rounds_xml, decisions, outcomes = [], [], []
        k = 0
        final = None
        while spec is not None:
            wall = Wall.shuffled(rng)
            dice = (rng.randrange(6), rng.randrange(6))
            rnd = Round(spec, wall, dice, self.gameinfo, self.labels, k0=k, kyoku=len(outcomes))
            outcome = self._run(rnd)
            k = rnd.k
            decisions += rnd.decisions
            outcomes.append(outcome)
            nxt = next_round(spec, outcome)
            if nxt is None:
                final = final_scores(outcome.scores, outcome.kyotaku)
                rnd.events[-1][1]["owari"] = scoring.owari_attr(final)
            rounds_xml.append(rnd.xml())
            spec = nxt
        xml = header(self.labels, self.seed) + "".join(rounds_xml) + "</mjloggm>"
        meta = {"generated": True, "engine": "MahjongAI.engine (T-053)", "players": self.labels, "seed": self.seed,
                "temperature": self.temperature, "git_head": git_head(os.path.dirname(os.path.abspath(__file__))),
                "seats": [p.describe() for p in self.players], "rules": RULES, "go_type": GO_TYPE,
                "rounds": len(outcomes), "final_scores": final,
                "placements": placements(final), "n_decisions": len(decisions),
                "duration_s": round(time.perf_counter() - t0, 3)}
        return GameRecord(xml, decisions, meta, outcomes)

    def _run(self, rnd: Round) -> RoundOutcome:
        gen = rnd.play()
        try:
            point = next(gen)
            while True:
                choice = self.players[point.seat](point.halfturn, point.tokens, point.seat)
                point = gen.send(choice)
        except StopIteration as stop:
            return stop.value
