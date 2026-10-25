"""Players: `Player = Callable[[HalfTurn, tokens, seat], Choice]` (the chess play.py player functions).

ModelPlayer runs the EncoderConvModel on one pending halfturn: `build_sample` builds the three inputs exactly as
the training tensor builders do (agents/encoder_conv.py TransformerTensorProcessor: relative-seat history, state
planes, the head's legal mask), `predict` runs the forward pass of EncoderConvModel.forward (steps 1-6) on that
one sample. T-052's viewer/infer.py has the same two-function shape; phase 2 swaps it in. T=0 plays the argmax,
T>0 samples softmax(logits / T) with one seeded torch.Generator per player (the chess seeded sampling).
GreedyPlayer: the shanten-greedy discard of utils/baselines.py, takes a tsumo / ron / riichi, never calls.
UniformPlayer: a seeded uniform pick among the legal actions.
Identity: a ModelPlayer is identified by its checkpoint's sha256 (label `model:<file>@<sha256[:12]>`), never by
the file name alone (two runs both have a step_100.pt); `describe()` is the complete per-seat specification that
meta.json stores and `from_description` rebuilds for the replay audit.
"""
import functools
import hashlib
import os
import random
import time
from typing import Callable, Dict, List, NamedTuple, Tuple

import numpy as np
import torch

from MahjongAI.agents.encoder_conv import EncoderConvModel, TransformerTensorProcessor
from MahjongAI.turn import DiscardTurn, DuringTurn, HalfTurn, PostTurn
from MahjongAI.utils.baselines import shanten_after
from MahjongAI.utils.constants import MAX_SEQUENCE_LENGTH
from MahjongAI.engine.round import (Choice, DURING_PASS, DURING_RIICHI, DURING_TSUMO, POST_PASS, POST_RON,
                                    legal_mask)

Player = Callable[[HalfTurn, List[int], int], Choice]
_processor = TransformerTensorProcessor()


class Sample(NamedTuple):
    head: str
    history: torch.Tensor  # (1, 150) int64, seats relative to the actor
    state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mask: torch.Tensor  # (A,) float, 1 = legal
    padding: torch.Tensor  # (1, 150) bool, True = a real token


def head_of(halfturn: HalfTurn) -> str:
    if isinstance(halfturn, DuringTurn):
        return "during"
    if isinstance(halfturn, DiscardTurn):
        return "discard"
    assert isinstance(halfturn, PostTurn)
    return "post"


def build_sample(halfturn: HalfTurn, tokens: List[int], seat: int) -> Sample:
    """The model inputs of one pending halfturn for `seat` (a PostTurn: the responder's own state)."""
    head = head_of(halfturn)
    state_obj = halfturn.responder_states[seat] if head == "post" else halfturn.stateObj
    assert head == "post" or halfturn.player == seat, (halfturn, seat)
    idx = halfturn.encoding_idx
    assert idx <= len(tokens) and idx < MAX_SEQUENCE_LENGTH, (idx, len(tokens))
    history = torch.zeros(1, MAX_SEQUENCE_LENGTH, dtype=torch.int64)
    history[0, :idx] = torch.tensor(tokens[:idx], dtype=torch.int64)
    history = _processor._relative_seats(history, seat)
    padding = torch.arange(MAX_SEQUENCE_LENGTH).unsqueeze(0) < idx
    mask = torch.from_numpy(legal_mask(halfturn, seat))
    assert mask.any(), f"no legal action for seat {seat} at {halfturn!r}"
    return Sample(head, history, _processor._convert_state_obj_to_tensors(state_obj), mask, padding)


@torch.no_grad()
def predict(model: EncoderConvModel, sample: Sample) -> torch.Tensor:
    """The head's masked logits (A,) for one sample: EncoderConvModel.forward steps 1-6 without the loss."""
    history_emb = model.token_embedder(sample.history)
    state_emb = model.state_net(sample.state).unsqueeze(1)
    x = torch.cat([history_emb, state_emb], dim=1)
    fused_mask = torch.cat([sample.padding, torch.ones(1, 1, dtype=torch.bool)], dim=1)
    attn_mask = fused_mask.unsqueeze(1)
    for i, block in enumerate(model.blocks):
        x = block(x, attn_mask, state_row_only=(i == len(model.blocks) - 1))
    logits = model.heads[sample.head](model.ln_f(x[:, -1, :]))[0]
    return logits.masked_fill(sample.mask == 0, -1e9)


@functools.lru_cache(maxsize=None)
def load_model(ckpt_path: str) -> EncoderConvModel:
    """Memoised per process: every seat gets its own ModelPlayer (own sampling stream) over one set of weights."""
    model = EncoderConvModel.from_checkpoint(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@functools.lru_cache(maxsize=None)
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelPlayer:
    def __init__(self, ckpt_path: str, temperature: float = 0.0, seed: int = 0):
        assert temperature >= 0.0, temperature
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.model = load_model(self.ckpt_path)
        self.temperature = float(temperature)
        self.generator = torch.Generator()
        self.seed = int(seed)
        self.reseed(seed)
        self.checkpoint_sha256 = sha256_of(self.ckpt_path)
        self.label = f"model:{os.path.basename(ckpt_path)}@{self.checkpoint_sha256[:12]}"

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.generator.manual_seed(int(seed))

    def describe(self) -> Dict[str, object]:
        return {"kind": "model", "label": self.label, "checkpoint": self.ckpt_path,
                "checkpoint_sha256": self.checkpoint_sha256, "temperature": self.temperature, "seed": self.seed}

    def __call__(self, halfturn: HalfTurn, tokens: List[int], seat: int) -> Choice:
        t0 = time.perf_counter()
        sample = build_sample(halfturn, tokens, seat)
        logits = predict(self.model, sample)
        legal = sample.mask.nonzero().flatten().tolist()
        if self.temperature == 0.0:
            probs = torch.softmax(logits, dim=0)
            action = int(torch.argmax(logits))
            sampled = False
        else:
            probs = torch.softmax(logits / self.temperature, dim=0)
            action = int(torch.multinomial(probs, 1, generator=self.generator))
            sampled = True
        candidates = {a: float(probs[a]) for a in legal}
        return Choice(action, candidates, sampled, (time.perf_counter() - t0) * 1000.0)


class GreedyPlayer:
    """Shanten-greedy: the legal discard leaving the lowest shanten (ties: the lowest slot), tsumo / riichi when
    offered, ron when offered, never a call."""
    label = "greedy"

    def reseed(self, seed: int) -> None:
        pass

    def describe(self) -> Dict[str, object]:
        return {"kind": "greedy", "label": self.label}

    def __call__(self, halfturn: HalfTurn, tokens: List[int], seat: int) -> Choice:
        t0 = time.perf_counter()
        mask = legal_mask(halfturn, seat)
        legal = np.flatnonzero(mask).tolist()
        head = head_of(halfturn)
        if head == "discard":
            hand = halfturn.stateObj.hand_tensor
            action = min(legal, key=lambda slot: (shanten_after(hand, slot), slot))
        elif head == "during":
            action = next(a for a in (DURING_TSUMO, DURING_RIICHI, DURING_PASS) if a in legal)
        else:
            action = POST_RON if POST_RON in legal else POST_PASS
        return Choice(action, {a: float(a == action) for a in legal}, False, (time.perf_counter() - t0) * 1000.0)


class UniformPlayer:
    label = "uniform"

    def __init__(self, seed: int = 0):
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def describe(self) -> Dict[str, object]:
        return {"kind": "uniform", "label": self.label, "seed": self.seed}

    def __call__(self, halfturn: HalfTurn, tokens: List[int], seat: int) -> Choice:
        t0 = time.perf_counter()
        legal = np.flatnonzero(legal_mask(halfturn, seat)).tolist()
        action = self.rng.choice(legal)
        return Choice(action, {a: 1.0 / len(legal) for a in legal}, True, (time.perf_counter() - t0) * 1000.0)


def make_player(spec: str, temperature: float = 0.0, seed: int = 0):
    """spec: 'greedy', 'uniform', or a checkpoint path (optionally 'model:<path>')."""
    if spec == "greedy":
        return GreedyPlayer()
    if spec == "uniform":
        return UniformPlayer(seed)
    path = spec[len("model:"):] if spec.startswith("model:") else spec
    return ModelPlayer(path, temperature, seed)


def from_description(d: Dict[str, object]):
    """The player a meta.json `seats` entry describes, rebuilt exactly (the replay audit); a model whose
    checkpoint is missing or whose sha256 differs raises instead of substituting another model."""
    if d["kind"] == "greedy":
        return GreedyPlayer()
    if d["kind"] == "uniform":
        return UniformPlayer(int(d["seed"]))
    assert d["kind"] == "model", d
    path = str(d["checkpoint"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint {path} of {d['label']} is missing: the replay cannot rebuild the seat")
    player = ModelPlayer(path, float(d["temperature"]), int(d["seed"]))
    if player.checkpoint_sha256 != d["checkpoint_sha256"]:
        raise ValueError(f"checkpoint {path} has sha256 {player.checkpoint_sha256[:12]}, meta.json says "
                         f"{str(d['checkpoint_sha256'])[:12]}")
    return player
