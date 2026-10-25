"""Label-free inference on one decision point (T-052, plan section A "Model inference").

Sample selection is separated from tensor construction: `build_sample` builds the tensors of ANY (halfturn, seat)
the parser offers a decision to, including the rows the training builders drop (a forced riichi discard, a responder
pre-empted by a higher-priority call, a during point with no option). The tensors themselves come from the training
code: `_convert_state_obj_to_tensors`, `_relative_seats`, the history cut at `encoding_idx`, `_discard_mask`.
`predict` runs the network steps of `EncoderConvModel.forward` on one sample in eval mode and returns the masked
logits, so G1 can compare logits with the training forward.
"""
import argparse
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from MahjongAI.agents.encoder_conv import (
    DISCARD_ACTION_DIM,
    DURING_ABORT_IDX,
    DURING_TURN_ACTION_DIM,
    POST_TURN_ACTION_DIM,
    EncoderConvModel,
    TransformerTensorProcessor,
    device,
)
from MahjongAI.turn import DiscardTurn, DuringTurn, HalfTurn, PostTurn
from MahjongAI.utils.constants import (
    DECISION_ABORT_IDX,
    DECISION_AGARI_IDX,
    DECISION_NAKI_IDX,
    DECISION_REACH_IDX,
    MAX_SEQUENCE_LENGTH,
)

HEAD_OF = {HalfTurn.DURING: "during", HalfTurn.DISCARD: "discard", HalfTurn.POST: "post"}
ACTION_DIM = {"during": DURING_TURN_ACTION_DIM, "discard": DISCARD_ACTION_DIM, "post": POST_TURN_ACTION_DIM}


class Sample(NamedTuple):
    """One model input: the head, the seat that decides, and the (1, ...) tensors the forward consumes."""
    head: str
    seat: int
    tokens: torch.Tensor  # (1, 150) int64, seats relative to `seat`
    x: tuple  # (x1 (1, N_X1_PLANES, 37, 4), x2 (1, 7, 4), x3 (1, N_X3))
    mask: torch.Tensor  # (1, A) float, 1 = legal
    padding: torch.Tensor  # (1, 150) bool, True = valid position


_processor = TransformerTensorProcessor()


def _history(tokens: list, encoding_idx: int, seat: int) -> tuple:
    """(tokens (1, 150), padding (1, 150)): the round history cut at `encoding_idx`, seats relative to `seat`."""
    assert encoding_idx < MAX_SEQUENCE_LENGTH, encoding_idx
    t = torch.zeros(MAX_SEQUENCE_LENGTH, dtype=torch.int64)
    t[:encoding_idx] = torch.tensor(tokens[:encoding_idx], dtype=torch.int64)
    t = _processor._relative_seats(t.unsqueeze(0), seat)
    padding = (torch.arange(MAX_SEQUENCE_LENGTH) < encoding_idx).unsqueeze(0)
    return t, padding


def during_mask(turn: DuringTurn) -> torch.Tensor:
    """(72,) mask of the during head: pass, plus every offered decision (as the training builder sets it)."""
    mask = torch.zeros(DURING_TURN_ACTION_DIM, dtype=torch.float32)
    mask[0] = 1.0
    if turn.decisions[DECISION_AGARI_IDX]:
        mask[1] = 1.0
    if turn.decisions[DECISION_REACH_IDX]:
        mask[2] = 1.0
    for d in turn.decisions[DECISION_NAKI_IDX]:
        mask[d.naki.get_during_turn_filter_idx()] = 1.0
    if turn.decisions[DECISION_ABORT_IDX]:
        mask[DURING_ABORT_IDX] = 1.0
    return mask


def during_executed(turn: DuringTurn) -> int:
    """The during-head index the drawer executed (0 = pass: the drawer went on to discard)."""
    if any(d.executed for d in turn.decisions[DECISION_AGARI_IDX]):
        return 1
    if any(d.executed for d in turn.decisions[DECISION_REACH_IDX]):
        return 2
    for d in turn.decisions[DECISION_NAKI_IDX]:
        if d.executed:
            return d.naki.get_during_turn_filter_idx()
    if any(d.executed for d in turn.decisions[DECISION_ABORT_IDX]):
        return DURING_ABORT_IDX
    return 0


def post_mask(turn: PostTurn, seat: int) -> Optional[torch.Tensor]:
    """(154,) mask of responder `seat`: pass plus its offered ron / calls; None when nothing was offered."""
    agari, _, naki = turn.decisions[seat]
    if not agari and not naki:
        return None
    mask = torch.zeros(POST_TURN_ACTION_DIM, dtype=torch.float32)
    mask[0] = 1.0
    if agari:
        mask[1] = 1.0
    for d in naki:
        mask[d.naki.get_post_turn_filter_idx()] = 1.0
    return mask


def post_executed(turn: PostTurn, seat: int) -> int:
    """The post-head index responder `seat` executed (0 = none)."""
    agari, _, naki = turn.decisions[seat]
    if any(d.executed for d in agari):
        return 1
    for d in naki:
        if d.executed:
            return d.naki.get_post_turn_filter_idx()
    return 0


def forced_riichi_discard(turn: DiscardTurn) -> bool:
    """A discard by a player in riichi that is not the declaration: the drawn tile is the only legal discard."""
    return bool(turn.stateObj.reaches[turn.player]) and not turn.is_riichi_declaration


def discard_mask(turn: DiscardTurn) -> torch.Tensor:
    """(37,) legal discard mask; a forced riichi discard keeps the drawn tile's slot only (red slot for a red five)."""
    if forced_riichi_discard(turn):
        drawn = turn.stateObj.drawn_tile
        assert drawn is not None, f"riichi discard by {turn.player} without a drawn tile"
        mask = torch.zeros(DISCARD_ACTION_DIM, dtype=torch.float32)
        mask[drawn[-1]] = 1.0
        return mask
    mask = _processor._discard_mask(turn)
    assert mask.any(), f"no legal discard for player {turn.player}: {turn.stateObj.hand_tensor.tolist()}"
    return mask


def build_sample(halfturn: HalfTurn, tokens: list, seat: int) -> Optional[Sample]:
    """The model input of `seat` at `halfturn`, or None when `seat` has no decision there (a during / discard point
    belongs to its actor; a post point to every responder with an offer)."""
    head = HEAD_OF[halfturn.type_]
    if head == "post":
        if seat == halfturn.player or seat not in halfturn.responder_states:
            return None
        mask = post_mask(halfturn, seat)
        if mask is None:
            return None
        state = halfturn.responder_states[seat]
    else:
        if seat != halfturn.player:
            return None
        mask = during_mask(halfturn) if head == "during" else discard_mask(halfturn)
        state = halfturn.stateObj
    history, padding = _history(tokens, halfturn.encoding_idx, seat)
    x1, x2, x3 = _processor._convert_state_obj_to_tensors(state)
    return Sample(head, seat, history, (x1, x2, x3), mask.unsqueeze(0), padding)


def deciding_seats(halfturn: HalfTurn) -> list:
    """The seats `build_sample` returns a sample for, in turn order after the actor for a post point."""
    if halfturn.type_ == HalfTurn.POST:
        order = [(halfturn.player + k) % 4 for k in (1, 2, 3)]
        return [p for p in order if post_mask(halfturn, p) is not None]
    return [halfturn.player]


@torch.no_grad()
def predict(model: EncoderConvModel, sample: Sample) -> tuple:
    """(mask (A,), masked logits (A,), probs (A,)) of one sample: the network steps of `forward` in eval mode."""
    model.eval()
    history_emb = model.token_embedder(sample.tokens.to(device))
    state_emb = model.state_net(tuple(t.to(device) for t in sample.x)).unsqueeze(1)
    fused = torch.cat([history_emb, state_emb], dim=1)
    padding = sample.padding.to(device)
    fused_mask = torch.cat([padding, torch.ones(1, 1, dtype=torch.bool, device=padding.device)], dim=1)
    attn_mask = fused_mask.unsqueeze(1)
    x = fused
    for i, block in enumerate(model.blocks):
        x = block(x, attn_mask, state_row_only=(i == len(model.blocks) - 1))
    final_state = model.ln_f(x[:, -1, :])
    logits = model.heads[sample.head](final_state)
    mask = sample.mask.to(device)
    logits = logits.masked_fill(mask == 0, -1e9)
    probs = F.softmax(logits, dim=1)
    return mask[0].cpu(), logits[0].cpu(), probs[0].cpu()


def load_model(ckpt_path: str) -> EncoderConvModel:
    """The checkpoint's model in eval mode (supervised_learning.py's load: weights_only; the architecture from its
    config, EncoderConvModel.from_checkpoint)."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = EncoderConvModel.from_checkpoint(checkpoint).to(device)
    model.eval()
    return model


def main(argv=None):
    """Print the model's top action per decision point of one log (a smoke run; the exporter writes the JSON)."""
    from MahjongAI.process_xml import process

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("log")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args(argv)
    model = load_model(args.ckpt)
    halfturns, tokens = process(args.log)
    for i, turn in enumerate(halfturns[args.round]):
        for seat in deciding_seats(turn):
            sample = build_sample(turn, tokens[args.round], seat)
            mask, logits, probs = predict(model, sample)
            top = int(probs.argmax())
            print(f"{i:4d} {sample.head:7s} seat {seat} top {top:3d} p={probs[top]:.3f} legal={int(mask.sum())}")


if __name__ == "__main__":
    main()
