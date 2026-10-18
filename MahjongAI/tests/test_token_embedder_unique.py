"""T-047 M2: TokenEmbedder on the unique (token, position) pairs, gathered back, computes the 468d71c full-size
embedding: forward bitwise equal, every table gradient equal in float64, on real batches, on heads with few unique
rows and on a batch with padding in some rows only; the whole model's logits are bitwise equal in eval."""
import copy
import glob
import os

import numpy as np
import pytest
import torch

from MahjongAI.agents.encoder_conv import (DURING_TURN_ACTION_DIM, MAX_ACTION_LEN, POST_TURN_ACTION_DIM,
                                           EncoderConvModel, TokenEmbedder, TransformerTensorProcessor)
from MahjongAI.process_xml import process
from MahjongAI.utils.constants import EventStateTypes

ATOL, RTOL = 1e-6, 1e-5


class ReferenceTokenEmbedder(TokenEmbedder):
    """TokenEmbedder.forward of 468d71c, verbatim apart from comments: every (B, T) slot computed."""

    def forward(self, encoding_tokens_batch):
        event_type = (encoding_tokens_batch >> 13) & 7
        embedding_type = (encoding_tokens_batch >> 10) & 7
        player_idx = (encoding_tokens_batch >> 8) & 3
        tile_encoding = encoding_tokens_batch & 0x7F
        tsumogiri = (encoding_tokens_batch >> 7) & 1
        naki_idx = encoding_tokens_batch & 0xFF
        adjusted_embedding_type = torch.clamp(embedding_type - 1, min=0)
        type_emb = self.type_embedding(adjusted_embedding_type)
        mask = event_type != EventStateTypes.EMPTY
        type_emb = type_emb * mask.unsqueeze(-1).float()
        adjusted_tile_encoding = torch.clamp(tile_encoding, max=36)
        tile_emb = self.tile_embedding(adjusted_tile_encoding)
        mask = (event_type == EventStateTypes.DISCARD) | (event_type == EventStateTypes.NEW_DORA)
        tile_emb = tile_emb * mask.unsqueeze(-1).float()
        player_idx = player_idx.to(self.player_coding_table.device)
        player_emb = self.player_coding_table[player_idx]
        mask = (
            (event_type == EventStateTypes.DISCARD)
            | (event_type == EventStateTypes.DURING)
            | (event_type == EventStateTypes.POST)
        ).to(player_emb.device)
        player_emb = player_emb * mask.unsqueeze(-1).float()
        tsumogiri_emb = self.tsumogiri_embedding(torch.tensor([0], device=self.type_embedding.weight.device))
        mask = (event_type == EventStateTypes.DISCARD) & (tsumogiri == 1)
        tsumogiri_emb = tsumogiri_emb * mask.unsqueeze(-1).float()
        pos_emb = self.position_coding_table[: encoding_tokens_batch.size(1)]
        pos_emb = pos_emb.unsqueeze(0)
        mask = event_type != EventStateTypes.EMPTY
        pos_emb = pos_emb * mask.unsqueeze(-1).float()
        adjusted_during_naki_idx = torch.clamp(naki_idx, max=DURING_TURN_ACTION_DIM - 1)
        during_naki_emb = self.during_naki_embedding(adjusted_during_naki_idx)
        mask = (event_type == EventStateTypes.DURING) & (naki_idx >= 3)
        during_naki_emb = during_naki_emb * mask.unsqueeze(-1).float()
        adjusted_post_naki_idx = torch.clamp(naki_idx, max=POST_TURN_ACTION_DIM - 1)
        post_naki_emb = self.post_naki_embedding(adjusted_post_naki_idx)
        mask = (event_type == EventStateTypes.POST) & (naki_idx >= 2)
        post_naki_emb = post_naki_emb * mask.unsqueeze(-1).float()
        x = (
            self.empty_embedding
            + type_emb
            + tile_emb
            + player_emb
            + tsumogiri_emb
            + pos_emb
            + during_naki_emb
            + post_naki_emb
        )
        return x


def _reference_of(module):
    ref = copy.deepcopy(module)
    for m in ref.modules():
        if type(m) is TokenEmbedder:
            m.__class__ = ReferenceTokenEmbedder
    assert [n for n, _ in ref.named_parameters()] == [n for n, _ in module.named_parameters()]
    return ref


GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
FIXTURE_LOGS = ("2021010100gm-00a9-0000-05d6c581.log", "2021010313gm-00a9-0000-5120e1eb.log")


@pytest.fixture(scope="module")
def games():
    halfturns, tokens = [], []
    for name in FIXTURE_LOGS:
        found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
        assert len(found) == 1, (name, found)
        h, t = process(found[0])
        halfturns.append(h)
        tokens.append(t)
    return halfturns, tokens


@pytest.fixture(scope="module")
def head_tokens(games):
    """The (B, 150) token matrix of each head on the two real games."""
    tensors = TransformerTensorProcessor().prepare_batches(*games)
    return {h: t[0] for h, t in zip(("during", "discard", "post"), tensors)}


def _constructed_batches(head_tokens):
    """Heads with few unique rows (one history repeated; every row empty) and a batch with padding in some rows only
    (rows of full length 150 next to padded rows)."""
    discard = head_tokens["discard"]
    lengths = (discard != 0).sum(1)
    full_row = discard[lengths.argmax()].clone()
    full_row[full_row == 0] = full_row[0]  # no padding left in this row
    assert (full_row != 0).all()
    mixed = discard[:6].clone()
    mixed[1], mixed[4] = full_row, full_row
    return {"repeated_history": discard[:1].repeat(9, 1), "all_empty": torch.zeros(5, MAX_ACTION_LEN, dtype=discard.dtype),
            "padding_in_some_rows": mixed}


def _embed(module, tokens):
    calls = []
    original = module._embed_codes

    def spy(t, p):
        calls.append(t.numel())
        return original(t, p)

    module._embed_codes = spy
    try:
        out = module(tokens)
    finally:
        del module._embed_codes
    assert len(calls) == 1 and calls[0] == torch.unique(tokens * MAX_ACTION_LEN + torch.arange(tokens.size(1))).numel()
    return out


def _all_batches(head_tokens):
    return {**head_tokens, **_constructed_batches(head_tokens)}


def test_forward_bitwise_equal(head_tokens):
    torch.manual_seed(0)
    module = TokenEmbedder()
    for m in module.modules():  # nonzero tables everywhere (default init) so every term counts
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight)
    ref = _reference_of(module)
    for name, tokens in _all_batches(head_tokens).items():
        for train in (False, True):
            module.train(train), ref.train(train)
            out, expected = _embed(module, tokens), ref(tokens)
            assert out.shape == expected.shape == (*tokens.shape, module.type_embedding.embedding_dim), name
            assert out.dtype == expected.dtype and torch.equal(out, expected), f"{name} train={train}"
    real = head_tokens["discard"]
    unique = torch.unique(real * MAX_ACTION_LEN + torch.arange(real.size(1))).numel()
    print(f"discard head: {unique} unique (token, position) pairs of {real.numel()} = {unique / real.numel():.3f}")
    assert unique < real.numel() // 4  # the fast path has rows to save on real data


def test_gradients_equal_in_float64(head_tokens):
    torch.manual_seed(0)
    module = TokenEmbedder().double()
    ref = _reference_of(module)
    g = torch.Generator().manual_seed(1)
    for name, tokens in _all_batches(head_tokens).items():
        upstream = torch.randn(*tokens.shape, module.type_embedding.embedding_dim, generator=g, dtype=torch.float64)
        grads = []
        for m in (module, ref):
            m.zero_grad(set_to_none=True)
            out = _embed(m, tokens) if m is module else m(tokens)
            (out * upstream).sum().backward()
            grads.append({n: p.grad for n, p in m.named_parameters()})
        a, b = grads
        assert a.keys() == b.keys() and len(a) == 5, a.keys()
        for n in a:
            assert a[n] is not None and b[n] is not None, f"{name}: no grad for {n}"
            assert torch.isfinite(a[n]).all(), f"{name}: non-finite grad {n}"
            torch.testing.assert_close(a[n], b[n], atol=ATOL, rtol=RTOL, msg=f"{name}: grad {n}")


def _cast_double(processor):
    prepare = processor.prepare_batches

    def cast(t):
        if isinstance(t, tuple):
            return tuple(cast(x) for x in t)
        return t.double() if t.is_floating_point() else t

    processor.prepare_batches = lambda *a, **k: tuple(cast(h) for h in prepare(*a, **k))


def test_model_logits_bitwise_and_gradients_equal(games):
    """Eval: every head's logits and loss bitwise equal. Train with dropout off, float64: every parameter gradient
    equal."""
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False).eval()
    ref = _reference_of(model)
    with torch.no_grad():
        out, expected = model(*games), ref(*games)
    assert out.keys() == expected.keys() == {"during", "discard", "post"}
    for h in out:
        assert torch.equal(out[h].logits, expected[h].logits) and torch.equal(out[h].loss, expected[h].loss), h

    model = model.double().train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, "attn_dropout_p"):
            m.attn_dropout_p = 0.0
    ref = _reference_of(model)
    grads = []
    for m in (model, ref):
        _cast_double(m.tensor_processor)
        m.zero_grad(set_to_none=True)
        sum(r.loss for r in m(*games).values()).backward()
        grads.append({n: p.grad for n, p in m.named_parameters()})
    a, b = grads
    assert a.keys() == b.keys()
    for n in a:
        assert a[n] is not None and torch.isfinite(a[n]).all(), f"grad {n}"
        torch.testing.assert_close(a[n], b[n], atol=ATOL, rtol=RTOL, msg=f"grad {n}")


def test_codes_decode_to_token_and_position():
    """The (token, position) code is invertible for every 16-bit token at every position."""
    tokens = torch.arange(0, 1 << 16, 7).view(-1, 1).repeat(1, MAX_ACTION_LEN)
    codes = tokens * MAX_ACTION_LEN + torch.arange(MAX_ACTION_LEN)
    back = torch.div(codes, MAX_ACTION_LEN, rounding_mode="floor")
    assert torch.equal(back, tokens) and torch.equal(codes - back * MAX_ACTION_LEN,
                                                     torch.arange(MAX_ACTION_LEN).expand_as(codes))
    assert np.iinfo(np.int64).max // MAX_ACTION_LEN > int(tokens.max())
