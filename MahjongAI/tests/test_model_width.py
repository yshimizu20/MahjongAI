"""T-010: the model width and depth are constructor arguments. Every width site takes the model's own width (real
dims and parameter counts at 64 x 1 / 128 x 1 / 128 x 2); a 64-wide and a 128-wide model live in one process; the
sinusoidal tables use the model's width in their frequencies; from_checkpoint and both inference loaders build the
checkpoint's architecture (64 x 1 when its config has no architecture keys) and strip a compiled model's prefix."""
import glob
import hashlib
import math
import os
import sys

import pytest
import torch

from MahjongAI.agents.encoder_conv import (DISCARD_ACTION_DIM, DURING_TURN_ACTION_DIM, MAX_ACTION_LEN, N_HEADS,
                                           POST_TURN_ACTION_DIM, EncoderConvModel, TokenEmbedder, arch_of,
                                           count_parameters)
from MahjongAI.engine import players
from MahjongAI.process_xml import process
from MahjongAI.viewer import infer, viewer_lib

ARCHS = [(64, 1, 201_311), (128, 1, 495_647), (128, 2, 693_919)]  # (embd_size, n_layers, parameters), plan T-010
GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
GAME = "2021010521gm-00a9-0000-e76b4550.log"  # triple ron, riichi discards: every head has rows


@pytest.fixture(scope="module")
def game():
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", GAME))
    assert len(found) == 1, found
    return process(found[0])


def _assert_dims(model, E, L):
    """Every width site of the model has width E, and it has L blocks."""
    te, sn = model.token_embedder, model.state_net
    assert model.embd_size == E and len(model.blocks) == L
    for table in (te.type_embedding, te.tile_embedding, te.tsumogiri_embedding, te.during_naki_embedding,
                  te.post_naki_embedding):
        assert table.embedding_dim == E
    assert te.empty_embedding.shape == (1, 1, E)
    assert te.position_coding_table.shape == (MAX_ACTION_LEN, E) and te.player_coding_table.shape == (4, E)
    for block in model.blocks:
        att = block.self_attention
        assert (att.num_heads, att.head_size) == (N_HEADS, E // N_HEADS)
        for proj in (att.key_proj, att.query_proj, att.value_proj, att.proj):
            assert (proj.in_features, proj.out_features) == (E, E)
        assert (block.ffwd.net[0].in_features, block.ffwd.net[0].out_features) == (E, 4 * E)
        assert (block.ffwd.net[2].in_features, block.ffwd.net[2].out_features) == (4 * E, E)
        assert block.ln1.normalized_shape == block.ln2.normalized_shape == (E,)
    assert sn.fc_main_1.out_features == E and (sn.fc_main_2.in_features, sn.fc_main_2.out_features) == (E, E)
    assert model.ln_f.normalized_shape == (E,)
    for head, out in ((model.discard_head, DISCARD_ACTION_DIM), (model.during_head, DURING_TURN_ACTION_DIM),
                      (model.post_head, POST_TURN_ACTION_DIM)):
        assert (head.in_features, head.out_features) == (E, out)


@pytest.mark.parametrize("E,L,n_params", ARCHS)
def test_dims_and_parameter_count(E, L, n_params):
    model = EncoderConvModel(n_layers=L, embd_size=E, verbose=False)
    _assert_dims(model, E, L)
    assert sum(p.numel() for p in model.parameters()) == count_parameters(E, L) == n_params


def test_defaults_are_64_by_1():
    _assert_dims(EncoderConvModel(verbose=False), 64, 1)
    assert count_parameters() == 201_311


def test_count_parameters_draws_no_random_number():
    """The trainer calls it after torch.manual_seed: a draw would move the model's initialisation."""
    torch.manual_seed(0)
    state = torch.get_rng_state()
    count_parameters(128, 2)
    assert torch.equal(torch.get_rng_state(), state)


# Frozen from the pre-T-010 code (f23862a, fixed width 64): sha256 over the seed-0 state_dict (name, shape, bytes in
# state_dict order) and the next three torch.rand draws after the build (the RNG state the build leaves behind). The
# digest holds on macOS only (torch 2.5.1 and 2.13 agree there); Linux CPU init writes other float bits from the same
# RNG stream, so the RNG-state check is the one that runs everywhere
PRE_T010_INIT_SHA256 = "464d068574eb18ed46252e960d41c231cc68a562b550ba8c439f05deaddcc44e"
PRE_T010_NEXT_RAND = [0.935202419757843, 0.6767680644989014, 0.6052327156066895]
OFF_DIGEST_PLATFORM = sys.platform != "darwin"


def _init_fingerprint(**kwargs):
    torch.manual_seed(0)
    sd = EncoderConvModel(verbose=False, **kwargs).state_dict()
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest(), torch.rand(3).tolist()


def test_default_model_leaves_the_rng_state_as_before():
    """The default and an explicit 64 x 1 build draw as many random numbers as the pre-T-010 build: same RNG after."""
    assert _init_fingerprint()[1] == PRE_T010_NEXT_RAND
    assert _init_fingerprint(n_layers=1, embd_size=64)[1] == PRE_T010_NEXT_RAND


@pytest.mark.skipif(OFF_DIGEST_PLATFORM, reason="frozen digest of macOS CPU init floats (torch 2.5.1, 2.13)")
def test_default_model_is_initialised_as_before():
    """The default and an explicit 64 x 1 model draw the pre-T-010 initial weights, byte for byte."""
    assert _init_fingerprint()[0] == PRE_T010_INIT_SHA256
    assert _init_fingerprint(n_layers=1, embd_size=64)[0] == PRE_T010_INIT_SHA256


def test_two_widths_coexist_in_one_process(game):
    """Built in either order, each model keeps its own dims, and both run a forward on a real game in turns: the
    history embedding and the state row reach the heads at the model's own width."""
    narrow, wide = EncoderConvModel(verbose=False), EncoderConvModel(n_layers=2, embd_size=128, verbose=False)
    wide_first = EncoderConvModel(n_layers=2, embd_size=128, verbose=False)
    narrow_after = EncoderConvModel(verbose=False)
    for model, E, L in ((narrow, 64, 1), (wide, 128, 2), (wide_first, 128, 2), (narrow_after, 64, 1)):
        _assert_dims(model, E, L)
    halfturns, tokens = game
    for model, E in ((narrow, 64), (wide, 128), (narrow, 64)):
        widths = []
        hooks = [model.token_embedder.register_forward_hook(lambda m, i, o: widths.append(o.shape[-1])),
                 model.ln_f.register_forward_hook(lambda m, i, o: widths.append(o.shape[-1]))]
        with torch.no_grad():
            out = model.eval()([halfturns], [tokens])
        for h in hooks:
            h.remove()
        assert out.keys() == {"during", "discard", "post"} and widths == [E] * 6, widths
        assert all(torch.isfinite(r.loss) for r in out.values())


@pytest.mark.parametrize("E", [64, 128])
def test_sinusoidal_tables_use_the_model_width(E):
    """position[p, 2i] = sin(p / 10000^(2i/E)), position[p, 2i+1] = cos(...); player[s, 2i] = sin(s pi/2 /
    10000^(2i/E)), player[s, 2i+1] = cos(...): computed here in float64 from the formula, not by the module code."""
    te = TokenEmbedder(E)
    pos = torch.tensor([[(math.sin if c % 2 == 0 else math.cos)(p / 10000 ** ((c - c % 2) / E)) for c in range(E)]
                        for p in range(MAX_ACTION_LEN)], dtype=torch.float64)
    ply = torch.tensor([[(math.sin if c % 2 == 0 else math.cos)(s * math.pi / 2 / 10000 ** ((c - c % 2) / E))
                         for c in range(E)] for s in range(4)], dtype=torch.float64)
    torch.testing.assert_close(te.position_coding_table.double(), pos, atol=1e-4, rtol=0)
    torch.testing.assert_close(te.player_coding_table.double(), ply, atol=1e-5, rtol=0)


# ---------------------------------------------------------------- loading a checkpoint
def _checkpoint(E, L, seed, config="arch", prefix=""):
    """A checkpoint dict of a seeded E x L model. config: "arch" (T-010 keys), "legacy" (a T-051 config: no
    architecture keys), None (no config at all, as the engine tests save)."""
    torch.manual_seed(seed)
    model = EncoderConvModel(n_layers=L, embd_size=E, verbose=False)
    ckpt = {"model_state_dict": {prefix + k: v for k, v in model.state_dict().items()}}
    if config == "arch":
        ckpt["config"] = {"model": "encoder_conv", "total_steps": 10, "embd_size": E, "n_layers": L}
    elif config == "legacy":
        ckpt["config"] = {"model": "encoder_conv", "total_steps": 10}
    return ckpt, model


def _assert_same_weights(model, reference):
    a, b = model.state_dict(), reference.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("E,L,config,prefix", [(64, 1, None, ""), (64, 1, "legacy", ""), (64, 1, "arch", ""),
                                               (128, 2, "arch", ""), (64, 1, "legacy", "_orig_mod."),
                                               (128, 2, "arch", "_orig_mod.")])
def test_from_checkpoint_builds_the_saved_architecture(E, L, config, prefix):
    ckpt, saved = _checkpoint(E, L, seed=3, config=config, prefix=prefix)
    model = EncoderConvModel.from_checkpoint(ckpt)
    _assert_dims(model, E, L)
    _assert_same_weights(model, saved)


def test_from_checkpoint_refuses_a_wrong_or_partial_architecture():
    ckpt, _ = _checkpoint(128, 2, seed=3, config="legacy")  # 128 x 2 weights under a config read as 64 x 1
    with pytest.raises(RuntimeError, match="size mismatch"):
        EncoderConvModel.from_checkpoint(ckpt)
    ckpt["config"]["embd_size"] = 128  # n_layers absent: neither legacy nor complete
    with pytest.raises(AssertionError, match="architecture keys"):
        EncoderConvModel.from_checkpoint(ckpt)
    assert arch_of({}) == {"embd_size": 64, "n_layers": 1}


@pytest.mark.parametrize("E,L,config", [(64, 1, "legacy"), (64, 1, None), (128, 2, "arch")])
def test_inference_loaders_load_the_checkpoint_architecture(tmp_path, game, E, L, config):
    """players.load_model and infer.load_model on a saved file: the saved dims and weights; the viewer's per-sample
    path equals the training forward on a real game (float64), and the engine's predict equals the viewer's."""
    ckpt, saved = _checkpoint(E, L, seed=5, config=config)
    path = str(tmp_path / f"step_{E}x{L}_{config}.pt")  # a distinct path: players.load_model is memoised per path
    torch.save(ckpt, path)
    engine_model, viewer_model = players.load_model(path), infer.load_model(path)
    for model in (engine_model, viewer_model):
        _assert_dims(model, E, L)
        _assert_same_weights(model, saved)
        assert not model.training
    halfturns, tokens = game
    report = viewer_lib.check_parity(viewer_lib.double_model(viewer_model.cpu()), halfturns, tokens)
    assert all(r["n"] > 0 and r["max_abs_delta"] < viewer_lib.PARITY_TOL for r in report.values()), report
    compared = 0
    for ri, round_turns in enumerate(halfturns):
        for turn in round_turns[:40]:
            for seat in infer.deciding_seats(turn):
                sample = infer.build_sample(turn, tokens[ri], seat)
                if sample is None:
                    continue
                _, viewer_logits, _ = infer.predict(viewer_model, sample)
                engine_logits = players.predict(engine_model, players.build_sample(turn, tokens[ri], seat))
                torch.testing.assert_close(engine_logits, viewer_logits, atol=1e-5, rtol=1e-5)
                compared += 1
    assert compared > 20, compared
