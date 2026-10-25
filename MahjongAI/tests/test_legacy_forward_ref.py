"""T-055 [R13]: games mode stays bit-identical. A frozen reference of EncoderConvModel.forward in training mode,
computed from the 9eb24c0 code before T-055 changed the model module: two fixture games, the sample cap triggered on
every head (per-head randperm), dropout on, then backward. The sha256 of the selected labels, the logits, the losses,
every gradient and the final CPU RNG state must equal the frozen values. A change of the per-head RNG order (randperm,
then that head's forward and dropout, in during / discard / post order) or of any float op changes a digest.
The digests are of this machine's float results (CPU, TORCH_THREADS threads) under one torch: one set per torch
major.minor (T-064), each computed from the 9eb24c0 code under that torch, so on every version the test still
says "main equals the 9eb24c0 code, bit for bit". A torch with no set fails, never skips."""
import glob
import hashlib
import os
import re

import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.process_xml import process

pytestmark = pytest.mark.mac_floats  # digests of this Mac's CPU floats: Linux CI deselects them (pytest.ini)

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")
GAMES = ("2021010101gm-00a9-0000-6422c07e.log", "2021010521gm-00a9-0000-e76b4550.log")
SAMPLE_SIZE = 40

# Keyed by torch major.minor (T-064). "2.5": computed on 9eb24c0 (T-010 tip, before T-055) under torch 2.5.1, CPU,
# 4 threads. A new key needs the T-064 D1 procedure first (op trace: the change is platform floats, not ours),
# then the digests of the 9eb24c0 code under that torch, never main's.
FROZEN_BY_TORCH = {"2.5": {
    "heads": ["during", "discard", "post"],
    "during.y": "f182bbfab4f00732",
    "during.logits": "326c33807a8758f1",
    "during.loss": "35dfa89f6a3377f5",
    "discard.y": "3cdeda39624e3d5f",
    "discard.logits": "e0b06b0a449bf59f",
    "discard.loss": "37d89504ab086880",
    "post.y": "1096135d614af7fb",
    "post.logits": "13e38fc469542c00",
    "post.loss": "aba98b27428393ab",
    "grad.token_embedder.type_embedding.weight": "71ae75a4f9ba47a2",
    "grad.token_embedder.tile_embedding.weight": "e93efa96f8a10332",
    "grad.token_embedder.tsumogiri_embedding.weight": "cda0110018a8b95f",
    "grad.token_embedder.during_naki_embedding.weight": "cd67b70cc5a05698",
    "grad.token_embedder.post_naki_embedding.weight": "7526459228285fda",
    "grad.blocks.0.self_attention.key_proj.weight": "9c11feaf2c3903f4",
    "grad.blocks.0.self_attention.key_proj.bias": "04e279841133def4",
    "grad.blocks.0.self_attention.query_proj.weight": "99a174d258101366",
    "grad.blocks.0.self_attention.query_proj.bias": "ffd82d7567f99db1",
    "grad.blocks.0.self_attention.value_proj.weight": "934209fef7f30d6c",
    "grad.blocks.0.self_attention.value_proj.bias": "d081a7c61c9329f5",
    "grad.blocks.0.self_attention.proj.weight": "c58028f1e90a7ec9",
    "grad.blocks.0.self_attention.proj.bias": "ecbcce7fbae87b08",
    "grad.blocks.0.ffwd.net.0.weight": "e9d66abe8e804d43",
    "grad.blocks.0.ffwd.net.0.bias": "46544962093b07e8",
    "grad.blocks.0.ffwd.net.2.weight": "34806445c1ec63b1",
    "grad.blocks.0.ffwd.net.2.bias": "29abd49e9c7055ad",
    "grad.blocks.0.ln1.weight": "76feb203cf4f07eb",
    "grad.blocks.0.ln1.bias": "41d09eda1d67740b",
    "grad.blocks.0.ln2.weight": "d684adf4f61a05e7",
    "grad.blocks.0.ln2.bias": "f34adb648961c193",
    "grad.state_net.suit_conv1d_1.weight": "b3417998be97b183",
    "grad.state_net.suit_conv1d_1.bias": "e939c3d18384cbbc",
    "grad.state_net.suit_conv1d_2.weight": "6249d27822cae1e4",
    "grad.state_net.suit_conv1d_2.bias": "c4ef75dbbaefceb4",
    "grad.state_net.honors_fc1.weight": "8ab6cae7fbf57ded",
    "grad.state_net.honors_fc1.bias": "e27cfa95e04cc9d4",
    "grad.state_net.honors_fc2.weight": "44f741b3ef87c9e2",
    "grad.state_net.honors_fc2.bias": "6891eea8acfc0e64",
    "grad.state_net.reds_fc.weight": "962a01b6b0126ab3",
    "grad.state_net.reds_fc.bias": "3cbfd5d5245deff3",
    "grad.state_net.fc_main_1.weight": "602f7c68725b5a88",
    "grad.state_net.fc_main_1.bias": "cd998118c09b7f45",
    "grad.state_net.fc_main_2.weight": "01b0f26b3ecf1576",
    "grad.state_net.fc_main_2.bias": "7a4a07dac363dc4b",
    "grad.ln_f.weight": "eb1785729a6d5136",
    "grad.ln_f.bias": "955294a81ccfe014",
    "grad.discard_head.weight": "dc9ba037e503b73c",
    "grad.discard_head.bias": "213be4c356e2cc6b",
    "grad.during_head.weight": "550e4d8ed6c8fa55",
    "grad.during_head.bias": "2dff6db77187c4ac",
    "grad.post_head.weight": "96868679218473c7",
    "grad.post_head.bias": "7e7b1384cf415cd9",
    "rng": "92964edc20a11bd4",
}}


def frozen_for(version: str) -> dict:
    """The frozen 9eb24c0 digests of this torch version (its major.minor); an AssertionError when there are none."""
    key = ".".join(version.split("+")[0].split(".")[:2])
    assert key in FROZEN_BY_TORCH, (f"no frozen digests for torch {version}: run the T-064 D1 procedure, then add "
                                    f"the 9eb24c0 digests under this torch as FROZEN_BY_TORCH[{key!r}]")
    return FROZEN_BY_TORCH[key]


def _digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def legacy_digests(sample_size=SAMPLE_SIZE, model_seed=0) -> dict:
    halfturns, tokens = zip(*(process(glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", g))[0]) for g in GAMES))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(SL.TORCH_THREADS)
        torch.manual_seed(model_seed)
        model = EncoderConvModel(verbose=False).train()
        sizes = {h: len(t[3]) for h, t in zip(SL.HEADS, model.tensor_processor.prepare_batches(halfturns, tokens))}
        assert all(n > sample_size for n in sizes.values()), f"the cap must trigger on every head: {sizes}"
        torch.manual_seed(5)
        results = model(list(halfturns), list(tokens), sample=True, sample_size=sample_size)
        sum(r.loss for r in results.values()).backward()
        out = {"heads": list(results)}
        for h, r in results.items():
            assert len(r.y) == sample_size
            out[f"{h}.y"], out[f"{h}.logits"], out[f"{h}.loss"] = _digest(r.y), _digest(r.logits), _digest(r.loss)
        for name, p in model.named_parameters():
            out[f"grad.{name}"] = _digest(p.grad)
        out["rng"] = _digest(torch.get_rng_state())
        return out
    finally:
        torch.set_num_threads(threads)


def test_games_mode_forward_equals_the_frozen_9eb24c0_reference():
    got = legacy_digests()
    frozen = frozen_for(torch.__version__)
    assert frozen, f"no frozen reference: {got}"
    diff = {k: (frozen.get(k), got.get(k)) for k in frozen.keys() | got.keys() if frozen.get(k) != got.get(k)}
    assert not diff, diff


def test_frozen_for_picks_the_set_of_the_torch_major_minor():
    """Negative controls of the lookup: another minor ("2.50" is not "2.5") or an unknown version fails."""
    assert frozen_for("2.5.1") is frozen_for("2.5.0+cpu") is FROZEN_BY_TORCH["2.5"]
    for version in ("2.50.1", "2.99.0", "3.5.1"):
        with pytest.raises(AssertionError, match=rf"no frozen digests for torch {re.escape(version)}: run the T-064"):
            frozen_for(version)


@pytest.mark.parametrize("change", [{"sample_size": SAMPLE_SIZE - 1}, {"model_seed": 1}])
def test_the_frozen_reference_detects_a_change(change):
    """Negative control: another cap or other weights move the digests."""
    frozen = frozen_for(torch.__version__)
    got = legacy_digests(**change)
    assert any(frozen.get(k) != v for k, v in got.items())
