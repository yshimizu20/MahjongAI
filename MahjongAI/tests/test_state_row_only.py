"""T-011 speed levers: the fast paths compute exactly what the full paths compute (same weights, same inputs)."""
import pytest
import torch

from MahjongAI.agents.encoder_conv import EMBD_SIZE, N_HEADS, EncoderBlock


def _padded_batch(B=6, T=151, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, EMBD_SIZE, generator=g)
    valid = torch.randint(1, T, (B,), generator=g)
    mask = torch.arange(T).unsqueeze(0) < valid.unsqueeze(1)
    mask[:, -1] = True  # the state token is always valid
    return x, mask.unsqueeze(1)


def _no_dropout(block):
    for m in block.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    block.self_attention.attn_dropout_p = 0.0


@pytest.mark.parametrize("mode", ["eval", "train_p0"])
def test_state_row_only_equals_full_last_row(mode):
    torch.manual_seed(0)
    block = EncoderBlock(EMBD_SIZE, N_HEADS)
    if mode == "eval":
        block.eval()
    else:  # train mode takes the math attention kernel with the mask; dropout off so both paths are deterministic
        block.train()
        _no_dropout(block)
    x, mask = _padded_batch()

    full = block(x, mask)[:, -1:]
    full.sum().backward()
    grads_full = {n: p.grad.clone() for n, p in block.named_parameters()}
    block.zero_grad()

    fast = block(x, mask, state_row_only=True)
    fast.sum().backward()

    assert fast.shape == (x.shape[0], 1, EMBD_SIZE)
    torch.testing.assert_close(fast, full, atol=1e-6, rtol=1e-5)
    for n, p in block.named_parameters():
        torch.testing.assert_close(p.grad, grads_full[n], atol=1e-6, rtol=1e-5, msg=f"grad {n}")
