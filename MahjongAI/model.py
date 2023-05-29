import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ENCODER_EMBD_DIM = 1024
DECODER_STATE_OBJ_DIM = (37, 28)
DECODER_EMBD_DIM = 1024
DISCARD_ACTION_DIM = 37
REACH_ACTION_DIM = 2
AGARI_ACTION_DIM = 2
MELD_ACTION_DIM = 1024
MAX_ACTION_LEN = 128
EMBD_SIZE = 128
N_HEADS = 8
N_LAYERS = 3
DROPOUT_RATIO = 0.2

max_iters = 10_000
eval_interval = 100
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(idx):
    raise NotImplementedError


def decode(idx):
    raise NotImplementedError


def get_batch():
    raise NotImplementedError


@torch.no_grad()
def estimate_loss():
    raise NotImplementedError


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBD_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBD_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBD_SIZE, head_size, bias=False)

        self.dropout = nn.Dropout(DROPOUT_RATIO)

    def forward(self, k, v, q):
        k = self.key(k)
        q = self.query(q)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(v)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBD_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATIO)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, EMBD_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBD_SIZE, 4 * EMBD_SIZE),
            nn.GELU(),
            nn.Linear(4 * EMBD_SIZE, EMBD_SIZE),
            nn.Dropout(DROPOUT_RATIO),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, EMBD_SIZE, num_heads):
        super().__init__()
        head_size = EMBD_SIZE // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(EMBD_SIZE)
        self.ln1 = nn.LayerNorm(EMBD_SIZE)
        self.ln2 = nn.LayerNorm(EMBD_SIZE)
        self.ln3 = nn.LayerNorm(EMBD_SIZE)
        self.ln4 = nn.LayerNorm(EMBD_SIZE)

    def forward(self, k, v, q):
        k = self.ln1(k)
        v = self.ln2(v)
        q = self.ln3(q)

        q = q + self.attention(k, v, q)
        q = q + self.ffwd(self.ln4(q))
        return q


class EncoderBlock(nn.Module):
    def __init__(self, EMBD_SIZE, num_heads):
        super().__init__()
        head_size = EMBD_SIZE // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(EMBD_SIZE)
        self.ln1 = nn.LayerNorm(EMBD_SIZE)
        self.ln2 = nn.LayerNorm(EMBD_SIZE)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attention(x, x, x)
        x = x + self.ffwd(self.ln2(x))
        return x


class StateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(DECODER_STATE_OBJ_DIM[1], 16, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(16)
        self.lm = nn.Linear(DECODER_STATE_OBJ_DIM[0] * 16, EMBD_SIZE)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.lm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        self.state_net = StateNet()
        self.blocks = nn.Sequential(
            *[DecoderBlock(EMBD_SIZE, N_HEADS) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(EMBD_SIZE)
        self.discard_head = nn.Linear(EMBD_SIZE, DISCARD_ACTION_DIM)
        self.reach_head = nn.Linear(EMBD_SIZE, REACH_ACTION_DIM)
        self.agari_head = nn.Linear(EMBD_SIZE, AGARI_ACTION_DIM)
        self.meld_head = nn.Linear(EMBD_SIZE, MELD_ACTION_DIM)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, enc_out: torch.tensor, state_obj: torch.tensor, target: int = None
    ):
        q = self.state_net(state_obj)
        x = self.blocks(enc_out, enc_out, q)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, target)

        return logits, loss


class Encoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        self.token_embedding_table = nn.Embedding(ENCODER_EMBD_DIM, EMBD_SIZE)
        self.position_embedding_table = nn.Embedding(MAX_ACTION_LEN, EMBD_SIZE)
        self.blocks = nn.Sequential(
            *[EncoderBlock(EMBD_SIZE, N_HEADS) for _ in range(n_layers)]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, enc_indices: torch.Tensor):
        B, T = enc_indices.shape
        tok = self.token_embedding_table(enc_indices)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, enc_indices: np.array, state_obj: torch.tensor, targets=None):
        enc_out = self.encoder(enc_indices)
        logits, loss = self.decoder(enc_out, state_obj, targets)
        return logits, loss


if __name__ == "__main__":
    model = TransformerModel().to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    raise NotImplementedError

    for iter in range(max_iters):
        if iter % eval_interval == eval_interval - 1:
            losses = estimate_loss()
            print(
                f"iter {iter} | train loss {losses['train']} | val loss {losses['val']}"
            )

        xb, yb = get_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
