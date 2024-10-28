import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MahjongAI.utils.constants import EventTypes, EventStateTypes

# Hyperparameters
ENCODER_EMBD_DIM = 315
DECODER_STATE_OBJ_DIM = [(37, 3), (4, 7), (1, 4)]
DECODER_EMBD_DIM = 1024
DISCARD_ACTION_DIM = 37
DURING_TURN_ACTION_DIM = 34 * 2 + 1 + 1 + 1 # 71; ankan, kakan, riichi, tsumo, pass
POST_TURN_ACTION_DIM = 154 # pass, agari (ron), naki
MAX_ACTION_LEN = 150
EMBD_SIZE = 128
N_HEADS = 8
N_LAYERS = 3
DROPOUT_RATIO = 0.2

max_iters = 10_000
eval_interval = 100
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(
        self, enc_indices: np.array, state_obj: torch.tensor, head: str, target=None
    ):
        enc_out = self.encoder(enc_indices)
        logits, loss = self.decoder(enc_out, state_obj, target, head)
        return logits, loss


class Encoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        self.type_embedding = nn.Embedding(5, EMBD_SIZE)  # embedding_type: 0-4
        self.tile_embedding = nn.Embedding(37, EMBD_SIZE)  # tile_encoding: 0-36
        self.tsumogiri_embedding = nn.Embedding(1, EMBD_SIZE)

        self.position_coding_table = (
            self._get_position_encoding().detach()
        )  # Non-trainable
        self.player_coding_table = self._get_player_encoding().detach()  # Non-trainable

        self._init_embeddings()

        self.blocks = nn.Sequential(
            *[EncoderBlock(EMBD_SIZE, N_HEADS) for _ in range(n_layers)]
        )

        self.apply(self._init_weights)

    def _init_embeddings(self):
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.normal_(self.tile_embedding.weight, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def _get_position_encoding():
        position = torch.arange(MAX_ACTION_LEN).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, EMBD_SIZE, 2).float() * -(np.log(10000.0) / EMBD_SIZE)
        )
        pos_encoding = torch.zeros(MAX_ACTION_LEN, EMBD_SIZE)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.requires_grad_(False)  # Ensure non-trainable

    @staticmethod
    def _get_player_encoding():
        encoding = torch.empty((4, EMBD_SIZE), dtype=torch.float32)

        base = torch.pow(
            10000, torch.arange(0, EMBD_SIZE, 2, dtype=torch.float32) / EMBD_SIZE
        )

        player_numbers = torch.arange(4).float()
        angle_rads = player_numbers * (2 * np.pi) / 4

        angle_rads_matrix = angle_rads[:, None] / base[None, :]

        sin = torch.sin(angle_rads_matrix)
        cos = torch.cos(angle_rads_matrix)

        encoding[:, 0::2] = sin
        encoding[:, 1::2] = cos

        return encoding.requires_grad_(False)  # Ensure non-trainable

    def forward(self, tokens: torch.tensor):
        """
        Args:
            tokens: (B, 150) tensor of token indices

        Returns:
            x: (B, 150, 128) tensor of embeddings
        """
        event_type = (tokens >> 13) & 7
        embedding_type = (tokens >> 10) & 7
        player_idx = (tokens >> 8) & 3
        tile_encoding = tokens & 0x7F
        tsumogiri = (tokens >> 7) & 1
        naki_idx = tokens & 0xFF

        type_emb = self.type_embedding(embedding_type)  # Shape: (B, T, EMBD_SIZE)

        # retrieve tile embeddings if event_type is DISCARD or NEW_DORA
        tile_emb = self.tile_embedding(tile_encoding)  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DISCARD) & (
            event_type != EventStateTypes.NEW_DORA
        )
        tile_emb = tile_emb * mask.unsqueeze(-1).float()

        # Retrieve player encodings if event_type is DISCARD, DURING, or POST
        player_emb = self.player_coding_table[player_idx]  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DISCARD) & (
            event_type != EventStateTypes.DURING
        ) & (event_type != EventStateTypes.POST)
        player_emb = player_emb * mask.unsqueeze(-1).float()

        # tsumogiri embedding only if event_type is DISCARD and tsumogiri is 1
        tsumogiri_emb = self.tsumogiri_embedding(tsumogiri)  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DISCARD) | (tsumogiri == 0)
        tsumogiri_emb = tsumogiri_emb * mask.unsqueeze(-1).float()

        # Retrieve position encodings (shape: (150, EMBD_SIZE)) if embedding type is not EMPTY
        pos_emb = self.position_coding_table[: tokens.size(1)]  # Shape: (T, EMBD_SIZE)
        pos_emb = pos_emb.unsqueeze(0)  # Shape: (1, T, EMBD_SIZE)
        mask = event_type != EventStateTypes.EMPTY
        pos_emb = pos_emb * mask.unsqueeze(-1).float()

        # Retrieve during naki encodings if event_type is DURING and naki_idx is >= 3
        during_naki_emb = self.tile_embedding(naki_idx)  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DURING) | (naki_idx < 3)
        during_naki_emb = during_naki_emb * mask.unsqueeze(-1).float()

        # retrieve post naki encodings if event_type is POST and naki_idx is >= 2
        post_naki_emb = self.tile_embedding(naki_idx)  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.POST) | (naki_idx < 2)
        post_naki_emb = post_naki_emb * mask.unsqueeze(-1).float()

        x = type_emb + tile_emb + player_emb + tsumogiri_emb + pos_emb + during_naki_emb + post_naki_emb # Shape: (B, T, EMBD_SIZE)

        x = self.blocks(x)
        return x  # Shape: (B, T, EMBD_SIZE)


class Decoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        self.state_net = StateNet()
        self.blocks = nn.Sequential(
            *[DecoderBlock(EMBD_SIZE, N_HEADS) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(EMBD_SIZE)
        self.discard_head = nn.Linear(EMBD_SIZE, DISCARD_ACTION_DIM)
        self.during_head = nn.Linear(EMBD_SIZE, DURING_TURN_ACTION_DIM)
        self.post_head = nn.Linear(EMBD_SIZE, POST_TURN_ACTION_DIM)

        self.heads = {
            "discard": self.discard_head,
            "during": self.during_head,
            "post": self.post_head,
        }

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        enc_out: torch.tensor,
        state_obj: torch.tensor,
        head: str,
        target: int = None,
    ):
        q = self.state_net(state_obj)
        x = self.blocks(enc_out, enc_out, q)
        x = self.ln_f(x)
        logits = self.heads[head](x)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, target)

        return logits, loss


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


class StateNet(nn.Module):
    def __init__(self):
        super().__init__()

        # subnetwork 1
        self.conv1_1 = nn.Conv2d(DECODER_STATE_OBJ_DIM[0][1], 8, 3, padding=1)
        self.norm1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = nn.Conv2d(8, 8, 3, padding=1)
        self.norm1_2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(DECODER_STATE_OBJ_DIM[0][0] * 8, 64)

        # subnetwork 2
        self.conv2_1 = nn.Conv2d(DECODER_STATE_OBJ_DIM[1][1], 8, 3, padding=1)
        self.norm2_1 = nn.BatchNorm2d(8)
        self.fc2 = nn.Linear(DECODER_STATE_OBJ_DIM[1][0] * 8, 16)

        # main network
        self.fc = nn.Linear(
            64 + 16 + DECODER_STATE_OBJ_DIM[2][0] * DECODER_STATE_OBJ_DIM[2][1],
            EMBD_SIZE,
        )

    def forward(self, state_obj):
        x1, x2, x3 = state_obj

        x1 = F.relu(self.norm1_1(self.conv1_1(x1)))
        x1 = F.relu(self.norm1_2(self.conv1_2(x1)))
        x1 = x1.view(x1.shape[0], -1)
        x1 = F.relu(self.fc1(x1))

        x2 = F.relu(self.norm2_1(self.conv2_1(x2)))
        x2 = x2.view(x2.shape[0], -1)
        x2 = F.relu(self.fc2(x2))

        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.relu(self.fc(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBD_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATIO)

    def forward(self, k, v, q):
        out = torch.cat([h(k, v, q) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


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

        v = self.value(v)
        out = wei @ v
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


if __name__ == "__main__":
    model = TransformerModel().to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
