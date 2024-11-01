import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

sys.path.append("..")

from MahjongAI.utils.constants import EventStateTypes

# Hyperparameters
ENCODER_EMBD_DIM = 315
DECODER_STATE_OBJ_DIM = [(37, 3), (4, 7), (1, 4)]
DECODER_EMBD_DIM = 1024
DISCARD_ACTION_DIM = 37
DURING_TURN_ACTION_DIM = 71  # pass, agari (tsumo), riichi, ankan, kakan
POST_TURN_ACTION_DIM = 154  # pass, agari (ron), naki
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
        self,
        encoding_tokens_batch: torch.tensor,
        state_obj_tensor_batch: Tuple[torch.tensor, torch.tensor, torch.tensor],
        action_mask_batch: torch.tensor,
        y_tensor: torch.tensor,
        head: str,
    ):
        enc_out = self.encoder(encoding_tokens_batch)  # Shape: (B, 150, EMBD_SIZE)
        logits, loss = self.decoder(
            enc_out, state_obj_tensor_batch, action_mask_batch, y_tensor, head
        )
        return logits, loss


class Encoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()

        self.empty_embedding = torch.zeros(
            (1, 1, EMBD_SIZE), device=device, requires_grad=False
        )
        self.type_embedding = nn.Embedding(4, EMBD_SIZE)  # embedding_type: 1-4
        self.tile_embedding = nn.Embedding(37, EMBD_SIZE)  # tile_encoding: 0-36
        self.tsumogiri_embedding = nn.Embedding(1, EMBD_SIZE)
        self.during_naki_embedding = nn.Embedding(DURING_TURN_ACTION_DIM, EMBD_SIZE)
        self.post_naki_embedding = nn.Embedding(POST_TURN_ACTION_DIM, EMBD_SIZE)

        self.position_coding_table = (
            self._get_position_encoding().to(device).detach()
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

    def forward(self, encoding_tokens_batch: torch.tensor):
        """
        Args:
            encoding_tokens_batch: (B, 150) tensor of encoding tokens

        Returns:
            x: (B, 150, 128) tensor of embeddings
        """
        # TODO: instead of doing this makeshift adjustment, find the root cause
        encoding_tokens_batch = encoding_tokens_batch.to(torch.int64)

        event_type = (encoding_tokens_batch >> 13) & 7
        embedding_type = (encoding_tokens_batch >> 10) & 7
        player_idx = (encoding_tokens_batch >> 8) & 3
        tile_encoding = encoding_tokens_batch & 0x7F
        tsumogiri = (encoding_tokens_batch >> 7) & 1
        naki_idx = encoding_tokens_batch & 0xFF

        # retrieve type embeddings if event_type is not EMPTY
        adjusted_embedding_type = torch.clamp(embedding_type - 1, min=0)
        type_emb = self.type_embedding(
            adjusted_embedding_type
        )  # Shape: (B, T, EMBD_SIZE)
        mask = event_type == EventStateTypes.EMPTY
        type_emb = type_emb * mask.unsqueeze(-1).float()

        # retrieve tile embeddings if event_type is DISCARD or NEW_DORA
        # clamp tile_encoding to max 36
        adjusted_tile_encoding = torch.clamp(tile_encoding, max=36)
        tile_emb = self.tile_embedding(
            adjusted_tile_encoding
        )  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DISCARD) & (
            event_type != EventStateTypes.NEW_DORA
        )
        tile_emb = tile_emb * mask.unsqueeze(-1).float()

        # Retrieve player encodings if event_type is DISCARD, DURING, or POST
        player_idx = player_idx.to(self.player_coding_table.device)
        player_emb = self.player_coding_table[player_idx]  # Shape: (B, T, EMBD_SIZE)
        mask = (
            (event_type != EventStateTypes.DISCARD)
            & (event_type != EventStateTypes.DURING)
            & (event_type != EventStateTypes.POST)
        ).to(player_emb.device)
        player_emb = player_emb * mask.unsqueeze(-1).float()
        player_emb = player_emb.to(device)

        # tsumogiri embedding only if event_type is DISCARD and tsumogiri is 1
        tsumogiri_emb = self.tsumogiri_embedding(torch.tensor([0], device=device))  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DISCARD) | (tsumogiri == 0)
        tsumogiri_emb = tsumogiri_emb * mask.unsqueeze(-1).float()

        # Retrieve position encodings (shape: (150, EMBD_SIZE)) if embedding type is not EMPTY
        pos_emb = self.position_coding_table[
            : encoding_tokens_batch.size(1)
        ]  # Shape: (T, EMBD_SIZE)
        pos_emb = pos_emb.unsqueeze(0)  # Shape: (1, T, EMBD_SIZE)
        mask = event_type != EventStateTypes.EMPTY
        pos_emb = pos_emb * mask.unsqueeze(-1).float()

        # Retrieve during naki encodings if event_type is DURING and naki_idx is >= 3
        adjusted_during_naki_idx = torch.clamp(naki_idx, max=DURING_TURN_ACTION_DIM - 1)
        during_naki_emb = self.during_naki_embedding(
            adjusted_during_naki_idx
        )  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.DURING) | (naki_idx < 3)
        during_naki_emb = during_naki_emb * mask.unsqueeze(-1).float()

        # retrieve post naki encodings if event_type is POST and naki_idx is >= 2
        adjusted_post_naki_idx = torch.clamp(naki_idx, max=POST_TURN_ACTION_DIM - 1)
        post_naki_emb = self.post_naki_embedding(
            adjusted_post_naki_idx
        )  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type != EventStateTypes.POST) | (naki_idx < 2)
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
        )  # Shape: (B, T, EMBD_SIZE)

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
        state_obj_tensor_batch: Tuple[torch.tensor, torch.tensor, torch.tensor],
        action_mask_batch: torch.tensor,
        y_tensor: torch.tensor,
        head: str,
    ):
        """
        Args:
            enc_out: (B, 150, EMBD_SIZE) tensor
            state_obj_tensor_batch: Tuple of (B, 37, 3), (B, 4, 7), (B, 1, 4) tensors
            action_mask_batch: (B, action_dim) tensor
            y_tensor: (B,) tensor
            head: str indicating the head to use
        """
        assert torch.all(
            (action_mask_batch == 0) | (action_mask_batch == 1)
        ), "action_mask_batch must be binary"
        assert torch.any(
            action_mask_batch, dim=1
        ).all(), "Each sample must have at least one allowed action"

        q = self.state_net(state_obj_tensor_batch)
        x = self.blocks(enc_out, enc_out, q)
        x = self.ln_f(x)
        logits = self.heads[head](x)

        assert (
            logits.shape == action_mask_batch.shape
        ), f"Logits shape {logits.shape} and mask shape {action_mask_batch.shape} do not match."

        # mask out invalid actions
        logits = logits.masked_fill(action_mask_batch == 0, -1e9)

        loss = F.cross_entropy(logits, y_tensor, reduction="mean")
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
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M params")

    # # Create dummy data
    # B = 2  # Batch size
    # encoding_tokens_batch = torch.ones((B, 150), dtype=torch.long).to(device)
    # state_obj_tensor_batch = (
    #     torch.randn(B, 3, 37).to(device),
    #     torch.randn(B, 4, 7).to(device),
    #     torch.randn(B, 1, 4).to(device),
    # )
    # action_mask_batch = torch.ones((B, DURING_TURN_ACTION_DIM), dtype=torch.float32).to(
    #     device
    # )
    # y_tensor = torch.randint(0, DURING_TURN_ACTION_DIM, (B,), dtype=torch.long).to(
    #     device
    # )
    # head = "during"

    # logits, loss = model(
    #     encoding_tokens_batch, state_obj_tensor_batch, action_mask_batch, y_tensor, head
    # )
    # print("Logits shape:", logits.shape)  # Expected: (B, DURING_TURN_ACTION_DIM)
    # print("Loss:", loss.item())
