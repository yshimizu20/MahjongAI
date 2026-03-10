import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

sys.path.append("../..")

from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.decision import Decision, NakiDecision
from MahjongAI.utils.constants import (
    DECISION_AGARI_IDX,
    DECISION_REACH_IDX,
    DECISION_NAKI_IDX,
    TILE2IDX,
    MAX_SEQUENCE_LENGTH,
    EventStateTypes,
)

# Hyperparameters
DISCARD_ACTION_DIM = 37
DURING_TURN_ACTION_DIM = 71  # pass, agari (tsumo), riichi, ankan, kakan
POST_TURN_ACTION_DIM = 154  # pass, agari (ron), naki
MAX_ACTION_LEN = 150
EMBD_SIZE = 64
N_HEADS = 8
N_LAYERS = 1
DROPOUT_RATIO = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.tensor_processor = TransformerTensorProcessor()

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
        all_halfturns_list: List[list[HalfTurn]],
        all_encoding_tokens_list: List[list[int]],
        sample=False,
        sample_size=2250,
    ):
        (
            tensors_during,
            tensors_discard,
            tensors_post,
        ) = self.tensor_processor.prepare_batches(
            all_halfturns_list, all_encoding_tokens_list
        )

        for tensors, head in zip(
            [tensors_during, tensors_discard, tensors_post],
            ["during", "discard", "post"],
        ):
            if len(tensors[0]) == 0:
                continue

            if sample and len(tensors[0]) > sample_size:
                idx = torch.randperm(len(tensors[0]))[:sample_size]

                tensors = (
                    tensors[0][idx],
                    (
                        tensors[1][0][idx],
                        tensors[1][1][idx],
                        tensors[1][2][idx],
                    ),
                    tensors[2][idx],
                    tensors[3][idx],
                    tensors[4][idx],  # padding_mask
                )

            (
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
                padding_mask_batch,
            ) = tensors
            encoding_tokens_batch = encoding_tokens_batch.to(device)
            state_obj_tensor_batch = tuple(
                tensor.to(device) for tensor in state_obj_tensor_batch
            )
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)
            padding_mask_batch = padding_mask_batch.to(device)

            enc_out = self.encoder(encoding_tokens_batch, padding_mask_batch)  # Shape: (B, 150, EMBD_SIZE)
            logits, loss = self.decoder(
                enc_out, state_obj_tensor_batch, action_mask_batch, y_tensor, head, padding_mask_batch
            )
            yield logits, loss, head


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

    def forward(self, encoding_tokens_batch: torch.tensor, padding_mask: torch.tensor):
        """
        Args:
            encoding_tokens_batch: (B, 150) tensor of encoding tokens
            padding_mask: (B, 150) boolean tensor where True = valid position, False = padded (REQUIRED)

        Returns:
            x: (B, 150, EMBD_SIZE) tensor of embeddings
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
        mask = event_type != EventStateTypes.EMPTY
        type_emb = type_emb * mask.unsqueeze(-1).float()

        # retrieve tile embeddings if event_type is DISCARD or NEW_DORA
        # clamp tile_encoding to max 36
        adjusted_tile_encoding = torch.clamp(tile_encoding, max=36)
        tile_emb = self.tile_embedding(
            adjusted_tile_encoding
        )  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type == EventStateTypes.DISCARD) | (
            event_type == EventStateTypes.NEW_DORA
        )
        tile_emb = tile_emb * mask.unsqueeze(-1).float()

        # Retrieve player encodings if event_type is DISCARD, DURING, or POST
        player_idx = player_idx.to(self.player_coding_table.device)
        player_emb = self.player_coding_table[player_idx]  # Shape: (B, T, EMBD_SIZE)
        mask = (
            (event_type == EventStateTypes.DISCARD)
            | (event_type == EventStateTypes.DURING)
            | (event_type == EventStateTypes.POST)
        ).to(player_emb.device)
        player_emb = player_emb * mask.unsqueeze(-1).float()
        player_emb = player_emb.to(device)

        # tsumogiri embedding only if event_type is DISCARD and tsumogiri is 1
        tsumogiri_emb = self.tsumogiri_embedding(
            torch.tensor([0], device=device)
        )  # Shape: (B, T, EMBD_SIZE)
        mask = (event_type == EventStateTypes.DISCARD) & (tsumogiri == 1)
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
        mask = (event_type == EventStateTypes.DURING) & (naki_idx >= 3)
        during_naki_emb = during_naki_emb * mask.unsqueeze(-1).float()

        # retrieve post naki encodings if event_type is POST and naki_idx is >= 2
        adjusted_post_naki_idx = torch.clamp(naki_idx, max=POST_TURN_ACTION_DIM - 1)
        post_naki_emb = self.post_naki_embedding(
            adjusted_post_naki_idx
        )  # Shape: (B, T, EMBD_SIZE)
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
        )  # Shape: (B, T, EMBD_SIZE)

        # Prepare attention mask: (B, 150) -> (B, 1, 150) for broadcasting over queries
        attn_mask = padding_mask.unsqueeze(1)  # (B, 1, T)

        for block in self.blocks:
            x = block(x, attn_mask)

        # Zero out encoder outputs at padded positions to prevent signal leakage
        x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

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
        encoder_padding_mask: torch.tensor,
    ):
        """
        Args:
            enc_out: (B, 150, EMBD_SIZE) tensor
            state_obj_tensor_batch: Tuple of (B, 3, 37), (B, 7, 4), (B, 4) tensors
            action_mask_batch: (B, action_dim) tensor
            y_tensor: (B,) tensor
            head: str indicating the head to use
            encoder_padding_mask: (B, 150) boolean tensor where True = valid position, False = padded (REQUIRED)
        """
        assert torch.all(
            (action_mask_batch == 0) | (action_mask_batch == 1)
        ), "action_mask_batch must be binary"
        assert torch.any(
            action_mask_batch, dim=1
        ).all(), "Each sample must have at least one allowed action"

        # Prepare encoder mask for cross-attention: (B, 150) -> (B, 1, 150)
        encoder_mask = encoder_padding_mask.unsqueeze(1)  # (B, 1, T)

        x = self.state_net(state_obj_tensor_batch)  # [B, EMBD_SIZE]
        for block in self.blocks:
            x = block(x, enc_out, encoder_mask)
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
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(EMBD_SIZE)
        self.ln1 = nn.LayerNorm(EMBD_SIZE)
        self.ln2 = nn.LayerNorm(EMBD_SIZE)

    def forward(self, x, attn_mask):
        """
        Args:
            x: Input tensor of shape [B, T, EMBD_SIZE]
            attn_mask: Optional boolean tensor of shape [B, 1, T].
                       True = keep (valid position), False = mask out (padded position)
        """
        x_norm = self.ln1(x)
        x = x + self.self_attention(x_norm, x_norm, x_norm, attn_mask)
        x = x + self.ffwd(self.ln2(x))

        return x


class DecoderBlock(nn.Module):
    def __init__(self, EMBD_SIZE, num_heads):
        super().__init__()
        head_size = EMBD_SIZE // num_heads
        self.cross_attention_history_1 = MultiHeadAttention(num_heads, head_size)
        self.cross_attention_history_2 = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(EMBD_SIZE)
        self.ln1 = nn.LayerNorm(EMBD_SIZE)
        self.ln2 = nn.LayerNorm(EMBD_SIZE)
        self.ln3 = nn.LayerNorm(EMBD_SIZE)

    def forward(self, x, enc_out, encoder_mask):
        """
        Args:
            x: State embedding of shape [B, EMBD_SIZE]
            enc_out: Encoder output of shape [B, T, EMBD_SIZE]
            encoder_mask: Optional boolean tensor of shape [B, 1, T].
                          True = keep (valid position), False = mask out (padded position)
        """
        # Cross-attention pass 1: state queries history to get a context vector
        x_norm = self.ln1(x).unsqueeze(1)  # [B, 1, EMBD_SIZE]
        cross_attn_1 = self.cross_attention_history_1(enc_out, enc_out, x_norm, encoder_mask)  # [B, 1, EMBD_SIZE]
        cross_attn_1 = cross_attn_1.squeeze(1)  # [B, EMBD_SIZE]
        x = x + cross_attn_1  # [B, EMBD_SIZE]

        # Cross-attention pass 2: state queries history again after residual update (refinement)
        x_norm = self.ln2(x).unsqueeze(1)  # [B, 1, EMBD_SIZE] - query is post-pass-1 residual, then LN
        cross_attn_2 = self.cross_attention_history_2(enc_out, enc_out, x_norm, encoder_mask)  # [B, 1, EMBD_SIZE]
        cross_attn_2 = cross_attn_2.squeeze(1)  # [B, EMBD_SIZE]
        x = x + cross_attn_2  # [B, EMBD_SIZE]

        # Feed Forward Network
        x_norm = self.ln3(x)  # [B, EMBD_SIZE]
        ffwd = self.ffwd(x_norm)  # [B, EMBD_SIZE]
        x = x + ffwd  # [B, EMBD_SIZE]

        return x  # [B, EMBD_SIZE]


class StateNet(nn.Module):
    def __init__(self):
        super(StateNet, self).__init__()

        # Suit-aware convolutions
        # Input per suit: C*K = 3*4 = 12 channels, R = 9 ranks
        self.suit_conv1d_1 = nn.Conv1d(
            in_channels=12, out_channels=16, kernel_size=3, padding=1
        )
        self.suit_conv1d_2 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        
        # Honors MLP
        # Input: C*H*K = 3*7*4 = 84
        self.honors_fc1 = nn.Linear(84, 32)
        self.honors_fc2 = nn.Linear(32, 16)
        
        # Red fives features
        # Input: C*S*K = 3*3*4 = 36
        self.reds_fc = nn.Linear(36, 8)
        
        self.relu = nn.ReLU()

        # Calculate combined feature size
        # Suits: 16 features (after pooling and aggregating S=3)
        # Honors: 16 features
        # Reds: 8 features
        # Raw x1: 3*37*4 = 444
        suits_features = 16
        honors_features = 16
        reds_features = 8
        raw_x1_features = 3 * 37 * 4
        
        combined_features = suits_features + honors_features + reds_features + raw_x1_features
        
        # Main network to combine with x2 and x3
        self.fc_main_1 = nn.Linear(combined_features + 7 * 4 + 4, EMBD_SIZE)
        self.fc_main_2 = nn.Linear(EMBD_SIZE, EMBD_SIZE)
        self.relu_fc_main = nn.ReLU()

    def forward(self, state_obj):
        x1, x2, x3 = state_obj  # Unpack the tuple
        batch_size = x1.size(0)
        
        # Initial x1 shape: [batch_size, C=3, T=37, K=4]
        
        # --- 1. Process Suits (indices 0-26) ---
        # Extract Man (0-8), Pin (9-17), Sou (18-26)
        x1_suits = torch.stack([
            x1[:, :, 0:9, :],   # Man
            x1[:, :, 9:18, :],  # Pin
            x1[:, :, 18:27, :]  # Sou
        ], dim=1)  # Shape: [B, S=3, C=3, R=9, K=4]
        
        # Reshape to [B, S, C*K, R]
        x1_suits = x1_suits.permute(0, 1, 2, 4, 3).contiguous()  # [B, S, C, K, R]
        x1_suits = x1_suits.view(batch_size, 3, 12, 9)  # [B, S=3, C*K=12, R=9]
        
        # Merge suits into batch for Conv1d: [B*S, C*K, R]
        x1_suits = x1_suits.view(batch_size * 3, 12, 9)  # [B*3, 12, 9]
        
        # Apply Conv1d over rank
        suits_feat = self.suit_conv1d_1(x1_suits)  # [B*3, 16, 9]
        suits_feat = self.relu(suits_feat)
        suits_feat = self.suit_conv1d_2(suits_feat)  # [B*3, 16, 9]
        suits_feat = self.relu(suits_feat)
        
        # Global pooling over rank dimension
        suits_feat = torch.mean(suits_feat, dim=2)  # [B*3, 16]
        
        # Reshape back and aggregate over suits
        suits_feat = suits_feat.view(batch_size, 3, 16)  # [B, S=3, 16]
        suits_feat = torch.mean(suits_feat, dim=1)  # [B, 16]
        
        # --- 2. Process Honors (indices 27-33) ---
        x1_honors = x1[:, :, 27:34, :]  # Shape: [B, C=3, H=7, K=4]
        x1_honors = x1_honors.reshape(batch_size, -1)  # [B, C*H*K=84]
        
        honors_feat = self.honors_fc1(x1_honors)  # [B, 32]
        honors_feat = self.relu(honors_feat)
        honors_feat = self.honors_fc2(honors_feat)  # [B, 16]
        honors_feat = self.relu(honors_feat)
        
        # --- 3. Process Red Fives (indices 34-36) ---
        x1_reds = x1[:, :, 34:37, :]  # Shape: [B, C=3, S=3, K=4]
        x1_reds = x1_reds.reshape(batch_size, -1)  # [B, C*S*K=36]
        
        reds_feat = self.reds_fc(x1_reds)  # [B, 8]
        reds_feat = self.relu(reds_feat)
        
        # --- 4. Preserve raw x1 ---
        raw_x1_flat = x1.view(batch_size, -1)  # [B, 3*37*4=444]
        
        # --- 5. Combine all features ---
        x1_combined = torch.cat([
            suits_feat,     # [B, 16]
            honors_feat,    # [B, 16]
            reds_feat,      # [B, 8]
            raw_x1_flat     # [B, 444]
        ], dim=1)  # [B, 484]
        
        # Flatten x2
        x2_flat = x2.view(x2.size(0), -1)  # Shape: [batch_size, 28]
        
        # x3 remains as [batch_size, 4]
        
        # Concatenate all processed features
        combined = torch.cat(
            [x1_combined, x2_flat, x3], dim=1
        )  # Shape: [batch_size, 484 + 28 + 4 = 516]
        
        # Pass through main Linear layer
        x = self.fc_main_1(combined)  # Shape: [batch_size, EMBD_SIZE]
        x = self.relu_fc_main(x)  # Apply ReLU activation
        x = self.fc_main_2(x)
        x = self.relu_fc_main(x)
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # Fused projections for all heads at once
        self.key_proj = nn.Linear(EMBD_SIZE, num_heads * head_size, bias=True)
        self.query_proj = nn.Linear(EMBD_SIZE, num_heads * head_size, bias=True)
        self.value_proj = nn.Linear(EMBD_SIZE, num_heads * head_size, bias=True)
        self.proj = nn.Linear(num_heads * head_size, EMBD_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.attn_dropout_p = DROPOUT_RATIO

    def forward(self, k, v, q, attn_mask):
        """
        Args:
            k: Key tensor of shape [B, T, EMBD_SIZE]
            v: Value tensor of shape [B, T, EMBD_SIZE]
            q: Query tensor of shape [B, Q, EMBD_SIZE]
            attn_mask: Optional boolean tensor of shape [B, 1, T].
                       True = keep (valid position), False = mask out (padded position)
        """
        B, T, _ = k.shape
        Q = q.shape[1]
        H, HS = self.num_heads, self.head_size

        # Project all heads in one matmul, then split into heads
        k = self.key_proj(k).view(B, T, H, HS).transpose(1, 2)    # [B, H, T, HS]
        v = self.value_proj(v).view(B, T, H, HS).transpose(1, 2)  # [B, H, T, HS]
        q = self.query_proj(q).view(B, Q, H, HS).transpose(1, 2)  # [B, H, Q, HS]

        # Expand mask for heads: [B, 1, T] → [B, 1, 1, T] (broadcasts over H and Q)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, 1, T]

        dropout_p = self.attn_dropout_p if self.training else 0.0
        # scaled_dot_product_attention: boolean mask True=attend matches our convention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        # out: [B, H, Q, HS]

        # Guard: fully-masked positions produce NaN (same as original nan_to_num guard)
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, Q, H * HS)  # [B, Q, H*HS]
        out = self.proj(out)
        out = self.dropout(out)
        return out  # [B, Q, EMBD_SIZE]


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


class TransformerTensorProcessor:
    def __init__(self):
        pass

    def prepare_batches(
        self,
        all_halfturns_list: List[list[HalfTurn]],
        all_encoding_tokens_list: List[list[int]],
    ) -> Tuple[
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]
    ]:
        # Prepare batched tensors for all types of turns
        tensors_during_batch = []
        tensors_discard_batch = []
        tensors_post_batch = []

        for all_halfturns, all_encoding_tokens in zip(
            all_halfturns_list, all_encoding_tokens_list
        ):
            for halfturns, encoding_tokens in zip(all_halfturns, all_encoding_tokens):
                tensors_during, tensors_discard, tensors_post = self.build_tensors(
                    halfturns, encoding_tokens
                )

                tensors_during_batch.append(tensors_during)
                tensors_discard_batch.append(tensors_discard)
                tensors_post_batch.append(tensors_post)

        return self.stack_batches(
            tensors_during_batch, tensors_discard_batch, tensors_post_batch
        )

    def stack_batches(
        self, tensors_during_batch, tensors_discard_batch, tensors_post_batch
    ):
        # --------------------- During Turns ---------------------
        during_encodings, during_state_obj, during_filter, during_y, during_padding_mask = zip(
            *tensors_during_batch
        )
        during_x1, during_x2, during_x3 = zip(*during_state_obj)
        during_encodings = torch.cat(during_encodings, dim=0)  # Shape: (sum_N, 150)
        during_state_obj = (
            torch.cat(during_x1, dim=0),
            torch.cat(during_x2, dim=0),
            torch.cat(during_x3, dim=0),
        )
        during_filter = torch.cat(during_filter, dim=0)
        during_y = torch.cat(during_y, dim=0)
        during_padding_mask = torch.cat(during_padding_mask, dim=0)  # Shape: (sum_N, 150)

        # --------------------- Discard Turns ---------------------
        discard_encodings, discard_state_obj, discard_filter, discard_y, discard_padding_mask = zip(
            *tensors_discard_batch
        )
        discard_x1, discard_x2, discard_x3 = zip(*discard_state_obj)
        discard_encodings = torch.cat(discard_encodings, dim=0)
        discard_state_obj = (
            torch.cat(discard_x1, dim=0),
            torch.cat(discard_x2, dim=0),
            torch.cat(discard_x3, dim=0),
        )
        discard_filter = torch.cat(discard_filter, dim=0)
        discard_y = torch.cat(discard_y, dim=0)
        discard_padding_mask = torch.cat(discard_padding_mask, dim=0)  # Shape: (sum_N, 150)

        # --------------------- Post Turns ---------------------
        post_encodings, post_state_obj, post_filter, post_y, post_padding_mask = zip(*tensors_post_batch)
        post_x1, post_x2, post_x3 = zip(*post_state_obj)
        post_encodings = torch.cat(post_encodings, dim=0)
        post_state_obj = (
            torch.cat(post_x1, dim=0),
            torch.cat(post_x2, dim=0),
            torch.cat(post_x3, dim=0),
        )
        post_filter = torch.cat(post_filter, dim=0)
        post_y = torch.cat(post_y, dim=0)
        post_padding_mask = torch.cat(post_padding_mask, dim=0)  # Shape: (sum_N, 150)

        return (
            (during_encodings, during_state_obj, during_filter, during_y, during_padding_mask),
            (discard_encodings, discard_state_obj, discard_filter, discard_y, discard_padding_mask),
            (post_encodings, post_state_obj, post_filter, post_y, post_padding_mask),
        )

    def build_tensors(self, turns: List[HalfTurn], encoding_tokens: List[int]) -> Tuple[
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
    ]:
        """
        Processes the list of turns and builds tensors for each turn type.
        """
        during_turns, discard_turns, post_turns = [], [], []

        for turn in turns:
            if turn.type_ == HalfTurn.DURING:
                during_turns.append(turn)
            elif turn.type_ == HalfTurn.DISCARD:
                discard_turns.append(turn)
            elif turn.type_ == HalfTurn.POST:
                post_turns.append(turn)
            else:
                raise ValueError(f"Unknown turn type: {turn.type_}")

        encoding_token_tensor = torch.tensor(
            encoding_tokens, dtype=torch.int64, device=device
        )
        encoding_token_tensor = torch.nn.functional.pad(
            encoding_token_tensor, (0, 150 - len(encoding_tokens))
        )

        # Build tensors for each turn type
        tensors_during = self._build_during_tensor(during_turns, encoding_token_tensor)
        tensors_discard = self._build_discard_tensor(
            discard_turns, encoding_token_tensor
        )
        tensors_post = self._build_post_tensor(post_turns, encoding_token_tensor)

        return tensors_during, tensors_discard, tensors_post

    def _build_discard_tensor(
        self, discard_turns: List[DiscardTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds tensors for discard turns.

        Args:
            discard_turns (List[DiscardTurn]): List of DiscardTurn instances.
            encoding_token_tensor (torch.Tensor): Tensor of shape (150,).

        Returns:
            Tuple containing:
                - embeddings: Tensor of shape (sum_N, 150)
                - state_obj_tensors: Tuple of tensors (sum_N, 3, 37), (sum_N, 7, 4), (sum_N, 4)
                - filter_tensors: Tensor of shape (sum_N, 37)
                - y: Tensor of target labels
                - padding_mask: Boolean tensor of shape (sum_N, 150), True = valid position
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        encoding_indices = []

        for turn in discard_turns:
            encoding_idx = turn.encoding_idx
            assert encoding_idx < MAX_SEQUENCE_LENGTH

            # **Ensure 2D Tensor by Unsqueezing**
            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)

            X_embeddings.append(masked_encoding_token_tensor)
            encoding_indices.append(encoding_idx)

            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            hand_tensor = torch.tensor(
                turn.stateObj.hand_tensor, dtype=torch.float32, device=device
            )  # Shape: (37,)
            # Ensure maximum value is 1
            filter_tensor = torch.clamp(hand_tensor, max=1.0)  # Shape: (37,)

            # Get the target label (discarded tile index)
            discarded_tile_idx = TILE2IDX[turn.discarded_tile][0]

            state_obj_tensors.append((x1, x2, x3))
            filter_tensors.append(filter_tensor)
            y.append(discarded_tile_idx)

        # Concatenate lists into single tensors
        encoding_tokens_batch = (
            torch.cat(X_embeddings, dim=0)
            if X_embeddings
            else torch.empty(0, 150, device=device)
        )

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (
                torch.cat(x1_list, dim=0)
                if x1_list
                else torch.empty(0, 3, 37, 4, device=device)
            ),
            (
                torch.cat(x2_list, dim=0)
                if x2_list
                else torch.empty(0, 7, 4, device=device)
            ),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, 4, device=device),
        )

        filter_tensors_2d = [
            ft.unsqueeze(0) for ft in filter_tensors
        ]  # List of (1, 37)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0)
            if filter_tensors_2d
            else torch.empty(0, 37, device=device)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == 37
        if action_mask_batch.numel() > 0:
            assert torch.any(
                action_mask_batch, dim=1
            ).all(), "Each sample must have at least one allowed action"

        y_tensor = torch.tensor(y, dtype=torch.long, device=device)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long, device=device)  # (N,)
            position_indices = torch.arange(150, device=device).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool, device=device)

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask,
        )

    def _build_during_tensor(
        self, during_turns: List[DuringTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds tensors for during turns.

        Args:
            during_turns (List[DuringTurn]): List of DuringTurn instances.
            encoding_token_tensor (torch.Tensor): Tensor of shape (150,).

        Returns:
            Tuple containing:
                - embeddings: Tensor of shape (sum_N, 150)
                - state_obj_tensors: Tuple of tensors (sum_N, 3, 37), (sum_N, 7, 4), (sum_N, 4)
                - filter_tensors: Tensor of shape (sum_N, 71)
                - y: Tensor of target labels
                - padding_mask: Boolean tensor of shape (sum_N, 150), True = valid position
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        encoding_indices = []

        for turn in during_turns:
            n_decisions = sum(map(len, turn.decisions))
            if n_decisions == 0:
                continue

            encoding_idx = turn.encoding_idx
            assert encoding_idx < MAX_SEQUENCE_LENGTH

            # **Ensure 2D Tensor by Unsqueezing**
            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)

            X_embeddings.append(masked_encoding_token_tensor)
            encoding_indices.append(encoding_idx)

            # Convert stateObj to tensors
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)
            state_obj_tensors.append((x1, x2, x3))  # Each x1, x2, x3 is a tensor

            # Create filter_tensor
            filter_tensor = torch.zeros(71, dtype=torch.float32, device=device)
            filter_tensor[0] = 1.0  # pass

            target = 0  # default to pass

            for decision_idx, decision_type in enumerate(turn.decisions):
                if not decision_type:
                    continue

                if decision_idx == DECISION_AGARI_IDX:
                    filter_tensor[1] = 1.0
                    if decision_type[0].executed:
                        target = 1

                elif decision_idx == DECISION_REACH_IDX:
                    filter_tensor[2] = 1.0
                    if decision_type[0].executed:
                        target = 2

                elif decision_idx == DECISION_NAKI_IDX:
                    for decision in decision_type:
                        assert isinstance(decision, NakiDecision)
                        meld = decision.naki
                        idx = meld.get_during_turn_filter_idx()
                        filter_tensor[idx] = 1.0
                        if decision.executed:
                            target = idx

                else:
                    raise ValueError(f"Invalid decision_idx: {decision_idx}")

            y.append(target)
            filter_tensors.append(filter_tensor)  # Shape: (71,)

        encoding_tokens_batch = (
            torch.cat(X_embeddings, dim=0)
            if X_embeddings
            else torch.empty(0, 150, device=device)
        )

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (
                torch.cat(x1_list, dim=0)
                if x1_list
                else torch.empty(0, 3, 37, 4, device=device)
            ),
            (
                torch.cat(x2_list, dim=0)
                if x2_list
                else torch.empty(0, 7, 4, device=device)
            ),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, 4, device=device),
        )

        filter_tensors_2d = [
            ft.unsqueeze(0) for ft in filter_tensors
        ]  # Each filter_tensor is now (1, 71)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0)
            if filter_tensors_2d
            else torch.empty(0, 71, device=device)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == 71
        if action_mask_batch.numel() > 0:
            assert (
                torch.sum(action_mask_batch, dim=1).min() >= 2
            ), "Each sample must have at least two allowed actions"

        y_tensor = torch.tensor(y, dtype=torch.long, device=device)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long, device=device)  # (N,)
            position_indices = torch.arange(150, device=device).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool, device=device)

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask,
        )

    def _build_post_tensor(
        self, post_turns: List[PostTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds tensors for post turns.

        Args:
            post_turns (List[PostTurn]): List of PostTurn instances.
            encoding_token_tensor (torch.Tensor): Tensor of shape (150,).

        Returns:
            Tuple containing:
                - embeddings: Tensor of shape (sum_N, 150)
                - state_obj_tensors: Tuple of tensors (sum_N, 3, 37), (sum_N, 7, 4), (sum_N, 4)
                - filter_tensors: Tensor of shape (sum_N, 154)
                - y: Tensor of target labels
                - padding_mask: Boolean tensor of shape (sum_N, 150), True = valid position
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        encoding_indices = []
        players = [0, 1, 2, 3]

        for turn in post_turns:
            decisions = turn.decisions  # type: List[List[List[Decision]]]
            n_decisions = sum(len(item) for sublist in decisions for item in sublist)
            if n_decisions == 0:
                continue

            encoding_idx = turn.encoding_idx
            assert encoding_idx < MAX_SEQUENCE_LENGTH

            # **Ensure 2D Tensor by Unsqueezing**
            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)

            # Create state object tensor
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            # TODO: instead of going player by player, go decision type by decision type
            action_player = turn.player
            for player_idx in players[action_player + 1 :] + players[:action_player]:
                if sum(map(len, decisions[player_idx])) == 0:
                    continue

                contains_executed = False
                result = 0
                decision_mask = torch.zeros(154, dtype=torch.float32, device=device)

                for decision_idx, decision_type in enumerate(decisions[player_idx]):
                    if len(decision_type) == 0:
                        continue

                    decision_mask[0] = 1.0  # pass

                    if decision_idx == DECISION_AGARI_IDX:
                        decision_mask[1] = 1.0
                        if decision_type[0].executed:
                            result = 1
                            contains_executed = True

                    elif decision_idx == DECISION_REACH_IDX:
                        decision_mask[2] = 1.0
                        if decision_type[0].executed:
                            result = 2
                            contains_executed = True

                    elif decision_idx == DECISION_NAKI_IDX:
                        for meld_decision in decision_type:
                            idx = meld_decision.naki.get_post_turn_filter_idx()
                            decision_mask[idx] = 1.0

                            if meld_decision.executed:
                                result = idx
                                contains_executed = True

                    else:
                        raise ValueError(f"Invalid decision_idx: {decision_idx}")

                X_embeddings.append(masked_encoding_token_tensor)
                state_obj_tensors.append((x1, x2, x3))
                filter_tensors.append(decision_mask)
                y.append(result)
                encoding_indices.append(encoding_idx)

                if contains_executed:
                    break

        encoding_tokens_batch = (
            torch.cat(X_embeddings, dim=0)
            if X_embeddings
            else torch.empty(0, 150, device=device)
        )

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (
                torch.cat(x1_list, dim=0)
                if x1_list
                else torch.empty(0, 3, 37, 4, device=device)
            ),
            (
                torch.cat(x2_list, dim=0)
                if x2_list
                else torch.empty(0, 7, 4, device=device)
            ),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, 4, device=device),
        )

        filter_tensors_2d = [
            ft.unsqueeze(0) for ft in filter_tensors
        ]  # Each filter_tensor is now (1, 154)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0)
            if filter_tensors_2d
            else torch.empty(0, 154, device=device)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == 154
        if action_mask_batch.numel() > 0:
            assert (
                torch.sum(action_mask_batch, dim=1).min() >= 2
            ), "Each sample must have at least two allowed actions"

        y_tensor = torch.tensor(y, dtype=torch.long, device=device)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long, device=device)  # (N,)
            position_indices = torch.arange(150, device=device).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool, device=device)

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask,
        )

    def _convert_state_obj_to_tensors(
        self, stateObj
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a StateObject instance into tensors.

        Args:
            stateObj (StateObject): The state object to convert.

        Returns:
            Tuple containing:
                - x1: Tensor of shape (1, 3, 37, 4)
                - x2: Tensor of shape (1, 7, 4)
                - x3: Tensor of shape (1, 4)
        """
        # x1: Contains hand_tensor, remaining_tiles_pov, sutehai_tensor
        # Transform hand_tensor
        hand_tensor = torch.tensor(
            stateObj.hand_tensor, dtype=torch.long, device=device
        )
        range_tensor = torch.arange(4, device=device)
        hand_tensor = (range_tensor < hand_tensor.unsqueeze(1)).float().unsqueeze(0) # Shape: (1, 37, 4)

        # Transform remaining_tiles_pov
        remaining_tiles_pov = torch.tensor(
            stateObj.remaining_tiles_pov, dtype=torch.long, device=device
        )
        remaining_tiles_pov = (range_tensor < remaining_tiles_pov.unsqueeze(1)).float().unsqueeze(0) # Shape: (1, 37, 4)

        # Transform sutehai_tensor
        sutehai_tensor = torch.tensor(
            stateObj.sutehai_tensors[stateObj.player], dtype=torch.long, device=device
        )
        sutehai_tensor = (range_tensor < sutehai_tensor.unsqueeze(1)).float().unsqueeze(0) # Shape: (1, 37, 4)

        # **Ensure 2D Tensor by Stacking Along New Dimension**
        x1 = torch.stack(
            [hand_tensor, remaining_tiles_pov, sutehai_tensor], dim=1
        )  # Shape: (1, 3, 37, 4)

        # x2: Contains scores, parent_tensor, parent_rounds_remaining, double_reaches, reaches, ippatsu, is_menzen
        scores = torch.tensor(
            stateObj.scores, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        parent_tensor = torch.tensor(
            stateObj.parent_tensor, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        parent_rounds_remaining = torch.tensor(
            stateObj.parent_rounds_remaining, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        double_reaches = torch.tensor(
            stateObj.double_reaches, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        reaches = torch.tensor(
            stateObj.reaches, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        ippatsu = torch.tensor(
            stateObj.ippatsu, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)
        is_menzen = torch.tensor(
            stateObj.is_menzen, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 4)

        x2 = torch.stack(
            [
                scores,
                parent_tensor,
                parent_rounds_remaining,
                double_reaches,
                reaches,
                ippatsu,
                is_menzen,
            ],
            dim=1,
        )  # Shape: (1, 7, 4)

        # x3: Contains remaining_tsumo, kyotaku, honba, rounds_remaining
        x3 = torch.tensor(
            [
                [
                    stateObj.remaining_tsumo,
                    stateObj.kyotaku,
                    stateObj.honba,
                    stateObj.rounds_remaining,
                ]
            ],
            dtype=torch.float32,
            device=device,
        )  # Shape: (1, 4)

        return x1, x2, x3


if __name__ == "__main__":
    model = TransformerConvModel().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M params")
