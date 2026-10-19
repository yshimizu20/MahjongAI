import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from collections import Counter, namedtuple

sys.path.append("../..")

# per-head model output: masked logits (B, A), mean cross-entropy, and the targets (B,)
HeadOutput = namedtuple("HeadOutput", ["logits", "loss", "y"])

from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.decision import Decision, NakiDecision
from MahjongAI.utils.constants import (
    DECISION_AGARI_IDX,
    DECISION_REACH_IDX,
    DECISION_NAKI_IDX,
    DECISION_ABORT_IDX,
    TILE2IDX,
    MAX_SEQUENCE_LENGTH,
    EventStateTypes,
)
from MahjongAI.utils.legal import legal_discard_mask
from mahjong.constants import EAST, WEST

# Hyperparameters
DISCARD_ACTION_DIM = 37
DURING_TURN_ACTION_DIM = 72  # 0 pass, 1 agari (tsumo), 2 riichi, 3-36 ankan, 37-70 kakan, 71 kyuushu abort
DURING_ABORT_IDX = 71
POST_TURN_ACTION_DIM = 154  # pass, agari (ron), naki
MAX_ACTION_LEN = 150
# x1 planes, all from the ACTOR's point of view, opponents in relative seat order
# (1 = shimocha / next to play, 2 = toimen, 3 = kamicha / the seat a chi is called from):
#   0 own hand, 1 unseen tile counts, 2 own discards, 3 drawn tile,
#   4-6 opponents' discards, 7-9 tiles safe against each opponent's riichi (zero while not in riichi)
N_X1_PLANES = 10
# x3 scalars: remaining_tsumo/70, kyotaku/4, honba/8, rounds_remaining/8 (0 in an extension round), then the round
# wind (bakaze) one-hot E, S, W (no North slot: 0 North rounds in 2.2M games; process_xml rejects one, T-038)
N_X3 = 7
EMBD_SIZE = 64
N_HEADS = 8
N_LAYERS = 1
DROPOUT_RATIO = 0.2

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")


class EncoderConvModel(nn.Module):
    def __init__(self, n_layers=N_LAYERS, verbose=True):
        super().__init__()
        self.token_embedder = TokenEmbedder()
        self.blocks = nn.Sequential(*[EncoderBlock(EMBD_SIZE, N_HEADS) for _ in range(n_layers)])
        self.state_net = StateNet()
        self.ln_f = nn.LayerNorm(EMBD_SIZE)
        self.discard_head = nn.Linear(EMBD_SIZE, DISCARD_ACTION_DIM)
        self.during_head = nn.Linear(EMBD_SIZE, DURING_TURN_ACTION_DIM)
        self.post_head = nn.Linear(EMBD_SIZE, POST_TURN_ACTION_DIM)
        self.heads = {
            "discard": self.discard_head,
            "during": self.during_head,
            "post": self.post_head,
        }
        self.tensor_processor = TransformerTensorProcessor()

        self.apply(self._init_weights)

        if verbose:
            self.print_model_summary()

    def print_model_summary(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        te = self.token_embedder
        sn = self.state_net

        print("\n" + "=" * 70)
        print("ENCODER-ONLY (EARLY FUSION) MODEL CONFIGURATION")
        print("=" * 70)
        n_layers = len(self.blocks)
        print(f"  EMBD_SIZE={EMBD_SIZE}, N_HEADS={N_HEADS}, N_LAYERS={n_layers}")
        print(f"  Discard actions={DISCARD_ACTION_DIM}, During actions={DURING_TURN_ACTION_DIM}, Post actions={POST_TURN_ACTION_DIM}")
        print("=" * 70)

        print("\nTOKEN EMBEDDER")
        print("-" * 70)
        print(f"  {'Embedding: type (4→64):':<45}{count(te.type_embedding):>10,}")
        print(f"  {'Embedding: tile (37→64):':<45}{count(te.tile_embedding):>10,}")
        print(f"  {'Embedding: tsumogiri (1→64):':<45}{count(te.tsumogiri_embedding):>10,}")
        print(f"  {f'Embedding: during_naki ({DURING_TURN_ACTION_DIM}→64):':<45}{count(te.during_naki_embedding):>10,}")
        print(f"  {'Embedding: post_naki (154→64):':<45}{count(te.post_naki_embedding):>10,}")
        print(f"  {'Position/player coding (non-trainable):':<45}{'0':>10}")
        print(f"  {'TokenEmbedder subtotal:':<45}{count(te):>10,}")

        print("\nENCODER BLOCKS")
        print("-" * 70)
        enc_block = self.blocks[0]
        print(f"  {f'EncoderBlock x{n_layers}:':<45}")
        print(f"    {'MultiHeadAttention:':<43}{count(enc_block.self_attention):>10,}")
        print(f"    {'FeedForward:':<43}{count(enc_block.ffwd):>10,}")
        print(f"    {'LayerNorm x2:':<43}{count(enc_block.ln1) + count(enc_block.ln2):>10,}")
        print(f"  {'Blocks subtotal:':<45}{count(self.blocks):>10,}")

        print("\nSTATE NET")
        print("-" * 70)
        print(f"  {f'Conv1d suit_conv1d_1 ({N_X1_PLANES * 4}→16, k=3):':<45}{count(sn.suit_conv1d_1):>10,}")
        print(f"  {'Conv1d suit_conv1d_2 (16→16, k=3):':<45}{count(sn.suit_conv1d_2):>10,}")
        print(f"  {f'Linear honors_fc1 ({N_X1_PLANES * 28}→32):':<45}{count(sn.honors_fc1):>10,}")
        print(f"  {'Linear honors_fc2 (32→16):':<45}{count(sn.honors_fc2):>10,}")
        print(f"  {f'Linear reds_fc ({N_X1_PLANES * 12}→8):':<45}{count(sn.reds_fc):>10,}")
        print(f"  {f'Linear fc_main_1 ({sn.fc_main_1.in_features}→64):':<45}{count(sn.fc_main_1):>10,}")
        print(f"  {'Linear fc_main_2 (64→64):':<45}{count(sn.fc_main_2):>10,}")
        print(f"  {'StateNet subtotal:':<45}{count(sn):>10,}")

        print("\nOUTPUT HEADS")
        print("-" * 70)
        print(f"  {'ln_f LayerNorm(64):':<45}{count(self.ln_f):>10,}")
        print(f"  {f'discard_head Linear(64→{DISCARD_ACTION_DIM}):':<45}{count(self.discard_head):>10,}")
        print(f"  {f'during_head Linear(64→{DURING_TURN_ACTION_DIM}):':<45}{count(self.during_head):>10,}")
        print(f"  {f'post_head Linear(64→{POST_TURN_ACTION_DIM}):':<45}{count(self.post_head):>10,}")

        total = count(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print()
        print("=" * 70)
        print(f"  {'TOTAL PARAMETERS:':<45}{total:>10,}")
        print(f"  {'Trainable:':<45}{trainable:>10,}")
        print(f"  {'Non-trainable:':<45}{total - trainable:>10,}")
        print("=" * 70)
        print(f"  Model size: ~{total / 1e6:.3f}M parameters")
        print("=" * 70 + "\n")

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
        head_core=None,
    ) -> dict:
        """{head: HeadOutput} of a batch of games. head_core: the per-head network (default self.head_core, eager);
        the trainer passes its compiled copy of it for training updates only, and evaluation calls the eager module
        (T-050). Preparation, sampling, the device moves, the safety checks and the unique step of the token
        embedder run here, eager, around it."""
        core = self.head_core if head_core is None else head_core
        (
            tensors_during,
            tensors_discard,
            tensors_post,
        ) = self.tensor_processor.prepare_batches(all_halfturns_list, all_encoding_tokens_list)

        results = {}
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
            state_obj_tensor_batch = tuple(tensor.to(device) for tensor in state_obj_tensor_batch)
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)
            padding_mask_batch = padding_mask_batch.to(device)

            # Assertions, eager and before the per-head network: a compiled network turns tensor asserts into no-ops
            # (T-050 P1). Same conditions and messages as before; the target check ran after the logits until T-050.
            assert torch.all((action_mask_batch == 0) | (action_mask_batch == 1)), "action_mask_batch must be binary"
            assert torch.any(action_mask_batch, dim=1).all(), "Each sample must have at least one allowed action"
            assert torch.all(
                action_mask_batch[torch.arange(y_tensor.size(0)), y_tensor] == 1
            ), "Target action is masked out as invalid!"
            check_negative_targets(y_tensor)

            # Step 1a: the unique (token, position) pairs of the history (eager: the output shape depends on the data)
            unique_codes, inverse = self.token_embedder.unique_codes(encoding_tokens_batch)

            # Steps 1b-6: the per-head network, through the loss
            logits, loss = core(head, unique_codes, inverse, *state_obj_tensor_batch, action_mask_batch, y_tensor,
                                padding_mask_batch)

            assert (
                logits.shape == action_mask_batch.shape
            ), f"Logits shape {logits.shape} and mask shape {action_mask_batch.shape} do not match."
            results[head] = HeadOutput(logits, loss, y_tensor)

        return results

    def head_core(self, head, unique_codes, inverse, x1, x2, x3, action_mask_batch, y_tensor, padding_mask_batch):
        """(masked logits (B, A), mean cross-entropy) of one head from the eager preparation: the unique history codes
        and their inverse (TokenEmbedder.unique_codes), the state tensors, the binary action mask, the targets and the
        padding mask. The function the trainer compiles (T-050): no tensor assert, check or raise in here (they run
        eager in forward, before it), no data-dependent shape."""
        # Step 1b: Embed history tokens
        history_emb = self.token_embedder.embed_unique(unique_codes, inverse)  # (B, 150, EMBD_SIZE)

        # Step 2: State embedding
        state_emb = self.state_net((x1, x2, x3))  # (B, EMBD_SIZE)
        state_emb = state_emb.unsqueeze(1)  # (B, 1, EMBD_SIZE)

        # Step 3: Early fusion — concatenate history + state token
        fused = torch.cat([history_emb, state_emb], dim=1)  # (B, 151, EMBD_SIZE)

        # Step 4: Extend padding mask (state token is always valid)
        B = padding_mask_batch.size(0)
        state_mask = torch.ones(B, 1, dtype=torch.bool, device=padding_mask_batch.device)
        fused_mask = torch.cat([padding_mask_batch, state_mask], dim=1)  # (B, 151)

        # Step 5: Bidirectional self-attention (no causal mask). Only the state token reaches the heads, so the
        # last block computes its row alone (keys / values still come from every position): exact, T-011
        attn_mask = fused_mask.unsqueeze(1)  # (B, 1, 151)
        x = fused
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask, state_row_only=(i == len(self.blocks) - 1))  # last: (B, 1, EMBD_SIZE)

        # Step 6: Extract state token (last position), output head, loss
        final_state = x[:, -1, :]  # (B, EMBD_SIZE)
        final_state = self.ln_f(final_state)
        logits = self.heads[head](final_state)

        # Mask out invalid actions
        logits = logits.masked_fill(action_mask_batch == 0, -1e9)

        loss = F.cross_entropy(logits, y_tensor, reduction="mean")
        return logits, loss


def check_negative_targets(y_tensor):
    """Raise as F.cross_entropy does for a negative target that is not its ignore_index (-100): eager, before the
    per-head network, which may be compiled (T-050 P1). Such a target passes the target-legality check (it indexes
    the mask from the end); a target >= the action count fails that check's indexing first, as before."""
    bad = (y_tensor < 0) & (y_tensor != -100)
    if bad.any():
        raise IndexError(f"Target {int(y_tensor[bad][0])} is out of bounds.")


class TokenEmbedder(nn.Module):
    """Converts encoding tokens to summed embeddings (no attention blocks)."""

    def __init__(self):
        super().__init__()

        self.register_buffer("empty_embedding", torch.zeros((1, 1, EMBD_SIZE)))
        self.type_embedding = nn.Embedding(4, EMBD_SIZE)  # embedding_type: 1-4
        self.tile_embedding = nn.Embedding(37, EMBD_SIZE)  # tile_encoding: 0-36
        self.tsumogiri_embedding = nn.Embedding(1, EMBD_SIZE)
        self.during_naki_embedding = nn.Embedding(DURING_TURN_ACTION_DIM, EMBD_SIZE)
        self.post_naki_embedding = nn.Embedding(POST_TURN_ACTION_DIM, EMBD_SIZE)

        self.register_buffer("position_coding_table", self._get_position_encoding())
        self.register_buffer("player_coding_table", self._get_player_encoding())

    @staticmethod
    def _get_position_encoding():
        position = torch.arange(MAX_ACTION_LEN).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, EMBD_SIZE, 2).float() * -(np.log(10000.0) / EMBD_SIZE))
        pos_encoding = torch.zeros(MAX_ACTION_LEN, EMBD_SIZE)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.requires_grad_(False)  # Ensure non-trainable

    @staticmethod
    def _get_player_encoding():
        encoding = torch.empty((4, EMBD_SIZE), dtype=torch.float32)

        base = torch.pow(10000, torch.arange(0, EMBD_SIZE, 2, dtype=torch.float32) / EMBD_SIZE)

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
        Converts encoding tokens to embeddings (no attention blocks).

        A slot's embedding depends only on its (token, position) pair, and the histories of a batch share most pairs
        (every sample's history is a prefix of its round's tokens): 2.5-19 % of the B * T pairs are unique. So the
        summed embedding is computed once per unique pair and gathered back (T-047 M2). The per-element float ops
        and their order are the full-size computation's, so the output is bitwise equal to it; the backward reaches
        the tables through one gather over the unique rows.

        Args:
            encoding_tokens_batch: (B, 150) tensor of encoding tokens

        Returns:
            x: (B, 150, EMBD_SIZE) tensor of summed embeddings
        """
        return self.embed_unique(*self.unique_codes(encoding_tokens_batch))  # (B, T, EMBD_SIZE)

    def unique_codes(self, encoding_tokens_batch: torch.Tensor):
        """(unique codes (U,), inverse (B, T)) of the (token, position) pairs: the eager step of forward (its output
        shape depends on the data; T-050 keeps it out of the compiled network). Raises as the type embedding lookup
        does for a token whose embedding type is out of its table (the lookup may run compiled)."""
        T = encoding_tokens_batch.size(1)
        assert T <= MAX_ACTION_LEN
        positions = torch.arange(T, device=encoding_tokens_batch.device)
        codes = encoding_tokens_batch * MAX_ACTION_LEN + positions  # (B, T): one code per (token, position)
        unique_codes, inverse = torch.unique(codes, return_inverse=True)  # (U,), (B, T)
        type_idx = torch.clamp(((torch.div(unique_codes, MAX_ACTION_LEN, rounding_mode="floor") >> 10) & 7) - 1, min=0)
        if (type_idx >= self.type_embedding.num_embeddings).any():
            raise IndexError("index out of range in self")
        return unique_codes, inverse

    def embed_unique(self, unique_codes, inverse):
        """The (B, T, EMBD_SIZE) embedding from unique_codes' output: each unique pair's summed embedding, gathered
        back."""
        tokens = torch.div(unique_codes, MAX_ACTION_LEN, rounding_mode="floor")
        rows = self._embed_codes(tokens, self.position_coding_table[unique_codes - tokens * MAX_ACTION_LEN])
        return F.embedding(inverse, rows)  # (B, T, EMBD_SIZE)

    def _embed_codes(self, encoding_tokens, pos_emb):
        """The summed embedding of tokens (U,) at positions whose encodings are pos_emb (U, EMBD_SIZE): (U,
        EMBD_SIZE). The ops of 468d71c's full-size forward, in its order."""
        event_type = (encoding_tokens >> 13) & 7
        embedding_type = (encoding_tokens >> 10) & 7
        player_idx = (encoding_tokens >> 8) & 3
        tile_encoding = encoding_tokens & 0x7F
        tsumogiri = (encoding_tokens >> 7) & 1
        naki_idx = encoding_tokens & 0xFF

        # retrieve type embeddings if event_type is not EMPTY
        adjusted_embedding_type = torch.clamp(embedding_type - 1, min=0)
        type_emb = self.type_embedding(adjusted_embedding_type)  # Shape: (U, EMBD_SIZE)
        mask = event_type != EventStateTypes.EMPTY
        type_emb = type_emb * mask.unsqueeze(-1).float()

        # retrieve tile embeddings if event_type is DISCARD or NEW_DORA
        # clamp tile_encoding to max 36
        adjusted_tile_encoding = torch.clamp(tile_encoding, max=36)
        tile_emb = self.tile_embedding(adjusted_tile_encoding)  # Shape: (U, EMBD_SIZE)
        mask = (event_type == EventStateTypes.DISCARD) | (event_type == EventStateTypes.NEW_DORA)
        tile_emb = tile_emb * mask.unsqueeze(-1).float()

        # Retrieve player encodings if event_type is DISCARD, DURING, or POST
        player_idx = player_idx.to(self.player_coding_table.device)
        player_emb = self.player_coding_table[player_idx]  # Shape: (U, EMBD_SIZE)
        mask = (
            (event_type == EventStateTypes.DISCARD)
            | (event_type == EventStateTypes.DURING)
            | (event_type == EventStateTypes.POST)
        ).to(player_emb.device)
        player_emb = player_emb * mask.unsqueeze(-1).float()

        # tsumogiri embedding only if event_type is DISCARD and tsumogiri is 1
        tsumogiri_emb = self.tsumogiri_embedding(torch.tensor([0], device=self.type_embedding.weight.device))  # (1, E)
        mask = (event_type == EventStateTypes.DISCARD) & (tsumogiri == 1)
        tsumogiri_emb = tsumogiri_emb * mask.unsqueeze(-1).float()

        # position encodings if embedding type is not EMPTY
        mask = event_type != EventStateTypes.EMPTY
        pos_emb = pos_emb * mask.unsqueeze(-1).float()

        # Retrieve during naki encodings if event_type is DURING and naki_idx is >= 3
        adjusted_during_naki_idx = torch.clamp(naki_idx, max=DURING_TURN_ACTION_DIM - 1)
        during_naki_emb = self.during_naki_embedding(adjusted_during_naki_idx)  # Shape: (U, EMBD_SIZE)
        mask = (event_type == EventStateTypes.DURING) & (naki_idx >= 3)
        during_naki_emb = during_naki_emb * mask.unsqueeze(-1).float()

        # retrieve post naki encodings if event_type is POST and naki_idx is >= 2
        adjusted_post_naki_idx = torch.clamp(naki_idx, max=POST_TURN_ACTION_DIM - 1)
        post_naki_emb = self.post_naki_embedding(adjusted_post_naki_idx)  # Shape: (U, EMBD_SIZE)
        mask = (event_type == EventStateTypes.POST) & (naki_idx >= 2)
        post_naki_emb = post_naki_emb * mask.unsqueeze(-1).float()

        x = (
            self.empty_embedding[0]
            + type_emb
            + tile_emb
            + player_emb
            + tsumogiri_emb
            + pos_emb
            + during_naki_emb
            + post_naki_emb
        )  # Shape: (U, EMBD_SIZE)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, EMBD_SIZE, num_heads):
        super().__init__()
        head_size = EMBD_SIZE // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(EMBD_SIZE)
        self.ln1 = nn.LayerNorm(EMBD_SIZE)
        self.ln2 = nn.LayerNorm(EMBD_SIZE)

    def forward(self, x, attn_mask, state_row_only=False):
        """
        Args:
            x: Input tensor of shape [B, T, EMBD_SIZE]
            attn_mask: Optional boolean tensor of shape [B, 1, T].
                       True = keep (valid position), False = mask out (padded position)
            state_row_only: compute only the last (state token) row: returns [B, 1, EMBD_SIZE]. Equal to
                       forward(x, attn_mask)[:, -1:] (every row is computed independently from the shared K / V)
        """
        x_norm = self.ln1(x)
        q, x = (x_norm[:, -1:], x[:, -1:]) if state_row_only else (x_norm, x)
        x = x + self.self_attention(x_norm, x_norm, q, attn_mask)
        x = x + self.ffwd(self.ln2(x))

        return x


def conv1d_k3_rank_major(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """conv(x.transpose(1, 2)).transpose(1, 2) for a Conv1d(kernel 3, padding 1, stride 1), on rank-major input:
    x [N, R, Cin] -> [N, R, Cout]. The 3 shifted rank slices side by side, then one matmul with conv's own weight and
    bias (the module stays the parameter holder, so the state_dict is unchanged)."""
    assert conv.kernel_size == (3,) and conv.padding == (1,) and conv.stride == (1,) and conv.dilation == (1,)
    assert conv.groups == 1
    R = x.shape[1]
    xp = F.pad(x, (0, 0, 1, 1))  # [N, R + 2, Cin]: a zero rank at both ends
    cols = torch.cat([xp[:, t:t + R] for t in range(3)], dim=2)  # [N, R, 3*Cin], column t*Cin + c = x[:, r + t - 1, c]
    weight = conv.weight.permute(0, 2, 1).reshape(conv.out_channels, -1)  # [Cout, 3*Cin], the same column order
    return F.linear(cols, weight, conv.bias)


class StateNet(nn.Module):
    def __init__(self):
        super(StateNet, self).__init__()

        C, K = N_X1_PLANES, 4

        # Suit-aware convolutions
        # Input per suit: C*K channels, R = 9 ranks
        self.suit_conv1d_1 = nn.Conv1d(in_channels=C * K, out_channels=16, kernel_size=3, padding=1)
        self.suit_conv1d_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        # Honors MLP
        # Input: C*H*K, H = 7
        self.honors_fc1 = nn.Linear(C * 7 * K, 32)
        self.honors_fc2 = nn.Linear(32, 16)

        # Red fives features
        # Input: C*S*K, S = 3
        self.reds_fc = nn.Linear(C * 3 * K, 8)

        self.relu = nn.ReLU()

        # Calculate combined feature size
        # Suits: 16 features (after pooling and aggregating S=3)
        # Honors: 16 features
        # Reds: 8 features
        # Raw x1: C*37*K
        suits_features = 16
        honors_features = 16
        reds_features = 8
        raw_x1_features = C * 37 * K  # 10 * 37 * 4 = 1480

        combined_features = suits_features + honors_features + reds_features + raw_x1_features

        # Main network to combine with x2 and x3
        self.fc_main_1 = nn.Linear(combined_features + 7 * 4 + N_X3, EMBD_SIZE)  # 1520 + 28 + 7 = 1555
        self.fc_main_2 = nn.Linear(EMBD_SIZE, EMBD_SIZE)
        self.relu_fc_main = nn.ReLU()

    def forward(self, state_obj):
        x1, x2, x3 = state_obj  # Unpack the tuple
        batch_size = x1.size(0)

        # Initial x1 shape: [batch_size, C=N_X1_PLANES, T=37, K=4]
        C, K = N_X1_PLANES, 4

        # --- 1. Process Suits (indices 0-26) ---
        # Man (0-8), Pin (9-17), Sou (18-26), suits merged into the batch and rank-major: [B*3, R=9, C*K], channel
        # c*K + k (the Conv1d channel order). Each Conv1d(k=3, pad=1) runs as one matmul over 3 shifted rank slices
        # with its own weights (T-011: the stock CPU Conv1d backward is slow); the state_dict is unchanged.
        x1_suits = x1[:, :, :27, :].reshape(batch_size, C, 3, 9, K)  # [B, C, S, R, K]
        x1_suits = x1_suits.permute(0, 2, 3, 1, 4).reshape(batch_size * 3, 9, C * K)  # [B*3, R, C*K]

        suits_feat = self.relu(conv1d_k3_rank_major(x1_suits, self.suit_conv1d_1))  # [B*3, 9, 16]
        suits_feat = self.relu(conv1d_k3_rank_major(suits_feat, self.suit_conv1d_2))  # [B*3, 9, 16]

        # Global pooling over rank dimension
        suits_feat = torch.mean(suits_feat, dim=1)  # [B*3, 16]

        # Reshape back and aggregate over suits
        suits_feat = suits_feat.view(batch_size, 3, 16)  # [B, S=3, 16]
        suits_feat = torch.mean(suits_feat, dim=1)  # [B, 16]

        # --- 2. Process Honors (indices 27-33) ---
        x1_honors = x1[:, :, 27:34, :]  # Shape: [B, C=10, H=7, K=4]
        x1_honors = x1_honors.reshape(batch_size, -1)  # [B, C*H*K=280]

        honors_feat = self.honors_fc1(x1_honors)  # [B, 32]
        honors_feat = self.relu(honors_feat)
        honors_feat = self.honors_fc2(honors_feat)  # [B, 16]
        honors_feat = self.relu(honors_feat)

        # --- 3. Process Red Fives (indices 34-36) ---
        x1_reds = x1[:, :, 34:37, :]  # Shape: [B, C=10, S=3, K=4]
        x1_reds = x1_reds.reshape(batch_size, -1)  # [B, C*S*K=120]

        reds_feat = self.reds_fc(x1_reds)  # [B, 8]
        reds_feat = self.relu(reds_feat)

        # --- 4. Preserve raw x1 ---
        raw_x1_flat = x1.view(batch_size, -1)  # [B, 10*37*4=1480]

        # --- 5. Combine all features ---
        x1_combined = torch.cat(
            [suits_feat, honors_feat, reds_feat, raw_x1_flat], dim=1  # [B, 16]  # [B, 16]  # [B, 8]  # [B, 1480]
        )  # [B, 1520]

        # Flatten x2
        x2_flat = x2.view(x2.size(0), -1)  # Shape: [batch_size, 28]

        # x3 remains as [batch_size, N_X3=7]

        # Concatenate all processed features
        combined = torch.cat([x1_combined, x2_flat, x3], dim=1)  # Shape: [batch_size, 1520 + 28 + 7 = 1555]

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
        k = self.key_proj(k).view(B, T, H, HS).transpose(1, 2)  # [B, H, T, HS]
        v = self.value_proj(v).view(B, T, H, HS).transpose(1, 2)  # [B, H, T, HS]
        q = self.query_proj(q).view(B, Q, H, HS).transpose(1, 2)  # [B, H, Q, HS]

        # Expand mask for heads: [B, 1, T] → [B, 1, 1, T] (broadcasts over H and Q)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, 1, T]

        dropout_p = self.attn_dropout_p if self.training else 0.0
        if Q == 1:
            out = self._single_query_attention(q, k, v, attn_mask, dropout_p)
        else:
            # scaled_dot_product_attention: boolean mask True=attend matches our convention
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        # out: [B, H, Q, HS]

        # Guard: fully-masked positions produce NaN (same as original nan_to_num guard)
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, Q, H * HS)  # [B, Q, H*HS]
        out = self.proj(out)
        out = self.dropout(out)
        return out  # [B, Q, EMBD_SIZE]

    @staticmethod
    def _single_query_attention(q, k, v, attn_mask, dropout_p):
        """SDPA for one query row, as broadcast multiply + sum (T-047 M1). In train mode SDPA takes its math path
        (dropout_p > 0 rules out the CPU flash kernel) and, in this torch build without MKL, each of its bmm runs a
        serial loop over the B*H [1, HS] x [HS, T] slices. With Q = 1 the scores are q . k per key, so
        (q * k).sum(-1) and (a * v).sum over the keys compute the same function. As the math path: scale 1 / sqrt(HS),
        masked keys get -inf, safe softmax (a fully masked row gives zero weights and finite gradients), dropout on the
        contiguous [B, H, 1, T] weights (the same shape, so the same RNG draws).
        q [B, H, 1, HS], k and v [B, H, T, HS], attn_mask [B, 1, 1, T] bool (True = attend) or None: [B, H, 1, HS]."""
        scores = (q * k).sum(-1).unsqueeze(2) * (q.shape[-1] ** -0.5)  # [B, H, 1, T]
        if attn_mask is not None:
            empty = ~attn_mask.any(-1, keepdim=True)  # [B, 1, 1, 1]: a row with no key to attend
            scores = scores.masked_fill(~(attn_mask | empty), float("-inf"))
            weights = torch.softmax(scores, dim=-1).masked_fill(empty, 0.0)
        else:
            weights = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            weights = F.dropout(weights, dropout_p, True)
        return (weights.transpose(-1, -2) * v).sum(2, keepdim=True)  # [B, H, 1, HS]


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


def _to_numpy(t):
    return t.numpy() if isinstance(t, torch.Tensor) else tuple(_to_numpy(x) for x in t)


def _to_torch(t):
    return tuple(_to_torch(x) for x in t) if isinstance(t, tuple) else torch.from_numpy(t)


class PrebuiltGame:
    """One game's per-round build_tensors() output, built in a data worker (T-011 D1). It stands in a batch's
    halfturns slot (the tokens slot holds None), and prepare_batches unpacks it where the serial path builds the game:
    the worker processor's tensor-level skips are added to the processor there, and a build error is raised there,
    after the rounds built before it (as in the serial path). Arrays travel as numpy: no shared-memory handles."""

    def __init__(self, rounds, skipped: dict, build_error=None):
        self.rounds = rounds  # [(during, discard, post)] of numpy, or the rounds built before build_error
        self.skipped = skipped
        self.build_error = build_error

    def unpack(self, processor: "TransformerTensorProcessor"):
        processor.skipped.update(self.skipped)
        for r in self.rounds:
            yield _to_torch(r)
        if self.build_error is not None:
            raise self.build_error


def build_game(all_halfturns, all_encoding_tokens, processor_cls=None) -> PrebuiltGame:
    """The data worker's tensor build of one parsed game: build_tensors() round by round with a fresh processor (the
    processor holds no state but `skipped`). A build error is kept, not raised: the trainer raises it at its own
    build point. processor_cls: tests force a skip or a failure with a subclass."""
    processor = (processor_cls or TransformerTensorProcessor)()
    rounds = []
    try:
        for h, t in zip(all_halfturns, all_encoding_tokens):
            rounds.append(_to_numpy(processor.build_tensors(h, t)))
    except Exception as e:
        return PrebuiltGame(rounds, dict(processor.skipped), e)
    return PrebuiltGame(rounds, dict(processor.skipped))


class TransformerTensorProcessor:
    def __init__(self):
        self.skipped = Counter()  # samples dropped by a tensor-level sanity check; gates.py asserts it stays empty

    @staticmethod
    def _relative_seats(tokens: torch.Tensor, actor: int) -> torch.Tensor:
        """Rewrite the `who` bits (8-9) of every seat-bearing token so that seat 0 is the actor.

        The model then sees the same history whichever absolute seat the actor sits in:
        0 = me, 1 = shimocha, 2 = toimen, 3 = kamicha. NEW_DORA and padding tokens carry no seat.
        """
        event_type = (tokens >> 13) & 7
        has_seat = (
            (event_type == EventStateTypes.DISCARD)
            | (event_type == EventStateTypes.DURING)
            | (event_type == EventStateTypes.POST)
        )
        relative = ((((tokens >> 8) & 3) - actor) % 4) << 8
        return torch.where(has_seat, (tokens & ~(3 << 8)) | relative, tokens)

    def prepare_batches(
        self,
        all_halfturns_list: List[list[HalfTurn]],
        all_encoding_tokens_list: List[list[int]],
    ) -> Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]]:
        # Prepare batched tensors for all types of turns
        tensors_during_batch = []
        tensors_discard_batch = []
        tensors_post_batch = []

        for all_halfturns, all_encoding_tokens in zip(all_halfturns_list, all_encoding_tokens_list):
            if isinstance(all_halfturns, PrebuiltGame):  # built in a data worker: its skips and errors land here
                rounds = all_halfturns.unpack(self)
            else:
                rounds = (self.build_tensors(h, t) for h, t in zip(all_halfturns, all_encoding_tokens))
            for tensors_during, tensors_discard, tensors_post in rounds:
                tensors_during_batch.append(tensors_during)
                tensors_discard_batch.append(tensors_discard)
                tensors_post_batch.append(tensors_post)

        return self.stack_batches(tensors_during_batch, tensors_discard_batch, tensors_post_batch)

    def stack_batches(self, tensors_during_batch, tensors_discard_batch, tensors_post_batch):
        # --------------------- During Turns ---------------------
        during_encodings, during_state_obj, during_filter, during_y, during_padding_mask = zip(*tensors_during_batch)
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

        encoding_token_tensor = torch.tensor(encoding_tokens, dtype=torch.int64)
        encoding_token_tensor = torch.nn.functional.pad(encoding_token_tensor, (0, 150 - len(encoding_tokens)))

        # Build tensors for each turn type
        tensors_during = self._build_during_tensor(during_turns, encoding_token_tensor)
        tensors_discard = self._build_discard_tensor(discard_turns, encoding_token_tensor)
        tensors_post = self._build_post_tensor(post_turns, encoding_token_tensor)

        return tensors_during, tensors_discard, tensors_post

    @staticmethod
    def _discard_mask(turn: DiscardTurn) -> torch.Tensor:
        """37-way LEGAL discard mask (MahjongAI.utils.legal): held tiles with the red-five split, minus the
        tiles that break a riichi declaration's tenpai and the kuikae tiles after a chi / pon."""
        return torch.from_numpy(
            legal_discard_mask(turn.stateObj.hand_tensor, turn.is_riichi_declaration, turn.after_call)
        )

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
                - state_obj_tensors: Tuple of tensors (sum_N, N_X1_PLANES, 37, 4), (sum_N, 7, 4), (sum_N, N_X3)
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
            if turn.stateObj.reaches[turn.player] and not turn.is_riichi_declaration:
                # a player in riichi must discard the drawn tile: nothing to learn, skip the sample
                drawn = turn.stateObj.drawn_tile
                if drawn is None or TILE2IDX[turn.discarded_tile][-1] != drawn[-1]:
                    # a parser bug, not a model matter: count it (gates.py asserts 0) instead of killing the run
                    self.skipped["riichi_discard_not_drawn"] += 1
                continue

            filter_tensor = self._discard_mask(turn)  # Shape: (37,)
            if not filter_tensor.any():
                # no legal discard: a parser / rules bug, not a model matter (gates.py asserts 0)
                self.skipped["discard_mask_empty"] += 1
                continue

            encoding_idx = turn.encoding_idx
            assert encoding_idx < MAX_SEQUENCE_LENGTH

            # **Ensure 2D Tensor by Unsqueezing**
            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, dtype=torch.int64),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)
            masked_encoding_token_tensor = self._relative_seats(masked_encoding_token_tensor, turn.player)

            X_embeddings.append(masked_encoding_token_tensor)
            encoding_indices.append(encoding_idx)

            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            # target: slot 34/35/36 for a red five (TILE2IDX[tile] = [kind, red slot]), else the kind
            discarded_tile_idx = TILE2IDX[turn.discarded_tile][-1]

            state_obj_tensors.append((x1, x2, x3))
            filter_tensors.append(filter_tensor)
            y.append(discarded_tile_idx)

        # Concatenate lists into single tensors
        encoding_tokens_batch = torch.cat(X_embeddings, dim=0) if X_embeddings else torch.empty(0, 150, dtype=torch.int64)

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (torch.cat(x1_list, dim=0) if x1_list else torch.empty(0, N_X1_PLANES, 37, 4)),
            (torch.cat(x2_list, dim=0) if x2_list else torch.empty(0, 7, 4)),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, N_X3),
        )

        filter_tensors_2d = [ft.unsqueeze(0) for ft in filter_tensors]  # List of (1, 37)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0) if filter_tensors_2d else torch.empty(0, 37)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == 37
        if action_mask_batch.numel() > 0:
            assert torch.any(action_mask_batch, dim=1).all(), "Each sample must have at least one allowed action"

        y_tensor = torch.tensor(y, dtype=torch.long)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long)  # (N,)
            position_indices = torch.arange(150).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool)

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
                - state_obj_tensors: Tuple of tensors (sum_N, N_X1_PLANES, 37, 4), (sum_N, 7, 4), (sum_N, N_X3)
                - filter_tensors: Tensor of shape (sum_N, DURING_TURN_ACTION_DIM)
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
                    torch.zeros(150 - encoding_idx, dtype=torch.int64),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)
            masked_encoding_token_tensor = self._relative_seats(masked_encoding_token_tensor, turn.player)

            X_embeddings.append(masked_encoding_token_tensor)
            encoding_indices.append(encoding_idx)

            # Convert stateObj to tensors
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)
            state_obj_tensors.append((x1, x2, x3))  # Each x1, x2, x3 is a tensor

            # Create filter_tensor
            filter_tensor = torch.zeros(DURING_TURN_ACTION_DIM, dtype=torch.float32)
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

                elif decision_idx == DECISION_ABORT_IDX:
                    filter_tensor[DURING_ABORT_IDX] = 1.0
                    if decision_type[0].executed:
                        target = DURING_ABORT_IDX

                else:
                    raise ValueError(f"Invalid decision_idx: {decision_idx}")

            y.append(target)
            filter_tensors.append(filter_tensor)  # Shape: (DURING_TURN_ACTION_DIM,)

        encoding_tokens_batch = torch.cat(X_embeddings, dim=0) if X_embeddings else torch.empty(0, 150, dtype=torch.int64)

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (torch.cat(x1_list, dim=0) if x1_list else torch.empty(0, N_X1_PLANES, 37, 4)),
            (torch.cat(x2_list, dim=0) if x2_list else torch.empty(0, 7, 4)),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, N_X3),
        )

        filter_tensors_2d = [ft.unsqueeze(0) for ft in filter_tensors]  # Each filter_tensor is now (1, DURING_TURN_ACTION_DIM)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0) if filter_tensors_2d else torch.empty(0, DURING_TURN_ACTION_DIM)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == DURING_TURN_ACTION_DIM
        if action_mask_batch.numel() > 0:
            assert torch.sum(action_mask_batch, dim=1).min() >= 2, "Each sample must have at least two allowed actions"

        y_tensor = torch.tensor(y, dtype=torch.long)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long)  # (N,)
            position_indices = torch.arange(150).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool)

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask,
        )

    @staticmethod
    def _post_turn_responders(turn: PostTurn) -> List[int]:
        """Responders of a PostTurn that become training samples, in turn order after the discarder.

        A responder that executed an action is always a sample (both winners of a double ron).
        A responder that executed nothing is a "pass" sample only when that pass was observed:
        priority is ron (3) > pon / daiminkan (2) > chi (1), and when another responder executed
        an action of HIGHER priority than ANY action this responder was offered, this responder
        may have chosen that weaker action and lost the race; the XML does not say, so the sample
        is dropped (~2% of responders). A responder offered only a ron while another ron'd, or
        offered only a ron while another pon'd, did pass (double ron is allowed; ron beats pon).
        """
        lowest, done = {}, {}
        for p in range(4):
            offered = [(3, d.executed) for d in turn.decisions[p][DECISION_AGARI_IDX]]
            offered += [(1 if d.naki.is_chi() else 2, d.executed) for d in turn.decisions[p][DECISION_NAKI_IDX]]
            if not offered:
                continue
            lowest[p] = min(prio for prio, _ in offered)
            done[p] = max((prio for prio, executed in offered if executed), default=0)
        top = max(done.values(), default=0)
        order = [(turn.player + k) % 4 for k in (1, 2, 3)]
        return [p for p in order if p in lowest and (done[p] > 0 or top <= lowest[p])]

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
                - state_obj_tensors: Tuple of tensors (sum_N, N_X1_PLANES, 37, 4), (sum_N, 7, 4), (sum_N, N_X3)
                - filter_tensors: Tensor of shape (sum_N, 154)
                - y: Tensor of target labels
                - padding_mask: Boolean tensor of shape (sum_N, 150), True = valid position
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        encoding_indices = []

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
                    torch.zeros(150 - encoding_idx, dtype=torch.int64),
                ]
            ).unsqueeze(
                0
            )  # Shape: (1, 150)

            for player_idx in self._post_turn_responders(turn):
                # the RESPONDER's hand / discards / tile counts, not the discarder's
                x1, x2, x3 = self._convert_state_obj_to_tensors(turn.responder_states[player_idx])
                history = self._relative_seats(masked_encoding_token_tensor, player_idx)

                result = 0
                decision_mask = torch.zeros(154, dtype=torch.float32)

                for decision_idx, decision_type in enumerate(decisions[player_idx]):
                    if len(decision_type) == 0:
                        continue

                    decision_mask[0] = 1.0  # pass

                    if decision_idx == DECISION_AGARI_IDX:
                        decision_mask[1] = 1.0
                        if decision_type[0].executed:
                            result = 1

                    elif decision_idx == DECISION_NAKI_IDX:
                        for meld_decision in decision_type:
                            idx = meld_decision.naki.get_post_turn_filter_idx()
                            decision_mask[idx] = 1.0

                            if meld_decision.executed:
                                result = idx

                    else:
                        raise ValueError(f"Invalid decision_idx: {decision_idx}")

                X_embeddings.append(history)
                state_obj_tensors.append((x1, x2, x3))
                filter_tensors.append(decision_mask)
                y.append(result)
                encoding_indices.append(encoding_idx)

        encoding_tokens_batch = torch.cat(X_embeddings, dim=0) if X_embeddings else torch.empty(0, 150, dtype=torch.int64)

        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        else:
            x1_list, x2_list, x3_list = [], [], []

        state_obj_tensor_batch = (
            (torch.cat(x1_list, dim=0) if x1_list else torch.empty(0, N_X1_PLANES, 37, 4)),
            (torch.cat(x2_list, dim=0) if x2_list else torch.empty(0, 7, 4)),
            torch.cat(x3_list, dim=0) if x3_list else torch.empty(0, N_X3),
        )

        filter_tensors_2d = [ft.unsqueeze(0) for ft in filter_tensors]  # Each filter_tensor is now (1, 154)
        action_mask_batch = (
            torch.cat(filter_tensors_2d, dim=0) if filter_tensors_2d else torch.empty(0, 154)
        )

        assert len(action_mask_batch.shape) == 2
        assert action_mask_batch.shape[1] == 154
        if action_mask_batch.numel() > 0:
            assert torch.sum(action_mask_batch, dim=1).min() >= 2, "Each sample must have at least two allowed actions"

        y_tensor = torch.tensor(y, dtype=torch.long)  # Shape: (sum_N,)

        # Generate padding mask from encoding_idx: True = valid position
        if encoding_indices:
            encoding_idx_tensor = torch.tensor(encoding_indices, dtype=torch.long)  # (N,)
            position_indices = torch.arange(150).unsqueeze(0)  # (1, 150)
            padding_mask = position_indices < encoding_idx_tensor.unsqueeze(1)  # (N, 150) boolean
        else:
            padding_mask = torch.empty(0, 150, dtype=torch.bool)

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
            padding_mask,
        )

    def _convert_state_obj_to_tensors(self, stateObj) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a StateObject into the three StateNet inputs, all from the point of view of
        stateObj.player ("me"). Seat axes are RELATIVE: index 0 = me, 1 = shimocha, 2 = toimen, 3 = kamicha.

        Returns:
            x1: (1, N_X1_PLANES, 37, 4) thermometer planes (plane c, column j = 1 iff count > j)
            x2: (1, 7, 4) per-seat rows: scores/250, parent, parent_rounds_remaining/2 (0 in an extension round),
                double_reaches, reaches, ippatsu, is_menzen
            x3: (1, N_X3=7): remaining_tsumo/70, kyotaku/4, honba/8, rounds_remaining/8 (0 in an extension round),
                then the round wind (bakaze) one-hot E, S, W in slots 4-6
        Every input is O(1): fc_main_1 has no LayerNorm before its ReLU, and raw scores (~250) killed
        14-53% of its units in the unnormalised model.
        """
        me = stateObj.player
        seats = [(me + d) % 4 for d in range(4)]  # relative seat -> absolute seat
        range_tensor = torch.arange(4)

        def plane(counts) -> torch.Tensor:  # (37,) counts -> (1, 37, 4)
            counts = torch.as_tensor(np.asarray(counts), dtype=torch.long)
            return (range_tensor < counts.unsqueeze(1)).float().unsqueeze(0)

        drawn = np.zeros(37, dtype=np.float32)
        if stateObj.drawn_tile is not None:
            drawn[stateObj.drawn_tile] = 1.0
        safe = stateObj.safe_tiles if stateObj.safe_tiles is not None else np.zeros((4, 37), dtype=np.float32)

        planes = (
            [
                plane(stateObj.hand_tensor),
                plane(stateObj.remaining_tiles_pov),
                plane(stateObj.sutehai_tensors[me]),
                plane(drawn),
            ]
            + [plane(stateObj.sutehai_tensors[q]) for q in seats[1:]]
            + [plane(safe[q]) for q in seats[1:]]
        )
        assert len(planes) == N_X1_PLANES
        x1 = torch.stack(planes, dim=1)  # (1, N_X1_PLANES, 37, 4)

        def row(values, scale=1.0) -> torch.Tensor:  # absolute per-seat values -> (1, 4) relative, scaled
            return torch.tensor([float(values[q]) / scale for q in seats], dtype=torch.float32).unsqueeze(0)

        x2 = torch.stack(
            [
                row(stateObj.scores, 250.0),  # 25000 points -> 1.0
                row(stateObj.parent_tensor),
                row(stateObj.parent_rounds_remaining, 2.0),
                row(stateObj.double_reaches),
                row(stateObj.reaches),
                row(stateObj.ippatsu),
                row(stateObj.is_menzen),
            ],
            dim=1,
        )  # (1, 7, 4)

        assert EAST <= stateObj.round_wind <= WEST, f"round_wind {stateObj.round_wind} is not East, South or West"
        x3 = torch.zeros((1, N_X3), dtype=torch.float32)
        x3[0, :4] = torch.tensor(
            [stateObj.remaining_tsumo / 70.0, stateObj.kyotaku / 4.0, stateObj.honba / 8.0, stateObj.rounds_remaining / 8.0]
        )
        x3[0, 4 + stateObj.round_wind - EAST] = 1.0  # bakaze one-hot: 4 East, 5 South, 6 West

        return x1, x2, x3


if __name__ == "__main__":
    model = EncoderConvModel(verbose=True).to(device)
