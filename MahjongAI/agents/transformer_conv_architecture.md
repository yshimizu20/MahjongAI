# Transformer-Conv Model Architecture

## Overview

The `TransformerConvModel` is a hybrid deep learning architecture that combines **Transformer** self-attention mechanisms with **Convolutional Neural Networks (CNNs)** to predict player actions in Japanese Mahjong (Riichi Mahjong). This model processes both sequential game history and spatial tile relationships to make informed decisions about discarding tiles, calling riichi, making melds (chi, pon, kan), and declaring wins.

### Purpose

- **Task**: Supervised learning for Mahjong action prediction
- **Input**: Game event history (up to 150 events) + current game state
- **Output**: Probability distribution over legal actions for three decision types:
  - **Discard decisions**: Which tile to discard (37 possible tiles)
  - **During-turn decisions**: Pass, tsumo, riichi, ankan, or kakan (71 actions)
  - **Post-turn decisions**: Pass, ron, chi, pon, or minkan (154 actions)

### High-Level Architecture

```mermaid
graph TB
    subgraph Input [Input Data]
        EncodingTokens[Encoding Tokens<br/>B x 150]
        StateObj[State Object Tensors<br/>x1: B x 3 x 37 x 4<br/>x2: B x 7 x 4<br/>x3: B x 4]
    end
    
    subgraph EncoderModule [Encoder Module]
        TokenDecode[Bit Unpacking]
        Embeddings[Embedding Tables<br/>Type Tile Player Position Naki]
        TransformerBlocks[Transformer Blocks<br/>Self-Attention + FFN]
        TokenDecode --> Embeddings
        Embeddings --> TransformerBlocks
    end
    
    subgraph DecoderModule [Decoder Module]
        StateNet[StateNet CNN<br/>Conv2d Layers + FC]
        DecoderBlocks[Decoder Blocks<br/>Cross-Attention + FFN]
        LayerNorm[Layer Normalization]
        StateNet --> DecoderBlocks
        DecoderBlocks --> LayerNorm
    end
    
    subgraph OutputHeads [Output Heads]
        DiscardHead[Discard Head<br/>37 actions]
        DuringHead[During Head<br/>71 actions]
        PostHead[Post Head<br/>154 actions]
    end
    
    EncodingTokens --> EncoderModule
    EncoderModule --> |B x 150 x 64| DecoderBlocks
    StateObj --> StateNet
    LayerNorm --> OutputHeads
```

### Model Statistics

- **Total Parameters**: ~0.7M parameters
- **Embedding Dimension**: 64
- **Attention Heads**: 8 (head size: 8)
- **Transformer Layers**: 1 (configurable via `N_LAYERS`)
- **Dropout Rate**: 0.2
- **Max Sequence Length**: 150 events

## Input Representations

### 1. Encoding Tokens (Game History)

The encoding tokens are **bit-packed integers** that efficiently represent the sequence of game events (discards, melds, riichi declarations, new dora reveals). Each token is a 16-bit integer with the following structure:

```mermaid
graph LR
    subgraph EncodingToken [16-bit Encoding Token]
        Bits15_13[Bits 13-15<br/>Event State Type]
        Bits12_10[Bits 10-12<br/>Event Type]
        Bits9_8[Bits 8-9<br/>Player Index]
        Bit7[Bit 7<br/>Tsumogiri]
        Bits6_0[Bits 0-6<br/>Tile/Naki Index]
    end
```

**Bit Field Breakdown:**

| Bits | Field | Description | Values |
|------|-------|-------------|--------|
| 13-15 | Event State Type | Type of game state | 0=EMPTY, 1=DISCARD, 2=DURING, 3=POST, 4=NEW_DORA |
| 10-12 | Event Type | Type of event | 0=EMPTY, 1=DISCARD, 2=NAKI, 3=RIICHI, 4=NEW_DORA |
| 8-9 | Player Index | Which player (0-3) | 0-3 |
| 7 | Tsumogiri Flag | Discard from draw | 0=false, 1=true |
| 0-6 | Tile Index | For DISCARD and NEW_DORA events | 0-36 (37 tile types) |
| 0-7 | Naki Index | For DURING and POST naki events | 0-70 for during, 0-153 for post |

**Note on EventStateTypes vs EventTypes:**
- `EventStateTypes` (bits 13-15): Indicates the phase of decision context - whether we're in a discard phase, during-turn decision, post-turn decision, etc.
- `EventTypes` (bits 10-12): Indicates the concrete event kind - what action was recorded (discard, naki call, riichi declaration, new dora reveal).

**Note on Tile vs Naki Encoding:**
- **Tile events** (DISCARD, NEW_DORA) use bits 0-6 (7 bits) to store tile index 0-36
- **Naki events** (DURING/POST naki) use bits 0-7 (8 bits) to store action index:
  - DURING naki: 0-70 (pass, tsumo, riichi, 34 ankan, 34 kakan)
  - POST naki: 0-153 (pass, ron, chi variants, pon variants, minkan)
- The decoder extracts both fields from the same token and uses the appropriate one based on event type

**Example Encoding Functions** (from `process_xml.py`):

```python
# Discard encoding
def discard2idx(who: int, tile_idx: int, is_tsumogiri: bool):
    return (EventStateTypes.DISCARD << 13 | 
            EventTypes.DISCARD << 10 | 
            who << 8 | 
            (1 if is_tsumogiri else 0) << 7 | 
            tile_idx)

# Riichi encoding
def reach2idx(who: int):
    return EventStateTypes.DURING << 13 | EventTypes.RIICHI << 10 | who << 8 | 2

# New dora encoding
def new_dora2idx(tile_idx: int):
    return EventStateTypes.NEW_DORA << 13 | EventTypes.NEW_DORA << 10 | tile_idx
```

**Tile Encoding (0-36):**
- 0-8: Man (characters) 1-9
- 9-17: Pin (circles) 1-9
- 18-26: Sou (bamboo) 1-9
- 27-33: Honors (East, South, West, North, White, Green, Red)
- 34-36: Red fives (one per suit)

### 2. State Object Tensors

The `StateObject` encapsulates the current game state from a player's perspective. It's converted into three tensors:

#### **x1: Hand State Tensor** `(batch_size, 3, 37, 4)`

Three channels encoding tile counts with one-hot-like representation:
1. **Hand tiles**: Current player's hand
2. **Remaining tiles (POV)**: Tiles not yet seen by the player
3. **Sutehai (discard pile)**: Current player's discard pile

Each of the 37 tile types is encoded as a 4-element vector using the transformation:
```python
# Convert count to one-hot-like: [0,0,0,0] for 0, [1,0,0,0] for 1, [1,1,0,0] for 2, etc.
hand_tensor = (torch.arange(4) < count.unsqueeze(1)).float()
```

#### **x2: Game Metadata Tensor** `(batch_size, 7, 4)`

Seven features for each of the 4 players:
1. **Scores**: Current point totals
2. **Parent tensor**: One-hot indicator of dealer
3. **Parent rounds remaining**: Rounds left as dealer
4. **Double reaches**: Double riichi status (1 or 0)
5. **Reaches**: Riichi declaration status
6. **Ippatsu**: One-shot win opportunity flag
7. **Is menzen**: Closed hand status (no open melds)

#### **x3: Global Game State** `(batch_size, 4)`

Four scalar values:
1. **Remaining tsumo**: Tiles left in the wall
2. **Kyotaku**: Riichi stick count on table
3. **Honba**: Continuation counter
4. **Rounds remaining**: Rounds left in the game

## Encoder Architecture

The Encoder processes the sequence of encoding tokens and outputs contextual embeddings for each event in the game history.

### Embedding Layer

The encoder uses multiple embedding tables to convert bit-packed tokens into rich vector representations:

```mermaid
graph TB
    subgraph BitExtraction [Bit Extraction]
        Token[Encoding Token<br/>16-bit integer]
        EventStateType[Event State Type<br/>bits 13-15]
        EventType[Event Type<br/>bits 10-12]
        PlayerIdx[Player Index<br/>bits 8-9]
        TileEnc[Tile Encoding<br/>bits 0-6]
        Tsumogiri[Tsumogiri<br/>bit 7]
        NakiIdx[Naki Index<br/>bits 0-7]
        
        Token --> EventStateType
        Token --> EventType
        Token --> PlayerIdx
        Token --> TileEnc
        Token --> Tsumogiri
        Token --> NakiIdx
    end
    
    subgraph EmbeddingTables [Embedding Tables]
        EventTypeEmb[Event Type Embedding<br/>4 non-EMPTY types x 64 dims]
        TileEmb[Tile Embedding<br/>37 tiles x 64 dims]
        TsumoEmb[Tsumogiri Embedding<br/>1 x 64 dims]
        DuringNakiEmb[During Naki Embedding<br/>71 x 64 dims]
        PostNakiEmb[Post Naki Embedding<br/>154 x 64 dims]
    end
    
    subgraph FixedEncodings [Fixed Positional Encodings]
        PosEnc[Position Encoding<br/>Sinusoidal Non-trainable]
        PlayerEnc[Player Encoding<br/>Circular Sinusoidal]
    end
    
    EventType --> EventTypeEmb
    TileEnc --> TileEmb
    Tsumogiri --> TsumoEmb
    NakiIdx --> DuringNakiEmb
    NakiIdx --> PostNakiEmb
    PlayerIdx --> PlayerEnc
    
    EventTypeEmb --> Sum[Element-wise Sum]
    TileEmb --> Sum
    TsumoEmb --> Sum
    DuringNakiEmb --> Sum
    PostNakiEmb --> Sum
    PosEnc --> Sum
    PlayerEnc --> Sum
    
    Sum --> Output[Embedding Vector<br/>B x 150 x 64]
```

**Embedding Details:**

1. **Event Type Embedding** (4 non-EMPTY types): Embeds the `EventTypes` field (bits 10-12). For tokens where EventTypes == EMPTY, the embedding is not applied (vector = 0). For non-EMPTY EventTypes in {DISCARD, NAKI, RIICHI, NEW_DORA}, we index the 4-slot embedding table with (EventTypes - 1).
2. **Tile Embedding** (37 tiles): Learned representation for each tile type
3. **Tsumogiri Embedding**: Indicates if a discard was from the draw
4. **During Naki Embedding** (71 actions): For ankan/kakan decisions
5. **Post Naki Embedding** (154 actions): For chi/pon/minkan decisions

**Fixed Encodings (Non-trainable):**

**Position Encoding** (Sinusoidal):
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
Captures the temporal position of each event (0-149).

**Player Encoding** (Circular):
```python
# For 4 players arranged in a circle
angle = player_idx * (2π / 4)
PlayerEnc(player, 2i) = sin(angle / 10000^(2i/d))
PlayerEnc(player, 2i+1) = cos(angle / 10000^(2i/d))
```
Maintains rotational symmetry for the 4-player game.

**Conditional Embedding Application:**

Each embedding is only applied when relevant (controlled by masking):
- Type embedding: Applied for all non-EMPTY events
- Tile embedding: Applied for DISCARD and NEW_DORA events
- Player embedding: Applied for DISCARD, DURING, and POST events
- Tsumogiri embedding: Applied for DISCARD events with tsumogiri=1
- During naki embedding: Applied for DURING events with naki_idx >= 3
- Post naki embedding: Applied for POST events with naki_idx >= 2

### Transformer Blocks

Each `EncoderBlock` consists of:

1. **Multi-Head Self-Attention**:
   - 8 attention heads (head_size = 8)
   - Scaled dot-product attention
   - Query, Key, Value projections

2. **Feed-Forward Network**:
   - Linear layer: 64 → 256 (4× expansion)
   - GELU activation
   - Linear layer: 256 → 64
   - Dropout (0.2)

3. **Residual Connections** + **Layer Normalization**:
   - Pre-normalization architecture
   - x = x + SelfAttention(LayerNorm(x))
   - x = x + FFN(LayerNorm(x))

**Encoder Output**: `(B, 150, 64)` - Contextualized embeddings for all game events

## StateNet: Convolutional Component

The `StateNet` is a **key distinguishing feature** of this hybrid architecture. It processes the current game state tensors using convolutional layers to capture spatial relationships between tiles, then combines this with other game metadata.

```mermaid
graph TB
    subgraph Inputs [State Object Inputs]
        x1[x1: Hand State<br/>B x 3 x 37 x 4]
        x2[x2: Game Metadata<br/>B x 7 x 4]
        x3[x3: Global State<br/>B x 4]
    end
    
    subgraph ConvPath [Convolutional Path]
        Conv1[Conv2d Layer 1<br/>3→8 channels<br/>kernel 3x3 padding 1]
        ReLU1[ReLU]
        Conv2[Conv2d Layer 2<br/>8→8 channels<br/>kernel 3x3 padding 1]
        ReLU2[ReLU]
        Flatten1[Flatten<br/>8 x 37 x 4 = 1184]
        
        Conv1 --> ReLU1 --> Conv2 --> ReLU2 --> Flatten1
    end
    
    subgraph RawPath [Raw Path]
        FlattenRaw[Flatten Raw x1<br/>3 x 37 x 4 = 444]
    end
    
    subgraph Combine [Combine All Features]
        Concat1[Concatenate<br/>1184 + 444 = 1628]
        FC1[Fully Connected<br/>1628 → 64]
        ReLUFC1[ReLU]
        FlattenX2[Flatten x2<br/>7 x 4 = 28]
        Concat2[Concatenate<br/>64 + 28 + 4 = 96]
        FC2[Fully Connected<br/>96 → 64]
        ReLUFC2[ReLU]
        FC3[Fully Connected<br/>64 → 64]
        ReLUFC3[ReLU]
    end
    
    x1 --> Conv1
    x1 --> FlattenRaw
    Flatten1 --> Concat1
    FlattenRaw --> Concat1
    Concat1 --> FC1 --> ReLUFC1
    x2 --> FlattenX2
    x3 --> Concat2
    ReLUFC1 --> Concat2
    FlattenX2 --> Concat2
    Concat2 --> FC2 --> ReLUFC2 --> FC3 --> ReLUFC3
    
    ReLUFC3 --> StateEmbedding[State Embedding<br/>B x 64]
```

### Why Convolutional Layers?

**Spatial Tile Relationships**: Mahjong tiles have inherent spatial structure:
- Sequential tiles (e.g., 3-4-5 of the same suit) form valid melds (chi)
- Same tiles (e.g., three 5s) form valid melds (pon)
- Tile proximity matters for calculating shanten (tiles needed to win)

**Convolution Benefits**:
- **Local patterns**: 3×3 kernels detect tile combinations
- **Translation invariance**: Same patterns across different tile types
- **Feature hierarchy**: Layer 1 detects basic patterns, Layer 2 combines them

### StateNet Processing Steps

1. **Convolutional Feature Extraction**:
   ```
   x1: (B, 3, 37, 4) → Conv2d → (B, 8, 37, 4) → Conv2d → (B, 8, 37, 4)
   Flattened: (B, 1184)
   ```

2. **Raw Feature Preservation**:
   ```
   x1 raw: (B, 3, 37, 4) → Flatten → (B, 444)
   ```
   Preserves exact tile counts alongside learned features

3. **Feature Fusion**:
   ```
   Concat(conv_features, raw_features): (B, 1628) → FC → (B, 64)
   ```

4. **Metadata Integration**:
   ```
   Concat(processed_x1, x2_flat, x3): (B, 96) → FC → FC → (B, 64)
   ```

**StateNet Output**: `(B, 64)` - Rich state representation combining CNN features and game metadata

## Decoder Architecture

The Decoder takes the StateNet output and attends to the full game history (encoder output) to produce action predictions.

### Decoder Blocks

Each `DecoderBlock` applies cross-attention in two passes:

```mermaid
graph TB
    StateEmbedding[State Embedding<br/>B x 64]
    EncOut[Encoder Output<br/>B x 150 x 64]
    
    subgraph DecoderBlock [Decoder Block]
        LN1[Layer Norm]
        Unsqueeze1[Unsqueeze<br/>B x 64 → B x 1 x 64]
        CrossAttn1[Cross-Attn history 1<br/>Q from state K V from encoder]
        Squeeze1[Squeeze<br/>B x 1 x 64 → B x 64]
        Residual1[Residual Add]
        
        LN2[Layer Norm]
        Unsqueeze2[Unsqueeze]
        CrossAttn2[Cross-Attn history 2<br/>Q from updated state K V from encoder]
        Squeeze2[Squeeze]
        Residual2[Residual Add]
        
        LN3[Layer Norm]
        FFN[Feed Forward Network<br/>64 → 256 → 64]
        Residual3[Residual Add]
    end
    
    StateEmbedding --> LN1
    LN1 --> Unsqueeze1
    Unsqueeze1 --> CrossAttn1
    EncOut --> CrossAttn1
    CrossAttn1 --> Squeeze1
    Squeeze1 --> Residual1
    StateEmbedding --> Residual1
    
    Residual1 --> LN2
    LN2 --> Unsqueeze2
    Unsqueeze2 --> CrossAttn2
    EncOut --> CrossAttn2
    CrossAttn2 --> Squeeze2
    Squeeze2 --> Residual2
    Residual1 --> Residual2
    
    Residual2 --> LN3
    LN3 --> FFN
    FFN --> Residual3
    Residual2 --> Residual3
    
    Residual3 --> Output[Output<br/>B x 64]
```

**Attention Mechanism Details**:

1. **Cross-attention pass 1**: 
   - State queries history to get a context vector
   - Query from state embedding (current decision point)
   - Keys & Values from encoder output (all past events)

2. **Cross-attention pass 2**:
   - State queries history again after one residual update (a refinement pass)
   - Query from updated state (post-pass-1 residual, then layer normalized)
   - Keys & Values from encoder output (all past events)

3. **Feed-Forward Network**:
   - Non-linear transformation
   - 4× expansion then compression
   - GELU activation + Dropout

### Output Heads

After decoder processing and final layer normalization, the model has **three separate linear heads** for different decision types:

```mermaid
graph TB
    DecoderOutput[Decoder Output<br/>B x 64]
    
    subgraph OutputHeads [Three Output Heads]
        DiscardHead[Discard Head<br/>Linear 64 → 37]
        DuringHead[During Head<br/>Linear 64 → 71]
        PostHead[Post Head<br/>Linear 64 → 154]
    end
    
    subgraph ActionMasking [Action Masking]
        DiscardMask[Discard Mask<br/>B x 37]
        DuringMask[During Mask<br/>B x 71]
        PostMask[Post Mask<br/>B x 154]
    end
    
    subgraph Outputs [Final Outputs]
        DiscardLogits[Discard Logits<br/>37 legal actions]
        DuringLogits[During Logits<br/>71 legal actions]
        PostLogits[Post Logits<br/>154 legal actions]
    end
    
    DecoderOutput --> DiscardHead
    DecoderOutput --> DuringHead
    DecoderOutput --> PostHead
    
    DiscardHead --> |mask invalid| DiscardLogits
    DuringHead --> |mask invalid| DuringLogits
    PostHead --> |mask invalid| PostLogits
    
    DiscardMask -.-> DiscardLogits
    DuringMask -.-> DuringLogits
    PostMask -.-> PostLogits
```

#### Discard Head (37 actions)

Predicts which tile to discard from hand. Actions correspond to the 37 tile types (0-36).

**Action mask**: Based on `hand_tensor` - can only discard tiles in hand (clipped to max 1.0).

#### During Head (71 actions)

Predicts actions during the player's own turn after drawing a tile:
- **0**: Pass (continue to discard phase)
- **1**: Agari (tsumo - declare win)
- **2**: Riichi (declare ready hand)
- **3-36**: Ankan (closed kan) for each tile type
- **37-70**: Kakan (adding to exposed pon) for each tile type

**Action mask**: Constructed from available decisions (agari, riichi, valid ankan/kakan melds).

#### Post Head (154 actions)

Predicts actions after another player's discard:
- **0**: Pass
- **1**: Agari (ron - win on discard)
- **2-82**: Chi (sequential meld) - 63 basic patterns + 18 red five variants
- **83-119**: Pon (triple meld) - 27 basic + 3 red five variants + 7 honor tiles
- **120-153**: Minkan (open kan) - 34 tile types

**Action mask**: Constructed from legal meld decisions based on hand tiles and the discarded tile.

### Action Masking and Loss

**Masking Process**:
```python
# Mask out invalid actions (set logits to -inf)
logits = logits.masked_fill(action_mask == 0, -1e9)

# Cross-entropy loss over legal actions only
loss = F.cross_entropy(logits, target_action, reduction='mean')
```

This ensures:
1. Model only predicts legal moves
2. Loss is computed only over valid action space
3. Invalid actions have near-zero probability after softmax

## Training Process

### Data Pipeline

The `TransformerTensorProcessor` class handles batch preparation:

```mermaid
graph TB
    subgraph InputData [Input: Game Records]
        HalfTurns[List of HalfTurn Objects]
        EncodingTokens[List of Encoding Token Sequences]
    end
    
    subgraph Processing [TensorProcessor]
        Iterate[Iterate Over Games/Turns]
        BuildTensors[Build Tensors per Turn Type]
        Separate[Separate by Turn Type]
    end
    
    subgraph Batching [Batch Creation]
        DuringBatch[During Turn Batch]
        DiscardBatch[Discard Turn Batch]
        PostBatch[Post Turn Batch]
    end
    
    subgraph Training [Training Loop]
        SampleBatch[Optional Sampling<br/>if batch size > 2250]
        Forward[Forward Pass]
        ComputeLoss[Compute Loss]
        Backward[Backward Pass]
    end
    
    HalfTurns --> Iterate
    EncodingTokens --> Iterate
    Iterate --> BuildTensors
    BuildTensors --> Separate
    
    Separate --> DuringBatch
    Separate --> DiscardBatch
    Separate --> PostBatch
    
    DuringBatch --> SampleBatch
    DiscardBatch --> SampleBatch
    PostBatch --> SampleBatch
    
    SampleBatch --> Forward
    Forward --> ComputeLoss
    ComputeLoss --> Backward
```

### HalfTurn Objects

Training data consists of three types of `HalfTurn` objects:

1. **DuringTurn**: Player's own turn after drawing
   - Contains: `player`, `stateObj`, `decisions[][]`, `encoding_idx`
   - Decisions: List of lists for [agari, riichi, naki]
   - One executed decision marked as `executed=True`

2. **DiscardTurn**: Player discarding a tile
   - Contains: `player`, `stateObj`, `discarded_tile`, `encoding_idx`
   - Target: Index of the discarded tile (0-36)

3. **PostTurn**: After another player's discard
   - Contains: `player`, `stateObj`, `decisions[][][]`, `encoding_idx`
   - Decisions: 3D list [player][decision_type][options]
   - Multiple players may have decisions, but only one executes

### Batch Construction

For each turn type, the processor:

1. **Masks encoding sequence**: Zero out token values after `encoding_idx` and generate padding mask
   ```python
   masked_tokens = concat([tokens[:encoding_idx], zeros(150 - encoding_idx)])
   padding_mask = arange(150) < encoding_idx  # Boolean: True = valid position
   ```

2. **Converts state object**: Transform `StateObject` → `(x1, x2, x3)` tensors

3. **Creates action mask**: Binary tensor indicating legal actions

4. **Extracts target**: Index of the action that was actually taken

5. **Generates padding mask**: Boolean tensor derived from `encoding_idx` (not token values)

6. **Stacks into batch**: Concatenate all examples of same type

### Training Loop

```python
for halfturns_batch, encoding_tokens_batch in dataloader:
    # Process and separate by turn type
    tensors_during, tensors_discard, tensors_post = prepare_batches(...)
    
    for tensors, head in [(tensors_during, 'during'), 
                          (tensors_discard, 'discard'), 
                          (tensors_post, 'post')]:
        if len(tensors[0]) == 0:
            continue
        
        # Optional sampling if batch too large
        if sample and len(tensors[0]) > sample_size:
            idx = torch.randperm(len(tensors[0]))[:sample_size]
            tensors = sample_tensors(tensors, idx)
        
        # Unpack
        encoding_tokens, state_obj, action_mask, targets = tensors
        
        # Forward pass
        enc_out = encoder(encoding_tokens)
        logits, loss = decoder(enc_out, state_obj, action_mask, targets, head)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Loss Function

**Cross-Entropy Loss** over the discrete action space:
```python
loss = F.cross_entropy(logits, target_action, reduction='mean')
```

Where:
- `logits`: (B, num_actions) - masked logits for legal actions only
- `target_action`: (B,) - index of the executed action
- Reduction: Mean loss across batch

## Key Design Decisions

### 1. Hybrid Transformer + CNN Architecture

**Motivation**: Combine strengths of both architectures
- **Transformers**: Excel at sequence modeling, capture long-range dependencies in game history
- **CNNs**: Excel at local pattern recognition, capture spatial tile relationships

**Implementation**:
- Encoder (Transformer): Processes game event sequence
- StateNet (CNN): Processes current hand configuration
- Decoder: Fuses both via cross-attention

### 2. Bit-Packed Encoding Tokens

**Motivation**: Memory efficiency and compact representation

**Benefits**:
- Single 16-bit integer encodes multiple fields
- Reduces sequence storage by ~4× vs separate fields
- Fast bitwise operations for decoding
- Easy to extend with additional bit fields

**Tradeoff**: Requires careful bit manipulation and debugging

### 3. Relative Player Encoding

**Motivation**: Mahjong is rotationally symmetric - seat position shouldn't affect strategy

**Implementation**:
- Circular sinusoidal encoding for 4 players
- Maintains consistent relationships (player 0 is always "self")
- Model learns relative positions, not absolute seats

**Benefit**: Better generalization across different seating positions

### 4. Key-Padding Attention Masking

**Motivation**: Model should only attend to events that occurred before the current decision point (`encoding_idx`). Positions beyond `encoding_idx` are padding and must not influence the model's outputs.

**Implementation**:
- Generate a boolean padding mask from `encoding_idx`: `mask[i] = (i < encoding_idx)` where `True` = valid position
- Apply mask in attention: `attn_scores.masked_fill(~mask, -inf)` before softmax
- Zero out encoder outputs at padded positions after transformer blocks to prevent signal leakage through bias terms and LayerNorm
- Pass the same mask to decoder cross-attention to prevent attending to padded encoder positions

**Why attention masking (not just token zeroing)**:
- Simply zeroing token values still allows attention to padded positions
- Linear layers have bias terms, so zero embeddings produce non-zero keys/values
- Softmax normalization distributes probability mass across all positions including padding
- Proper attention masking with `-inf` ensures padded positions contribute exactly zero to attention outputs

**Note**: This is a **bidirectional encoder** with key-padding masking, not a causal (autoregressive) encoder. All valid positions can attend to all other valid positions within the prefix. Causal masking (preventing position `t` from attending to positions `> t` within the valid prefix) is not applied.

### 5. Action Masking

**Motivation**: Enforce legal moves and focus learning on valid actions

**Implementation**:
- Binary mask generated from game rules
- Invalid actions masked with -1e9 before softmax
- Loss computed only over legal action distribution

**Benefits**:
- Guarantees legal outputs
- Faster convergence (no wasted capacity on illegal moves)
- More interpretable predictions

### 6. Three Separate Output Heads

**Motivation**: Different decision types have different action spaces

**Alternative approaches**:
- Single unified output (wasteful - 37+71+154=262 dims always)
- Multi-task with shared layers (less specialized)

**Benefits of separate heads**:
- Specialized weights for each decision type
- No interference between task types
- Clearer training signal

### 7. Pre-Normalization Architecture

**Motivation**: Improved training stability

**Implementation**:
```python
x = x + SelfAttention(LayerNorm(x))  # Not LayerNorm(x + SelfAttention(x))
```

**Benefits**:
- Better gradient flow
- Easier optimization
- Standard in modern transformers (GPT-3, etc.)

### 8. StateNet Architecture Choices

**Why concatenate conv features with raw features?**
- Conv features: Learned spatial patterns
- Raw features: Preserve exact counts
- Both together: Best of learned + explicit representations

**Why two conv layers?**
- Layer 1: Detects basic tile combinations
- Layer 2: Combines layer 1 patterns into higher-level features
- Empirically 2 layers sufficient for Mahjong patterns

### 9. Single Transformer Layer

**Motivation**: Balance between capacity and overfitting

**Observation**:
- Mahjong decisions often depend on recent events
- Single layer sufficient for ~150 event sequence
- More layers risk overfitting on limited training data

**Configurable**: Can increase `N_LAYERS` for larger datasets

## Model Dimensions Summary

### Hyperparameters

```python
# Action dimensions
DISCARD_ACTION_DIM = 37        # 37 tile types
DURING_TURN_ACTION_DIM = 71    # pass, tsumo, riichi, 34 ankan, 34 kakan  
POST_TURN_ACTION_DIM = 154     # pass, ron, 63+18 chi, 27+3+7 pon, 34 minkan

# Sequence
MAX_ACTION_LEN = 150           # Max game events tracked
MAX_SEQUENCE_LENGTH = 150      # Same as above

# Architecture
EMBD_SIZE = 64                 # Embedding dimension
N_HEADS = 8                    # Attention heads (head_size = 8)
N_LAYERS = 1                   # Transformer encoder/decoder layers
DROPOUT_RATIO = 0.2            # Dropout probability
```

### Tensor Shapes Throughout Network

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| **Input** | encoding_tokens | (B, 150) | Bit-packed game history |
| | state_obj.x1 | (B, 3, 37, 4) | Hand state (3 channels) |
| | state_obj.x2 | (B, 7, 4) | Game metadata (7 features × 4 players) |
| | state_obj.x3 | (B, 4) | Global state (4 scalars) |
| | action_mask | (B, num_actions) | Legal action indicator |
| | target | (B,) | Target action index |
| **Encoder** | embeddings | (B, 150, 64) | After embedding sum |
| | enc_out | (B, 150, 64) | After transformer blocks |
| **StateNet** | conv1_out | (B, 8, 37, 4) | After 1st conv layer |
| | conv2_out | (B, 8, 37, 4) | After 2nd conv layer |
| | conv_flat | (B, 1184) | Flattened conv features |
| | raw_flat | (B, 444) | Flattened raw x1 |
| | x1_combined | (B, 1628) | Concatenated conv + raw |
| | x1_processed | (B, 64) | After FC layer |
| | combined | (B, 96) | All features concatenated |
| | state_emb | (B, 64) | Final state embedding |
| **Decoder** | dec_out | (B, 64) | After decoder blocks |
| | dec_norm | (B, 64) | After final layer norm |
| **Output** | logits_discard | (B, 37) | Discard head output |
| | logits_during | (B, 71) | During head output |
| | logits_post | (B, 154) | Post head output |
| **Loss** | loss | scalar | Cross-entropy loss |

### Parameter Count Breakdown

```
Encoder:
  - event_type_embedding:  4 × 64 = 256  # 4 non-EMPTY EventTypes
  - tile_embedding:       37 × 64 = 2,368
  - tsumogiri_embedding:   1 × 64 = 64
  - during_naki_embedding: 71 × 64 = 4,544
  - post_naki_embedding:  154 × 64 = 9,856
  - transformer_blocks:   ~150K (QKV projections, FFN)
  
StateNet:
  - conv2d_x1_1:          3 × 8 × 3 × 3 = 216 (+ 8 bias)
  - conv2d_x1_2:          8 × 8 × 3 × 3 = 576 (+ 8 bias)
  - fc_x1:                1628 × 64 = 104,192 (+ 64 bias)
  - fc_main_1:            96 × 64 = 6,144 (+ 64 bias)
  - fc_main_2:            64 × 64 = 4,096 (+ 64 bias)
  
Decoder:
  - decoder_blocks:       ~150K (QKV projections, FFN)
  - layer_norm:           64 × 2 = 128 (weight + bias)
  - discard_head:         64 × 37 = 2,368 (+ 37 bias)
  - during_head:          64 × 71 = 4,544 (+ 71 bias)
  - post_head:            64 × 154 = 9,856 (+ 154 bias)

Total: ~0.7M parameters
```

## Usage Example

```python
import torch
from MahjongAI.agents.transformer_conv import TransformerConvModel

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerConvModel().to(device)

# Print parameter count
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

# Training mode
model.train()
for logits, loss in model(all_halfturns_list, all_encoding_tokens_list, sample=True):
    # logits: (B, num_actions) - varies by head
    # loss: scalar tensor
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Inference mode (would require additional inference wrapper)
model.eval()
with torch.no_grad():
    # Process single game state
    # Returns action probabilities
    pass
```

## Comparison with Other Variants

This repository contains multiple model variants:

| Model | File | Key Differences |
|-------|------|-----------------|
| **TransformerConv** | `transformer_conv.py` | Base hybrid model (this document) |
| Transformer | `transformer.py` | No CNN - direct state tensor processing |
| TransformerConvLarge | `transformer_conv_large.py` | Larger embedding dimensions and more layers |
| TransformerConvOrdered | `transformer_conv_ordered.py` | Ordered player encoding, different masking |

The **TransformerConv** model balances:
- Architectural complexity (hybrid CNN+Transformer)
- Model size (~0.7M params)
- Training efficiency (single layer)
- Generalization (relative player encoding)

## References and Further Reading

### Mahjong AI
- **Mahjong Game Rules**: Understanding riichi mahjong mechanics is crucial for interpreting the model
- **Shanten Calculation**: Minimum tiles needed to reach tenpai (ready hand)
- **Yaku (Winning Hands)**: Various scoring patterns the model implicitly learns

### Deep Learning Architectures
- **Transformers**: "Attention is All You Need" (Vaswani et al., 2017)
- **Pre-Normalization**: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- **CNNs for Games**: AlphaGo/AlphaZero CNN architectures for board games

### Implementation Details
- **PyTorch**: All models implemented in PyTorch
- **Supervised Learning**: Trained on expert game records (haifu)
- **Data Processing**: `process_xml.py` converts XML game logs to training data

---

**Note**: This documentation describes the architecture as of the current implementation. For the most up-to-date code, refer to `transformer_conv.py` in the repository.
