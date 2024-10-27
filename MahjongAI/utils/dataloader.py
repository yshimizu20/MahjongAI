import os
import torch
import numpy as np
from typing import List, Tuple

from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.utils.constants import DECISION_AGARI_IDX, DECISION_REACH_IDX, DECISION_NAKI_IDX
from MahjongAI.model import TransformerModel
from MahjongAI.utils.constants import TILE2IDX, PAD_ID
from MahjongAI.decision import Decision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, path: str, model: TransformerModel):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.current_index = 0
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == len(self.file_list):
            self.current_index = 0

        filename = self.file_list[self.current_index]
        print(filename)
        self.current_index += 1

        try:
            all_halfturns, all_encoding_tokens = process(os.path.join(self.path, filename))
        except InvalidGameException as e:
            print(e.msg)
            return self.__next__()
        
        for round_idx, (halfturns, encoding_tokens) in enumerate(zip(all_halfturns, all_encoding_tokens)):
            all_encoding_token_tensor = self._build_encoding_token_tensor(encoding_tokens)

        
        # all_encoding_token_tensors = [self._build_encoding_token_tensor(encoding_tokens) for encoding_tokens in all_encoding_tokens]
        # return all_halfturns, all_encoding_token_tensors

    def build_tensors(self, turns, encoding_token_tensors):
        during_turns = []
        discard_turns = []
        post_turns = []

        for turn in turns:
            if turn.type_ == HalfTurn.DURING:
                during_turns.append(turn)
            elif turn.type_ == HalfTurn.DISCARD:
                discard_turns.append(turn)
            elif turn.type_ == HalfTurn.POST:
                post_turns.append(turn)
            else:
                raise ZeroDivisionError
        
        X_during, y_during, filter_tensors = self._build_during_tensor(during_turns)
        X_discard, state_obj_tensors, y_discard = self._build_discard_tensor(discard_turns, encoding_token_tensors)
        X_post, y_post = self._build_post_tensor(post_turns)

        return (X_during, y_during, filter_tensors), (X_discard, state_obj_tensors, y_discard), (X_post, y_post)

    def _build_during_tensor(self, during_turns):
        L = len(during_turns)
        X = []
        y = torch.zeros(L, 1, dtype=torch.long)
        filter_tensors = torch.zeros(L, 71, dtype=torch.long)

        for i, turn in enumerate(during_turns):
            n_decisions = sum(map(len, turn.decisions))

            if n_decisions == 0:
                continue

            for decision_idx, decision_type in enumerate(turn.decisions):
                if len(decision_type) == 0:
                    continue

                # populate filter tensor and y tensor
                if decision_idx == DECISION_AGARI_IDX:
                    filter_tensors[i, 1] = 1
                    if decision_type[0].executed:
                        y[i] = 1

                elif decision_idx == DECISION_REACH_IDX:
                    filter_tensors[i, 2] = 1
                    if decision_type[0].executed:
                        y[i] = 2

                elif decision_idx == DECISION_NAKI_IDX:
                    for meld in decision_type:
                        idx = meld.get_during_turn_filter_idx()
                        filter_tensors[i, idx] = 1
                        if meld.executed:
                            y[i] = idx

                filter_tensors[i, 0] = 1 # pass is always an option

            # TODO: optimize this
            # create state object tensor
            X.append(self.model.encoder(turn.encoding_tokens, turn.encoding_idx))

        return torch.cat(X, dim=0), y, filter_tensors

    def _build_discard_tensor(self, discard_turns, encoding_token_tensors):
        X_embeddings = []
        X_state_objs = []
        y = []

        max_seq_length = 150

        # Check if encoding_token_tensors exceeds max_seq_length
        if encoding_token_tensors.shape[0] > max_seq_length:
            raise ValueError(f"Length of encoding_token_tensors exceeds max_seq_length of {max_seq_length}")

        # Pad encoding_token_tensors to max_seq_length if it's shorter
        if encoding_token_tensors.shape[0] < max_seq_length:
            padding_length = max_seq_length - encoding_token_tensors.shape[0]  # Corrected
            padding = torch.zeros((padding_length, encoding_token_tensors.shape[1]), device=device)
            encoding_token_tensors = torch.cat([encoding_token_tensors, padding], dim=0)

        for turn in discard_turns:
            encoding_idx = turn.encoding_idx

            # Create a mask with shape (max_seq_length, 1) for broadcasting
            mask = torch.tensor(
                [1] * encoding_idx + [0] * (max_seq_length - encoding_idx),
                device=device,
                dtype=torch.float32  # Ensure correct dtype
            ).unsqueeze(1)  # Shape: (max_seq_length, 1)

            # Apply mask directly to encoding_token_tensors
            embeddings_padded = encoding_token_tensors * mask  # Shape: (max_seq_length, EMBED_SIZE)

            X_embeddings.append(embeddings_padded)  # Shape: (max_seq_length, EMBED_SIZE)

            # Convert stateObj to tensors
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)
            X_state_objs.append((x1, x2, x3))

            # Get the target label (discarded tile index)
            discarded_tile_idx = TILE2IDX[turn.discarded_tile][0]
            y.append(discarded_tile_idx)

        # Stack embeddings into tensor with final shape (batch_size, max_seq_length, EMBED_SIZE)
        X_embeddings_tensor = torch.stack(X_embeddings)

        # Stack state object tensors
        x1_list = [x1.unsqueeze(0) for (x1, _, _) in X_state_objs]
        x2_list = [x2.unsqueeze(0) for (_, x2, _) in X_state_objs]
        x3_list = [x3.unsqueeze(0) for (_, _, x3) in X_state_objs]

        x1_tensor = torch.cat(x1_list, dim=0)  # Shape: (batch_size, 3, 37)
        x2_tensor = torch.cat(x2_list, dim=0)  # Shape: (batch_size, 7, 4)
        x3_tensor = torch.cat(x3_list, dim=0)  # Shape: (batch_size, 4)

        state_obj_tensors = (x1_tensor, x2_tensor, x3_tensor)

        # Convert target labels to tensor
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)  # Shape: (batch_size,)

        return X_embeddings_tensor, state_obj_tensors, y_tensor

    def _build_post_tensor(self, post_turns):
        X = []
        y = []
        players = [0, 1, 2, 3]

        for turn in post_turns:
            decisions = turn.decisions # type: List[List[List[Decision]]]
            n_decisions = sum(map(sum, map(len, decisions)))
            if n_decisions == 0:
                continue

            # Create state object tensor
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            action_player = turn.player
            for player_idx in players[action_player + 1:] + players[:action_player]:
                if len(decisions[player_idx]) == 0:
                    continue
                    
                contains_executed = False
                decision_mask = torch.zeros(154, dtype=torch.float32)
                decision_mask[0] = 1.0  # pass

                for decision_idx, decision_type in enumerate(decisions[player_idx]):
                    if len(decision_type) == 0:
                        continue

                    if decision_idx == DECISION_AGARI_IDX:
                        decision_mask[1] = 1.0
                        if decision_type[0].executed:
                            y.append(1)
                            contains_executed = True

                    elif decision_idx == DECISION_NAKI_IDX:
                        for meld in decision_type:
                            idx = meld.get_post_turn_filter_idx()
                            decision_mask[idx] = 1.0

                            if meld.executed:
                                y.append(meld.get_post_turn_filter_idx())
                                contains_executed = True
                    
                    else:
                        raise ValueError(f"Invalid decision_idx: {decision_idx}")

                    if contains_executed:
                        break

                if not contains_executed:
                    y.append(0)

                X.append((x1, x2, x3, decision_mask))

        y_tensor = torch.tensor(y, dtype=torch.long, device=device)  # Shape: (batch_size,)
        return X, y_tensor

    def _build_encoding_token_tensor(self, encoding_tokens: List[int]):
        encoder = self.model.encoder

        enc_indices = (
            torch.tensor(encoding_tokens, dtype=torch.long).unsqueeze(0).to(device)
        )  # Shape: (1, sequence_length)

        embeddings = encoder(enc_indices)  # Shape: (1, sequence_length, EMBD_SIZE)
        embeddings = embeddings.squeeze(0)  # Shape: (sequence_length, EMBD_SIZE)

        return embeddings

    def _convert_state_obj_to_tensors(self, stateObj) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x1: Contains hand_tensor, remaining_tiles_pov, sutehai_tensor
        hand_tensor = torch.tensor(stateObj.hand_tensor, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 37)
        remaining_tiles_pov = torch.tensor(stateObj.remaining_tiles_pov, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 37)
        sutehai_tensor = torch.tensor(stateObj.sutehai_tensor, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 37)

        x1 = torch.stack([hand_tensor, remaining_tiles_pov, sutehai_tensor], dim=1)  # Shape: (1, 3, 37)

        # x2: Contains scores, parent_tensor, parent_rounds_remaining, double_reaches, reaches, ippatsu, is_menzen
        scores = torch.tensor(stateObj.scores, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        parent_tensor = torch.tensor(stateObj.parent_tensor, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        parent_rounds_remaining = torch.tensor(stateObj.parent_rounds_remaining, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        double_reaches = torch.tensor(stateObj.double_reaches, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        reaches = torch.tensor(stateObj.reaches, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        ippatsu = torch.tensor(stateObj.ippatsu, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)
        is_menzen = torch.tensor(stateObj.is_menzen, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)

        x2 = torch.stack([
            scores,
            parent_tensor,
            parent_rounds_remaining,
            double_reaches,
            reaches,
            ippatsu,
            is_menzen
        ], dim=1)  # Shape: (1, 7, 4)

        # x3: Contains remaining_tiles, kyotaku, honba, rounds_remaining
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

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)

        return x1, x2, x3
