import os
import torch
from typing import List, Tuple

from MahjongAI.process_xml import process, InvalidGameException
from MahjongAI.turn import HalfTurn, DuringTurn, DiscardTurn, PostTurn
from MahjongAI.utils.constants import (
    DECISION_AGARI_IDX,
    DECISION_REACH_IDX,
    DECISION_NAKI_IDX,
    TILE2IDX,
)
from MahjongAI.model import TransformerModel
from MahjongAI.decision import Decision, NakiDecision

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
        print(f"Processing file: {filename}")
        self.current_index += 1

        try:
            all_halfturns, all_encoding_tokens = process(
                os.path.join(self.path, filename)
            )
        except InvalidGameException as e:
            print(f"Invalid game detected: {e.msg}")
            return self.__next__()

        for halfturns, encoding_tokens in zip(all_halfturns, all_encoding_tokens):
            tensors_during, tensors_discard, tensors_post = self.build_tensors(
                halfturns, encoding_tokens
            )

            return tensors_during, tensors_discard, tensors_post

        raise StopIteration

    def build_tensors(self, turns: List[HalfTurn], encoding_tokens: List[int]) -> Tuple[
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
    ]:
        """
        Processes the list of turns and builds tensors for each turn type.

        Args:
            turns (List[HalfTurn]): List of HalfTurn instances.
            encoding_token_tensor (List[int]): List of encoding tokens.

        Returns:
            Tuple containing tensors for during_turns, discard_turns, and post_turns.
            Each element is a tuple: (embeddings, state_obj_tensors, filter_tensors, y)
        """
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
                raise ValueError(f"Unknown turn type: {turn.type_}")

        # convert encoding_tokens to tensor and pad it to 150
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

    def _build_during_tensor(
        self, during_turns: List[DuringTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Builds tensors for during turns.

        Args:
            during_turns (List[DuringTurn]): List of DuringTurn instances.
            encoding_token_tensor (torch.Tensor): Tensor of shape (total_seq_length,)

        Returns:
            Tuple containing:
                - embeddings: Tensor of shape (batch_size, max_seq_length)
                - state_obj_tensors: Tuple of tensors (x1, x2, x3)
                - filter_tensors: Tensor of shape (batch_size, 71)
                - y: Tensor of target labels
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []

        max_seq_length = 150

        for turn in during_turns:
            encoding_idx = turn.encoding_idx
            assert encoding_idx < max_seq_length

            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            )

            X_embeddings.append(masked_encoding_token_tensor)

            # Convert stateObj to tensors
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)
            state_obj_tensors.append((x1, x2, x3))  # Each x1, x2, x3 is a tensor

            # Create filter_tensor
            filter_tensor = torch.zeros(71, dtype=torch.float32, device=device)
            filter_tensor[0] = 1.0  # pass

            decision_idx = 0

            for decision_idx, decision_type in enumerate(turn.decisions):
                if not decision_type:
                    continue

                if decision_idx == DECISION_AGARI_IDX:
                    filter_tensor[1] = 1.0
                    if decision_type[0].executed:
                        decision_idx = 1

                elif decision_idx == DECISION_REACH_IDX:
                    filter_tensor[2] = 1.0
                    if decision_type[0].executed:
                        decision_idx = 2

                elif decision_idx == DECISION_NAKI_IDX:
                    for decision in decision_type:
                        assert isinstance(decision, NakiDecision)
                        meld = decision.naki
                        idx = meld.get_during_turn_filter_idx()
                        filter_tensor[idx] = 1.0
                        if decision.executed:
                            decision_idx = idx

                else:
                    raise ValueError(f"Invalid decision_idx: {decision_idx}")

            y.append(decision_idx)
            filter_tensors.append(filter_tensor)  # Shape: (71,)

        # Stack embeddings
        embeddings_tensor = torch.stack(
            X_embeddings
        )  # Shape: (batch_size, max_seq_length)

        x1_list, x2_list, x3_list = (
            zip(*state_obj_tensors) if state_obj_tensors else ([], [], [])
        )
        if x1_list:
            x1_tensor = torch.cat(x1_list, dim=0)  # Shape: (batch_size, 3, 37)
            x2_tensor = torch.cat(x2_list, dim=0)  # Shape: (batch_size, 7, 4)
            x3_tensor = torch.cat(x3_list, dim=0)  # Shape: (batch_size, 4)
            state_obj_tensor_batch = (x1_tensor, x2_tensor, x3_tensor)
        else:
            state_obj_tensor_batch = (
                torch.empty(0, 3, 37, device=device),
                torch.empty(0, 7, 4, device=device),
                torch.empty(0, 4, device=device),
            )

        # Stack filter_tensors
        filter_tensor_batch = torch.stack(filter_tensors)  # Shape: (batch_size, 71)

        # Convert y to tensor
        y_tensor = torch.tensor(
            y, dtype=torch.long, device=device
        )  # Shape: (batch_size,)

        return embeddings_tensor, state_obj_tensor_batch, filter_tensor_batch, y_tensor

    def _build_discard_tensor(
        self, discard_turns: List[DiscardTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Builds tensors for discard turns.

        Args:
            discard_turns (List[DiscardTurn]): List of DiscardTurn instances.
            encoding_token_tensor (torch.Tensor): Tensor of shape (total_seq_length,).

        Returns:
            Tuple containing:
                - embeddings: Tensor of shape (batch_size, max_seq_length)
                - state_obj_tensors: Tuple of tensors (x1, x2, x3)
                - filter_tensors: Tensor of shape (batch_size, 37)
                - y: Tensor of target labels
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []

        max_seq_length = 150

        for turn in discard_turns:
            encoding_idx = turn.encoding_idx
            assert encoding_idx < max_seq_length

            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            )

            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            hand_tensor = torch.tensor(
                turn.stateObj.hand_tensor, dtype=torch.float32, device=device
            )  # Shape: (37,)
            # Ensure maximum value is 1
            filter_tensor = torch.clamp(hand_tensor, max=1.0)  # Shape: (37,)

            # Get the target label (discarded tile index)
            discarded_tile_idx = TILE2IDX[turn.discarded_tile][0]

            X_embeddings.append(masked_encoding_token_tensor)
            state_obj_tensors.append((x1, x2, x3))
            filter_tensors.append(filter_tensor)
            y.append(discarded_tile_idx)

        # Stack embeddings
        embeddings_tensor = torch.stack(X_embeddings)  # Shape: (batch_size, 150)

        # Stack state_obj_tensors
        # Assuming each state_obj_tensor is a tuple (x1, x2, x3)
        if state_obj_tensors:
            x1_list, x2_list, x3_list = zip(*state_obj_tensors)
            x1_tensor = torch.cat(x1_list, dim=0)  # Shape: (batch_size, 3, 37)
            x2_tensor = torch.cat(x2_list, dim=0)  # Shape: (batch_size, 7, 4)
            x3_tensor = torch.cat(x3_list, dim=0)  # Shape: (batch_size, 4)
            state_obj_tensor_batch = (x1_tensor, x2_tensor, x3_tensor)
        else:
            state_obj_tensor_batch = (
                torch.empty(0, 3, 37, device=device),
                torch.empty(0, 7, 4, device=device),
                torch.empty(0, 4, device=device),
            )

        # Stack filter_tensors
        filter_tensor_batch = torch.stack(filter_tensors)  # Shape: (batch_size, 37)

        # Convert y to tensor
        y_tensor = torch.tensor(
            y, dtype=torch.long, device=device
        )  # Shape: (batch_size,)

        return embeddings_tensor, state_obj_tensor_batch, filter_tensor_batch, y_tensor

    def _build_post_tensor(
        self, post_turns: List[PostTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        players = [0, 1, 2, 3]
        max_seq_length = 150

        for turn in post_turns:
            decisions = turn.decisions  # type: List[List[List[Decision]]]
            n_decisions = sum(map(len, decisions))
            if n_decisions == 0:
                continue

            # Generate embeddings
            encoding_idx = turn.encoding_idx
            assert encoding_idx < max_seq_length

            masked_encoding_token_tensor = torch.cat(
                [
                    encoding_token_tensor[:encoding_idx],
                    torch.zeros(150 - encoding_idx, device=device),
                ]
            )

            # Create state object tensor
            x1, x2, x3 = self._convert_state_obj_to_tensors(turn.stateObj)

            action_player = turn.player
            for player_idx in players[action_player + 1 :] + players[:action_player]:
                if len(decisions[player_idx]) == 0:
                    continue

                contains_executed = False
                decision_mask = torch.zeros(154, dtype=torch.float32)

                for decision_idx, decision_type in enumerate(decisions[player_idx]):
                    if len(decision_type) == 0:
                        continue

                    if decision_idx == DECISION_AGARI_IDX:
                        decision_mask[1] = 1.0
                        if decision_type[0].executed:
                            y.append(1)
                            contains_executed = True

                    elif decision_idx == DECISION_REACH_IDX:
                        decision_mask[2] = 1.0
                        if decision_type[0].executed:
                            y.append(2)
                            contains_executed = True

                    elif decision_idx == DECISION_NAKI_IDX:
                        for meld in decision_type:
                            idx = meld.get_post_turn_filter_idx()
                            decision_mask[idx] = 1.0

                            if meld.executed:
                                y.append(meld.get_post_turn_filter_idx())
                                contains_executed = True

                X_embeddings.append(masked_encoding_token_tensor)
                state_obj_tensors.append((x1, x2, x3))
                filter_tensors.append(decision_mask)

                if contains_executed:
                    break

        # Stack embeddings into tensor with shape (batch_size, max_seq_length, EMBED_SIZE)
        X_embeddings_tensor = torch.stack(X_embeddings)

        # Stack state object tensors
        x1_list, x2_list, x3_list = zip(*state_obj_tensors)
        x1_tensor = torch.cat(x1_list, dim=0)  # Shape: (batch_size, 3, 37)
        x2_tensor = torch.cat(x2_list, dim=0)  # Shape: (batch_size, 7, 4)
        x3_tensor = torch.cat(x3_list, dim=0)  # Shape: (batch_size, 4)
        state_obj_tensor_batch = (x1_tensor, x2_tensor, x3_tensor)

        # Stack filter tensors
        filter_tensor_batch = torch.stack(filter_tensors)  # Shape: (batch_size, 154)

        # Convert target labels to tensor
        y_tensor = torch.tensor(
            y, dtype=torch.long, device=device
        )  # Shape: (batch_size,)

        return (
            X_embeddings_tensor,
            state_obj_tensor_batch,
            filter_tensor_batch,
            y_tensor,
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
                - x1: Tensor of shape (1, 3, 37)
                - x2: Tensor of shape (1, 7, 4)
                - x3: Tensor of shape (1, 4)
        """
        # x1: Contains hand_tensor, remaining_tiles_pov, sutehai_tensor
        hand_tensor = torch.tensor(
            stateObj.hand_tensor, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 37)
        remaining_tiles_pov = torch.tensor(
            stateObj.remaining_tiles_pov, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 37)
        sutehai_tensor = torch.tensor(
            stateObj.sutehai_tensor, dtype=torch.float32, device=device
        ).unsqueeze(
            0
        )  # Shape: (1, 37)

        x1 = torch.stack(
            [hand_tensor, remaining_tiles_pov, sutehai_tensor], dim=1
        )  # Shape: (1, 3, 37)

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
