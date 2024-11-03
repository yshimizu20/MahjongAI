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
    MAX_SEQUENCE_LENGTH,
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
        while self.current_index < len(self.file_list):
            filename = self.file_list[self.current_index]
            print(f"\nProcessing file: {filename}")
            self.current_index += 1

            try:
                all_halfturns, all_encoding_tokens = process(
                    os.path.join(self.path, filename)
                )
            except InvalidGameException as e:
                print(f"Invalid game detected: {e.msg}")
                continue  # Skip to the next file

            return self.prepare_batches(all_halfturns, all_encoding_tokens)

        raise StopIteration

    def prepare_batches(
        self, all_halfturns, all_encoding_tokens
    ) -> Tuple[
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]
    ]:
        # Prepare batched tensors for all types of turns
        tensors_during_batch = []
        tensors_discard_batch = []
        tensors_post_batch = []

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
        during_encodings, during_state_obj, during_filter, during_y = zip(
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

        # --------------------- Discard Turns ---------------------
        discard_encodings, discard_state_obj, discard_filter, discard_y = zip(
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

        # --------------------- Post Turns ---------------------
        post_encodings, post_state_obj, post_filter, post_y = zip(*tensors_post_batch)
        post_x1, post_x2, post_x3 = zip(*post_state_obj)
        post_encodings = torch.cat(post_encodings, dim=0)
        post_state_obj = (
            torch.cat(post_x1, dim=0),
            torch.cat(post_x2, dim=0),
            torch.cat(post_x3, dim=0),
        )
        post_filter = torch.cat(post_filter, dim=0)
        post_y = torch.cat(post_y, dim=0)

        return (
            (during_encodings, during_state_obj, during_filter, during_y),
            (discard_encodings, discard_state_obj, discard_filter, discard_y),
            (post_encodings, post_state_obj, post_filter, post_y),
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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
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
        """
        # # Return empty tensors if discard_turns is empty
        # if not discard_turns:
        #     return (
        #         torch.empty((0, MAX_SEQUENCE_LENGTH), device=device),
        #         (
        #             torch.empty((0, 3, 37), device=device),
        #             torch.empty((0, 7, 4), device=device),
        #             torch.empty((0, 4), device=device),
        #         ),
        #         torch.empty((0, 37), device=device),
        #         torch.empty((0,), dtype=torch.long, device=device),
        #     )

        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []

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
                else torch.empty(0, 3, 37, device=device)
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

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
        )

    def _build_during_tensor(
        self, during_turns: List[DuringTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
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
        """
        # # Return empty tensors if during_turns is empty
        # if not during_turns:
        #     return (
        #         torch.empty((0, MAX_SEQUENCE_LENGTH), device=device),
        #         (
        #             torch.empty((0, 3, 37), device=device),
        #             torch.empty((0, 7, 4), device=device),
        #             torch.empty((0, 4), device=device),
        #         ),
        #         torch.empty((0, 71), device=device),
        #         torch.empty((0,), dtype=torch.long, device=device),
        #     )

        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []

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

                filter_tensor[0] = 1.0  # pass

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
                else torch.empty(0, 3, 37, device=device)
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

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
            y_tensor,
        )

    def _build_post_tensor(
        self, post_turns: List[PostTurn], encoding_token_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
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
        """
        X_embeddings = []
        state_obj_tensors = []
        filter_tensors = []
        y = []
        players = [0, 1, 2, 3]

        for turn in post_turns:
            decisions = turn.decisions  # type: List[List[List[Decision]]]
            n_decisions = sum(len(item) for sublist in decisions for item in sublist)
            if n_decisions == 0:
                continue

            print(decisions, turn.player)

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
                            print(meld_decision.naki)
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
                else torch.empty(0, 3, 37, device=device)
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

        return (
            encoding_tokens_batch,
            state_obj_tensor_batch,
            action_mask_batch,
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

        # **Ensure 2D Tensor by Stacking Along New Dimension**
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
