import torch
import os
import sys

sys.path.append("..")

from MahjongAI.model import TransformerModel
from MahjongAI.utils.dataloader import DataLoader


eval_interval = 100
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(max_iters: int, verbose: bool = True):
    model = TransformerModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = DataLoader("data/processed/2021/", model)

    for iter in range(max_iters):
        print(f"iter {iter}")

        for tensors_during, tensors_discard, tensors_post in dataloader:
            # ----------------- DURING -----------------
            (
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
            ) = tensors_during
            encoding_tokens_batch = encoding_tokens_batch.to(device)
            for state_obj_tensor in state_obj_tensor_batch:
                state_obj_tensor = state_obj_tensor.to(device)
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)

            logits, loss = model(
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
                "during",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------- DISCARD -----------------
            (
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
            ) = tensors_discard
            encoding_tokens_batch = encoding_tokens_batch.to(device)
            for state_obj_tensor in state_obj_tensor_batch:
                state_obj_tensor = state_obj_tensor.to(device)
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)

            logits, loss = model(
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
                "discard",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------- POST -----------------
            (
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
            ) = tensors_post
            encoding_tokens_batch = encoding_tokens_batch.to(device)
            for state_obj_tensor in state_obj_tensor_batch:
                state_obj_tensor = state_obj_tensor.to(device)
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)

            logits, loss = model(
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
                "post",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % eval_interval == eval_interval - 1:
                losses = estimate_loss()
                print(
                    f"iter {iter} | train loss {losses['train']} | val loss {losses['val']}"
                )


@torch.no_grad()
def estimate_loss():
    raise NotImplementedError


if __name__ == "__main__":
    train(1)
