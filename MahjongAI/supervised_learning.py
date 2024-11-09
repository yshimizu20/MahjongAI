import torch
import os
import sys
import re
import glob
from datetime import datetime
from typing import Tuple, str

sys.path.append("..")

from MahjongAI.agents.transformer import TransformerModel
from MahjongAI.utils.dataloader import DataLoader

eval_interval = 10
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create log file
log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, "w") as f:
    f.write("Training log started\n")


def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[int, str]:
    """
    Finds the latest checkpoint in the given directory.
    Returns a tuple of (latest_iter, checkpoint_path).
    If no checkpoint is found, returns (0, None).
    """
    pattern = os.path.join(checkpoint_dir, "model_checkpoint_iter_*.pt")
    checkpoint_files = glob.glob(pattern)
    max_iter = -1
    latest_checkpoint = None

    for file in checkpoint_files:
        match = re.search(r"model_checkpoint_iter_(\d+)\.pt", file)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
                latest_checkpoint = file

    if latest_checkpoint:
        return max_iter, latest_checkpoint
    else:
        return 0, None


def train(model_name: str, max_iters: int, verbose: bool = True):
    # Ensure 'saved_models/' directory exists
    checkpoint_dir = "saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model
    if model_name == "transformer":
        model = TransformerModel().to(device)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Find latest checkpoint
    latest_iter, latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        # Load the model state
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        log_message(
            f"Loaded model from {latest_checkpoint}, starting from iter {latest_iter + 1}"
        )
        start_iter = latest_iter + 1
    else:
        log_message("No checkpoint found. Starting training from iter 0.")
        start_iter = 0

    train_dataloaders = [
        DataLoader(f"data/processed/{yr}/", model) for yr in range(2012, 2020)
    ]
    val_dataloader = DataLoader("data/processed/2021/", model)  # Validation DataLoader

    for iter in range(start_iter, start_iter + max_iters):
        log_message(f"iter {iter}")

        for train_dataloader in train_dataloaders:
            for game, (tensors_during, tensors_discard, tensors_post) in enumerate(
                train_dataloader
            ):
                # Training code for each turn type (DURING, DISCARD, POST)
                for tensors, turn_type in zip(
                    [tensors_during, tensors_discard, tensors_post],
                    ["during", "discard", "post"],
                ):
                    (
                        encoding_tokens_batch,
                        state_obj_tensor_batch,
                        action_mask_batch,
                        y_tensor,
                    ) = tensors
                    encoding_tokens_batch = encoding_tokens_batch.to(device)
                    state_obj_tensor_batch = tuple(
                        tensor.to(device) for tensor in state_obj_tensor_batch
                    )
                    action_mask_batch = action_mask_batch.to(device)
                    y_tensor = y_tensor.to(device)

                    logits, loss = model(
                        encoding_tokens_batch,
                        state_obj_tensor_batch,
                        action_mask_batch,
                        y_tensor,
                        turn_type,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate and save model every eval_interval iterations
            val_loss = estimate_loss(val_dataloader, model)
            log_message(f"iter {iter} | val loss {val_loss}")

            if (iter + 1) % eval_interval == 0:
                # Save model checkpoint
                model_path = os.path.join(
                    checkpoint_dir, f"model_checkpoint_iter_{iter}.pt"
                )
                torch.save(model.state_dict(), model_path)
                log_message(f"Model saved at {model_path}")

            train_dataloader.reset()

    log_message("Training completed.")


@torch.no_grad()
def estimate_loss(val_dataloader, model):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    num_batches = 0

    for _, (tensors_during, tensors_discard, tensors_post) in enumerate(val_dataloader):
        for tensors, turn_type in zip(
            [tensors_during, tensors_discard, tensors_post],
            ["during", "discard", "post"],
        ):
            (
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
            ) = tensors
            encoding_tokens_batch = encoding_tokens_batch.to(device)
            state_obj_tensor_batch = tuple(
                tensor.to(device) for tensor in state_obj_tensor_batch
            )
            action_mask_batch = action_mask_batch.to(device)
            y_tensor = y_tensor.to(device)

            _, loss = model(
                encoding_tokens_batch,
                state_obj_tensor_batch,
                action_mask_batch,
                y_tensor,
                turn_type,
            )
            total_loss += loss.item()
            num_batches += 1

    val_dataloader.reset()

    model.train()  # Return model to training mode
    return total_loss / num_batches  # Average validation loss


def log_message(message: str):
    """Logs a message to the console and to a log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    # get argv (first check if two arguments are provided and the first one is a string and the second one should be an integer)
    if len(sys.argv) != 2:
        raise ValueError("Please provide the number of iterations as an argument.")
    
    model_name = sys.argv[1]
    assert model_name in ["transformer"]

    try:
        max_iters = int(sys.argv[2])
    except ValueError:
        raise ValueError("Please provide the number of iterations as an integer.")
    

    train(model_name, max_iters)
