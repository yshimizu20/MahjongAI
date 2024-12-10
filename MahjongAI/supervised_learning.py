import torch
import os
import sys
import re
import glob
from datetime import datetime
from typing import Tuple

sys.path.append("..")

from MahjongAI.agents.transformer import TransformerModel
from MahjongAI.agents.transformer_conv import TransformerConvModel
from MahjongAI.agents.transformer_conv_large import TransformerConvLargeModel
from MahjongAI.agents.transformer_conv_ordered import TransformerConvOrderedModel
from MahjongAI.utils.dataloader import DataLoader

eval_interval = 1
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create log file
log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, "w") as f:
    f.write("Training log started\n")


def find_latest_checkpoint(model_name: str, checkpoint_dir: str) -> Tuple[int, str]:
    """
    Finds the latest checkpoint in the given directory.
    Returns a tuple of (latest_iter, checkpoint_path).
    If no checkpoint is found, returns (0, None).
    """
    pattern = os.path.join(checkpoint_dir, f"model_{model_name}_checkpoint_iter_*.pt")
    checkpoint_files = glob.glob(pattern)
    max_iter = -1
    latest_checkpoint = None

    # Updated regex to include model_name
    regex = rf"model_{re.escape(model_name)}_checkpoint_iter_(\d+)\.pt"

    for file in checkpoint_files:
        match = re.search(regex, file)
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
        batch_size = 5
    elif model_name == "transformer_conv":
        model = TransformerConvModel().to(device)
        batch_size = 5
    elif model_name == "transformer_conv_large":
        model = TransformerConvLargeModel(2).to(device)
        batch_size = 2
    elif model_name == "transformer_conv_large_3":
        model = TransformerConvLargeModel(3).to(device)
        batch_size = 2
    elif model_name == "transformer_conv_ordered_new":
        model = TransformerConvOrderedModel(1).to(device)
        batch_size = 4
    elif model_name == "transformer_conv_ordered_large_2":
        model = TransformerConvOrderedModel(2).to(device)
        batch_size = 2
    elif model_name == "transformer_conv_ordered_large_3":
        model = TransformerConvOrderedModel(3).to(device)
        batch_size = 2
    elif model_name == "transformer_conv_ordered_large_4":
        model = TransformerConvOrderedModel(4).to(device)
        batch_size = 1
    elif model_name == "transformer_conv_ordered_large_5":
        model = TransformerConvOrderedModel(5).to(device)
        batch_size = 1
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Find latest checkpoint
    latest_iter, latest_checkpoint = find_latest_checkpoint(model_name, checkpoint_dir)

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
        DataLoader(f"data/processed/{yr}/", batch_size, log_message) for yr in range(2012, 2020)
    ]
    val_dataloader = DataLoader("data/processed/2021/", batch_size)  # Validation DataLoader

    for iter in range(start_iter, start_iter + max_iters):
        log_message(f"iter {iter}")

        # for train_dataloader in train_dataloaders:
        train_dataloader = train_dataloaders[iter % len(train_dataloaders)]
        log_message(f"iter {iter} | Loading {repr(train_dataloader)}")

        for batch_count, (all_halfturns_list, all_encoding_tokens_list) in enumerate(
            train_dataloader
        ):
            for logits, loss in model(
                all_halfturns_list, all_encoding_tokens_list, sample=True
            ):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (batch_count * batch_size) % 1000 < batch_size:
                val_loss = estimate_loss(val_dataloader, model, batch_size)
                log_message(
                    f"iter {iter} | batch {batch_count} | val loss {val_loss} | loss {loss.item()}"
                )

        log_message(f"iter {iter} | Processed {repr(train_dataloader)}")

        if (iter + 1) % eval_interval == 0:
            # Save model checkpoint
            model_path = os.path.join(
                checkpoint_dir, f"model_{model_name}_checkpoint_iter_{iter}.pt"
            )
            torch.save(model.state_dict(), model_path)
            log_message(f"Model saved at {model_path}")

        train_dataloader.reset()

    log_message("Training completed.")


@torch.no_grad()
def estimate_loss(val_dataloader: DataLoader, model, batch_size: int):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    num_batches = 0

    for batch_count, (all_halfturns, all_encoding_tokens) in enumerate(val_dataloader):
        total_loss = 0
        num_batches = 0

        for logits, loss in model(all_halfturns, all_encoding_tokens, sample=False):
            total_loss += loss.item()
            num_batches += 1

        if batch_count * batch_size >= 20:
            break

    val_dataloader.reset()

    model.train()  # Return model to training mode
    return total_loss / num_batches  # Average validation loss


def log_message(message: str):
    """Logs a message to the console and to a log file."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


if __name__ == "__main__":
    # get argv (first check if two arguments are provided and the first one is a string and the second one should be an integer)
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: `python3 supervised_learning.py <model_name> <max_iters>`"
        )

    model_name = sys.argv[1]
    assert model_name in [
        "transformer",
        "transformer_conv",
        "transformer_conv_large",
        "transformer_conv_large_3",
        "transformer_conv_ordered_new",
        "transformer_conv_ordered_large_2",
        "transformer_conv_ordered_large_3",
        "transformer_conv_ordered_large_4",
        "transformer_conv_ordered_large_5",
    ], f"Model {model_name} not implemented."

    try:
        max_iters = int(sys.argv[2])
    except ValueError:
        raise ValueError("Please provide the number of iterations as an integer.")

    log_message(f"Training {model_name} for {max_iters} iterations")

    train(model_name, max_iters)
