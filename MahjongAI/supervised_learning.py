import torch
import os
import sys
import re
import glob
import math
import cProfile
import pstats
import io
import argparse
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


def train(model_name: str, max_iters: int, max_batches: int = None, verbose: bool = True):
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
    elif model_name == "transformer_conv_ordered":
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

    # Define learning rate scheduler parameters
    warmup_steps = 1000  # 0.1% of 1,000,000
    total_steps = 1000000  # Total training steps

    # Initialize the scheduler with a linear warmup and cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            float(step) / float(max(1, warmup_steps))
            if step < warmup_steps
            else 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (step - warmup_steps)
                    / float(max(1, total_steps - warmup_steps))
                )
            )
        ),
    )

    # Initialize step counter
    current_step = 0

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

    # Initialize dataloaders
    train_dataloaders = [
        DataLoader(f"data/processed/{yr}/", batch_size, log_message)
        for yr in range(2021, 2022)
    ]
    val_dataloader = DataLoader(
        "data/processed/2021/", batch_size
    )  # Validation DataLoader

    for iter in range(start_iter, start_iter + max_iters):
        log_message(f"iter {iter}")

        # Select dataloader in a round-robin fashion
        train_dataloader = train_dataloaders[iter % len(train_dataloaders)]
        log_message(f"iter {iter} | Loading {repr(train_dataloader)}")

        for batch_count, (all_halfturns_list, all_encoding_tokens_list) in enumerate(
            train_dataloader
        ):
            batch_losses = {}
            for logits, loss, head in model(
                all_halfturns_list, all_encoding_tokens_list, sample=True
            ):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the scheduler and step counter
                scheduler.step()
                current_step += 1
                batch_losses[head] = loss.item()

                # Log learning rate every 100,000 steps
                if current_step % 10000 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_message(f"Current learning rate: {current_lr}")

            loss_str = " | ".join(f"{h} {batch_losses[h]:.4f}" for h in ["during", "discard", "post"] if h in batch_losses)
            log_message(f"iter {iter} | batch {batch_count} | {loss_str}")

            if max_batches is not None and batch_count + 1 >= max_batches:
                log_message(f"iter {iter} | Stopping after {max_batches} batches.")
                break

            if (batch_count * batch_size) % 1000 < batch_size:
                val_loss = estimate_loss(val_dataloader, model, batch_size)
                log_message(
                    f"iter {iter} | batch {batch_count} | val loss {val_loss:.4f} | {loss_str}"
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
        for logits, loss, head in model(all_halfturns, all_encoding_tokens, sample=False):
            total_loss += loss.item()
            num_batches += 1

        if batch_count * batch_size >= 20:
            break

    val_dataloader.reset()

    model.train()  # Return model to training mode
    return total_loss / num_batches  # Average validation loss


def log_message(message: str):
    """Logs a message to the console and to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    valid_models = [
        "transformer",
        "transformer_conv",
        "transformer_conv_large",
        "transformer_conv_ordered",
    ]

    parser = argparse.ArgumentParser(description="Train a MahjongAI model.")
    parser.add_argument("model_name", choices=valid_models, help="Model to train.")
    parser.add_argument("max_iters", type=int, help="Number of dataset iterations.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help="Stop each iter after N batches (useful for profiling/debugging).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and print the top-50 slowest calls on exit.",
    )
    parser.add_argument(
        "--profile-out",
        default=None,
        metavar="FILE",
        help="Save raw cProfile stats to FILE (e.g. profile.out) for later analysis with snakeviz.",
    )
    args = parser.parse_args()

    log_message(
        f"Training {args.model_name} for {args.max_iters} iters"
        + (f", max {args.max_batches} batches/iter" if args.max_batches else "")
        + (" [profiling on]" if args.profile else "")
    )

    def _run():
        train(args.model_name, args.max_iters, max_batches=args.max_batches)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        _run()
        pr.disable()

        # Print top-50 functions by cumulative time
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(50)
        profile_report = buf.getvalue()
        print(profile_report)
        with open(log_file, "a") as f:
            f.write("\n=== cProfile report ===\n")
            f.write(profile_report)

        if args.profile_out:
            pr.dump_stats(args.profile_out)
            log_message(f"cProfile stats saved to {args.profile_out}")
    else:
        _run()
