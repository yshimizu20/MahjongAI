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

from MahjongAI.agents.transformer_conv import TransformerConvModel
from MahjongAI.utils.dataloader import DataLoader

SAVE_EVERY_STEPS = 2000
learning_rate = 1e-3
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

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
    pattern = os.path.join(checkpoint_dir, f"model_{model_name}_checkpoint_step_*.pt")
    checkpoint_files = glob.glob(pattern)
    max_iter = -1
    latest_checkpoint = None

    # Updated regex to include model_name
    regex = rf"model_{re.escape(model_name)}_checkpoint_step_(\d+)\.pt"

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
    if model_name == "transformer_conv":
        model = TransformerConvModel().to(device)
        batch_size = 2
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
            else 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
        ),
    )

    # Initialize step counter
    current_step = 0

    # Find latest checkpoint
    latest_iter, latest_checkpoint = find_latest_checkpoint(model_name, checkpoint_dir)

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            current_step = checkpoint.get('current_step', 0)
            log_message(f"Loaded checkpoint from {latest_checkpoint} — resuming at step {current_step}")
        else:
            # Old format: bare model state dict, no optimizer/scheduler state
            model.load_state_dict(checkpoint)
            log_message(f"Loaded model (old format) from {latest_checkpoint} — optimizer/scheduler not restored")
        start_iter = latest_iter + 1
    else:
        log_message("No checkpoint found. Starting from scratch.")
        start_iter = 0

    # Initialize dataloaders
    train_dataloaders = [DataLoader(f"data/processed/{yr}/", batch_size, log_message) for yr in range(2021, 2022)]
    val_dataloader = DataLoader("data/processed/2021/", batch_size)  # Validation DataLoader

    for iter in range(start_iter, start_iter + max_iters):
        log_message(f"iter {iter}")

        # Select dataloader in a round-robin fashion
        train_dataloader = train_dataloaders[iter % len(train_dataloaders)]
        log_message(f"iter {iter} | Loading {repr(train_dataloader)}")

        for batch_count, (all_halfturns_list, all_encoding_tokens_list) in enumerate(train_dataloader):
            results = model(all_halfturns_list, all_encoding_tokens_list, sample=True)
            if not results:
                continue

            total_loss = sum(loss for _, loss in results.values())

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                per_head = {h: f"{loss.item():.4f}" for h, (_, loss) in results.items()}
                raise RuntimeError(
                    f"NaN or Inf in total_loss at step {current_step} (per-head: {per_head})"
                )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update the scheduler and step counter
            scheduler.step()
            current_step += 1

            if current_step % SAVE_EVERY_STEPS == 0:
                model_path = os.path.join(checkpoint_dir, f"model_{model_name}_checkpoint_step_{current_step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_step': current_step,
                }, model_path)
                log_message(f"Model saved at {model_path}")

            batch_losses = {head: loss.item() for head, (_, loss) in results.items()}

            # Log learning rate every 100,000 steps
            if current_step % 10000 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log_message(f"Current learning rate: {current_lr}")

            loss_str = " | ".join(
                f"{h} {batch_losses[h]:.4f}" for h in ["during", "discard", "post"] if h in batch_losses
            )
            log_message(f"iter {iter} | batch {batch_count} | {loss_str}")

            if max_batches is not None and batch_count + 1 >= max_batches:
                log_message(f"iter {iter} | Stopping after {max_batches} batches.")
                break

            if (batch_count * batch_size) % 1000 < batch_size:
                val_losses = estimate_loss(val_dataloader, model, batch_size)
                val_str = " | ".join(
                    f"{h} {val_losses[h]:.4f}" for h in ["during", "discard", "post"] if h in val_losses
                )
                log_message(f"iter {iter} | batch {batch_count} | val {val_str}")

        log_message(f"iter {iter} | Processed {repr(train_dataloader)}")

        train_dataloader.reset()

    log_message("Training completed.")


@torch.no_grad()
def estimate_loss(val_dataloader: DataLoader, model, batch_size: int):
    model.eval()  # Set model to evaluation mode
    head_totals = {}
    head_counts = {}

    for batch_count, (all_halfturns, all_encoding_tokens) in enumerate(val_dataloader):
        results = model(all_halfturns, all_encoding_tokens, sample=False)
        for head, (_, loss) in results.items():
            head_totals[head] = head_totals.get(head, 0.0) + loss.item()
            head_counts[head] = head_counts.get(head, 0) + 1

        if batch_count * batch_size >= 20:
            break

    val_dataloader.reset()

    model.train()  # Return model to training mode
    return {h: head_totals[h] / head_counts[h] for h in head_totals}


def log_message(message: str):
    """Logs a message to the console and to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    valid_models = [
        "transformer_conv",
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
