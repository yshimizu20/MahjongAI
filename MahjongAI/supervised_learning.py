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
import pathlib
from datetime import datetime
from typing import Tuple

# Root of the MahjongAI package — so paths resolve correctly regardless of
# which directory the script is invoked from.
_PACKAGE_DIR = pathlib.Path(__file__).parent

sys.path.append("..")

from MahjongAI.agents.encoder_conv import EncoderConvModel
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster

SAVE_EVERY_STEPS = 2000
EVAL_EVERY_STEPS = 100
learning_rate = 1e-3
VAL_DAYS = ("20211230", "20211231")  # held out from training; estimate_loss reads the first VAL_GAMES of them
VAL_GAMES = 20
DATA_SEED = 0  # training file order; stored in the checkpoint so a resume continues where it stopped
MAX_BATCHES_WITHOUT_UPDATE = 50  # consecutive empty / sample-less batches before the run aborts (data failure)
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
    Returns a tuple of (latest_step, checkpoint_path).
    If no checkpoint is found, returns (0, None).
    """
    pattern = os.path.join(checkpoint_dir, f"model_{model_name}_checkpoint_step_*.pt")
    checkpoint_files = glob.glob(pattern)
    max_step = -1
    latest_checkpoint = None

    regex = rf"model_{re.escape(model_name)}_checkpoint_step_(\d+)\.pt"

    for file in checkpoint_files:
        match = re.search(regex, file)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num
                latest_checkpoint = file

    if latest_checkpoint:
        return max_step, latest_checkpoint
    else:
        return 0, None


def train(model_name: str, total_steps: int):
    # Ensure 'saved_models/' directory exists
    checkpoint_dir = "saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model
    if model_name == "encoder_conv":
        model = EncoderConvModel().to(device)
        batch_size = 2
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler parameters
    warmup_steps = min(1000, total_steps // 10)  # a 300-step smoke must leave warmup; a 1,100-step run must not collapse
    lr_total_steps = total_steps  # the cosine reaches ~0 at the end of THIS run's budget

    def lr_lambda(step: int) -> float:  # linear warmup, then cosine decay to ~0 at lr_total_steps
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / float(max(1, lr_total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Initialize step counter
    current_step = 0
    loader_state = None

    # Find latest checkpoint
    latest_step, latest_checkpoint = find_latest_checkpoint(model_name, checkpoint_dir)

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:  # a checkpoint from before the input layout changed (plan T-013)
                raise RuntimeError(
                    f"{latest_checkpoint} does not fit this model; move saved_models/ away to start fresh"
                ) from e
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # The optimizer state restores the LR of the OLD run's schedule; recompute it on THIS run's
            # horizon (total_steps may have changed) so the first resumed update already uses it.
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            current_step = checkpoint.get('current_step', 0)
            loader_state = checkpoint.get('loader_state')
            log_message(f"Loaded checkpoint from {latest_checkpoint} — resuming at step {current_step}")
        else:
            # Old format: bare model state dict, no optimizer/scheduler state
            model.load_state_dict(checkpoint)
            log_message(f"Loaded model (old format) from {latest_checkpoint} — optimizer/scheduler not restored")
    else:
        log_message("No checkpoint found. Starting from scratch.")

    # Initialize dataloaders — paths are anchored to the package directory so
    # the script works correctly from any working directory.
    dataloaders = [
        DataLoader(
            str(_PACKAGE_DIR / "data" / "processed" / str(yr)), batch_size, log_message,
            exclude_days=VAL_DAYS, seed=DATA_SEED,
        )
        for yr in range(2021, 2022)
    ]
    cluster = DataLoaderCluster(dataloaders)
    val_dataloader = DataLoader(  # seeded: the first VAL_GAMES games are a fixed sample ACROSS the VAL_DAYS
        str(_PACKAGE_DIR / "data" / "processed" / "2021"), batch_size, log_message, days=VAL_DAYS, seed=DATA_SEED
    )
    assert val_dataloader.file_list, f"no validation games found for days {VAL_DAYS}"
    for dl in dataloaders:
        assert not set(dl.file_list) & set(val_dataloader.file_list), "train and validation games overlap"
    if loader_state is not None:
        cluster.load_state_dict(loader_state)
        log_message(f"Data loader resumed at {loader_state}")

    def save_checkpoint():
        """Model, optimizer, scheduler, step and loader position: a resume continues on the next batch."""
        model_path = os.path.join(checkpoint_dir, f"model_{model_name}_checkpoint_step_{current_step}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_step': current_step,
            'loader_state': cluster.state_dict(),
        }, model_path)
        log_message(f"Model saved at {model_path}; skipped games: {dict(cluster.skipped())}; "
                    f"skipped samples: {dict(model.tensor_processor.skipped)}")

    # total_steps counts optimizer UPDATES: a batch without samples does not consume the budget, so the
    # cosine horizon (lr_total_steps) and the number of updates agree. A persistent data failure aborts.
    batches_without_update = 0
    while current_step < total_steps:
        step = current_step  # index of the update this batch is about to make (log / eval cadence)
        games, tokens = next(cluster)
        results = model(games, tokens, sample=True) if games else None  # no games: every one failed to parse
        if not results:
            batches_without_update += 1
            if batches_without_update >= MAX_BATCHES_WITHOUT_UPDATE:
                raise RuntimeError(
                    f"{batches_without_update} consecutive batches gave no training sample at step {current_step} "
                    f"(skipped games: {dict(cluster.skipped())}); check the data"
                )
            continue
        batches_without_update = 0

        total_loss = sum(out.loss for out in results.values())

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            per_head = {h: f"{out.loss.item():.4f}" for h, out in results.items()}
            raise RuntimeError(
                f"NaN or Inf in total_loss at step {step} (per-head: {per_head})"
            )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()
        current_step += 1

        if current_step % SAVE_EVERY_STEPS == 0:
            save_checkpoint()

        batch_losses = {head: out.loss.item() for head, out in results.items()}

        if current_step % 10000 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_message(f"Current learning rate: {current_lr}")

        loss_str = " | ".join(
            f"{h} {batch_losses[h]:.4f}" for h in ["during", "discard", "post"] if h in batch_losses
        )
        log_message(f"step {step} | {loss_str}")

        if step % EVAL_EVERY_STEPS == 0:
            val = estimate_loss(val_dataloader, model, batch_size, VAL_GAMES)
            val_str = " | ".join(
                f"{h} loss {val[h]['loss']:.4f} top1 {val[h]['top1']:.3f} (n={val[h]['n']})"
                for h in ["during", "discard", "post"] if h in val
            )
            log_message(f"step {step} | val {val_str}")

    if current_step % SAVE_EVERY_STEPS != 0:
        save_checkpoint()  # the final state of ANY budget (a 300-step smoke never reaches SAVE_EVERY_STEPS)
    log_message("Training completed.")


@torch.no_grad()
def estimate_loss(val_dataloader: DataLoader, model, batch_size: int, max_games: int = 20) -> dict:
    """Per-head sample-weighted loss and top-1 accuracy on the first `max_games` validation games.

    Discard top-1 counts only free choices: forced riichi discards are not samples (see
    TransformerTensorProcessor._build_discard_tensor). During/post top-1 include the "pass" majority.
    """
    model.eval()  # Set model to evaluation mode
    totals = {}

    for batch_count, (all_halfturns, all_encoding_tokens) in enumerate(val_dataloader):
        if not all_halfturns:
            continue
        results = model(all_halfturns, all_encoding_tokens, sample=False)
        for head, out in results.items():
            t = totals.setdefault(head, {"loss": 0.0, "correct": 0, "n": 0})
            n = out.y.numel()
            t["loss"] += out.loss.item() * n
            t["correct"] += (out.logits.argmax(dim=1) == out.y).sum().item()
            t["n"] += n

        if (batch_count + 1) * batch_size >= max_games:
            break

    val_dataloader.reset()

    model.train()  # Return model to training mode
    return {h: {"loss": t["loss"] / t["n"], "top1": t["correct"] / t["n"], "n": t["n"]} for h, t in totals.items()}


def log_message(message: str):
    """Logs a message to the console and to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    valid_models = [
        "encoder_conv",
        # transformer_conv.py is archived: it returns (logits, loss) tuples and reads the old inputs (plan T-013)
    ]

    parser = argparse.ArgumentParser(description="Train a MahjongAI model.")
    parser.add_argument("model_name", choices=valid_models, help="Model to train.")
    parser.add_argument("max_steps", type=int, help="Total number of training steps.")
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
        f"Training {args.model_name} for {args.max_steps} steps"
        + (" [profiling on]" if args.profile else "")
    )

    def _run():
        train(args.model_name, args.max_steps)

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
