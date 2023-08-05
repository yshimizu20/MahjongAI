import torch
from MahjongAI.model import TransformerModel


max_iters = 10_000
eval_interval = 100
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = TransformerModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == eval_interval - 1:
            losses = estimate_loss()
            print(
                f"iter {iter} | train loss {losses['train']} | val loss {losses['val']}"
            )

        turns = get_batch()
        xb, yb = process_turns(turns)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def estimate_loss():
    raise NotImplementedError


def process_turns(turns):
    raise NotImplementedError


def get_batch():
    raise NotImplementedError
