import torch
import os
import sys

sys.path.append("..")

from MahjongAI.model import TransformerModel
from MahjongAI.utils.dataloader import DataLoader


eval_interval = 100
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(max_iters: int):
    model = TransformerModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = DataLoader("data/processed/2021/", model)

    for iter in range(max_iters):
        halfturns = []
        for _ in range(10):
            turns = next(dataloader)
            halfturns.extend(turns)

    # logits, loss = model(xb, yb)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # if iter % eval_interval == eval_interval - 1:
    #     losses = estimate_loss()
    #     print(
    #         f"iter {iter} | train loss {losses['train']} | val loss {losses['val']}"
    #     )


@torch.no_grad()
def estimate_loss():
    raise NotImplementedError


if __name__ == "__main__":
    train(1)
