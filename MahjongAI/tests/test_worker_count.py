"""T-047 W: the data-worker count changes no batch. Against 2 workers (the 468d71c default), n workers give the same
tensors, file order, skips, log lines and loader state per batch, and one training step on a batch gives the same
selected targets, logits, every gradient and the final RNG state."""
import pytest
import torch

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel, TransformerTensorProcessor, build_game
from MahjongAI.utils.dataloader import DataLoader, DataLoaderCluster
from MahjongAI.utils.game_workers import GameWorkers


def _cluster(roots, workers, log):
    loaders = [DataLoader(str(r), 2, log, seed=0, expected_skips=SL.EXPECTED_SKIPS, logged_skips=SL.LOGGED_SKIPS,
                          workers=workers) for r in roots]
    return DataLoaderCluster(loaders, seed=0)


def _flat(batches):
    return [t for head in batches for t in (head[0], *head[1], head[2], head[3], head[4])]


def _run(roots, n, steps=9):
    """`steps` batches through n workers: per batch (files, cluster state, flat tensors), the log, the skip counts;
    then one sampled training step on a further batch: (targets, logits), gradients, final RNG state."""
    logs, out = [], []
    with GameWorkers(n, build_game, window=SL.DATA_PREFETCH) as w:
        cluster = _cluster(roots, w, logs.append)
        processor = TransformerTensorProcessor()
        for _ in range(steps):
            games, tokens = next(cluster)
            tensors = _flat(processor.prepare_batches(games, tokens)) if games else []
            out.append((list(cluster.last_batch_files), cluster.state_dict(), tensors))
        torch.manual_seed(0)
        model = EncoderConvModel(verbose=False).train()
        torch.manual_seed(5)
        games, tokens = next(cluster)
        results = model(games, tokens, sample=True, sample_size=20)
        sum(r.loss for r in results.values()).backward()
        step = ({h: (r.y, r.logits.detach()) for h, r in results.items()},
                {name: p.grad for name, p in model.named_parameters()}, torch.get_rng_state())
        assert len(w.pids()) == n
    return out, logs, cluster.skipped(), processor.skipped, step


@pytest.mark.parametrize("n", [3, 4, 6])
def test_worker_count_gives_the_same_batches_and_step(small_roots, n):
    base, other = _run(small_roots, 2), _run(small_roots, n)
    batches_a, logs_a, skips_a, tskips_a, (out_a, grads_a, rng_a) = base
    batches_b, logs_b, skips_b, tskips_b, (out_b, grads_b, rng_b) = other
    assert len(batches_a) == len(batches_b) == 9
    for k, (a, b) in enumerate(zip(batches_a, batches_b)):
        assert a[0] == b[0] and a[1] == b[1], f"batch {k}: files or loader state"
        assert len(a[2]) == len(b[2]), k
        for i, (x, y) in enumerate(zip(a[2], b[2])):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), f"batch {k} tensor {i}"
    assert logs_a == logs_b and skips_a == skips_b and tskips_a == tskips_b
    assert sum(skips_a.values()) >= 4  # the fixture's skipped games were met
    assert out_a.keys() == out_b.keys() == set(SL.HEADS)
    for h in out_a:
        assert len(out_a[h][0]) == 20 and torch.equal(out_a[h][0], out_b[h][0]), f"{h} targets"
        assert torch.equal(out_a[h][1], out_b[h][1]), f"{h} logits"
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert grads_a[name] is not None and torch.isfinite(grads_a[name]).all(), name
        assert torch.equal(grads_a[name], grads_b[name]), f"grad {name}"
    assert torch.equal(rng_a, rng_b)
