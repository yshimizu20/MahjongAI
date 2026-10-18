"""T-047 E1: ValSet.evaluate on PrebuiltGame objects, several games per forward call, gives the 468d71c evaluation
(one game per call, tensors built from the parsed game): game ids, sample counts, labels and top-1 counts exactly, NLL
sums within 1e-6 relative per head; on a trained state and on a constructed near tie."""
import pytest
import torch
import torch.nn.functional as F

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel, PrebuiltGame
from MahjongAI.utils.dataloader import DataLoader

N_VAL = 7  # validation games: chunks of 3 leave a remainder chunk of 1
K = 3


@torch.no_grad()
def reference_evaluate(model, ids, games, tensors):
    """ValSet.evaluate of 468d71c, verbatim apart from the arguments: one parsed game (halfturns, tokens) per call."""
    model.eval()
    try:
        out = {h: [] for h in SL.HEADS}
        for gid, (halfturns, tokens), t in zip(ids, games, tensors):
            results = model([halfturns], [tokens], sample=False)
            for head in SL.HEADS:
                y = t[head][1]
                if head not in results:
                    assert len(y) == 0, f"{gid}: no {head} output for {len(y)} samples"
                    out[head].append((gid, 0, 0.0, 0))
                    continue
                r = results[head]
                assert torch.equal(r.y.cpu(), y), f"{gid}: {head} labels differ from the stored validation labels"
                nll = F.cross_entropy(r.logits.double(), r.y, reduction="sum").item()
                out[head].append((gid, len(y), nll, int((r.logits.argmax(dim=1) == r.y).sum())))
        return out
    finally:
        model.train()


@pytest.fixture(scope="module")
def val_and_games(tmp_path_factory):
    """ValSet.load on the fixture validation day (production path) and the same games parsed (the reference input)."""
    from MahjongAI.tests.conftest import make_wide_val_root
    root = make_wide_val_root(tmp_path_factory.mktemp("wide") / "data")
    mp = pytest.MonkeyPatch()
    mp.setattr(SL, "DATA_ROOT", root)
    mp.setattr(SL, "VAL_GAMES", N_VAL)
    val = SL.ValSet.load(lambda *_: None)
    loader = DataLoader(str(root / SL.VAL_YEAR), 1, lambda *_: None, days=SL.VAL_DAYS, seed=SL.VAL_SEED,
                        expected_skips=SL.EXPECTED_SKIPS, logged_skips=SL.LOGGED_SKIPS)
    parsed = [SL._next_game(loader) for _ in range(N_VAL)]
    train = SL.train_cluster(lambda *_: None, 2)
    yield val, parsed, train
    mp.undo()


def _trained_model(train, steps=20):
    torch.manual_seed(0)
    model = EncoderConvModel(verbose=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    done = 0
    while done < steps:
        games, tokens = next(train)
        if not games:
            continue
        loss = sum(r.loss for r in model(games, tokens, sample=True).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        done += 1
    return model


def _assert_rows_equal(got, ref):
    """ids, n and top-1 counts exactly; NLL sums within 1e-6 relative per game and per head."""
    assert got.keys() == ref.keys() == set(SL.HEADS)
    for head in SL.HEADS:
        assert [r[0] for r in got[head]] == [r[0] for r in ref[head]], f"{head}: game ids"
        assert [r[1] for r in got[head]] == [r[1] for r in ref[head]], f"{head}: sample counts"
        assert [r[3] for r in got[head]] == [r[3] for r in ref[head]], f"{head}: top-1 counts"
        for a, b in zip(got[head], ref[head]):
            assert a[2] == pytest.approx(b[2], rel=1e-6, abs=1e-9), f"{head} {a[0]}: NLL sum {a[2]} vs {b[2]}"
        assert sum(r[2] for r in got[head]) == pytest.approx(sum(r[2] for r in ref[head]), rel=1e-6)


def _evaluate_both(model, val, parsed):
    ref = reference_evaluate(model, val.ids, [(h, t) for _, h, t in parsed], val.tensors)
    return val.evaluate(model, games_per_forward=K), ref


def test_valset_holds_prebuilt_games_of_the_reference_order(val_and_games):
    val, parsed, _ = val_and_games
    assert len(val.ids) == N_VAL and val.ids == [f.rsplit("/", 1)[-1] for f, _, _ in parsed]
    assert all(isinstance(g, PrebuiltGame) for g in val.games)


@pytest.mark.parametrize("state", ["init", "trained"])
def test_evaluate_equals_one_game_per_call(val_and_games, state):
    val, parsed, train = val_and_games
    if state == "init":
        torch.manual_seed(0)
        model = EncoderConvModel(verbose=False)
    else:
        model = _trained_model(train)
    got, ref = _evaluate_both(model, val, parsed)
    _assert_rows_equal(got, ref)
    assert sum(r[1] for r in got["discard"]) > 100 and sum(r[3] for r in got["discard"]) > 0
    assert model.training  # evaluate restores train mode


def test_near_tie_is_evaluated_alone_and_equals_one_game_per_call(val_and_games):
    """A trained model whose discard head scores two actions within ~1e-7 of each other wherever both are legal: the
    near tie is detected, the game is re-run alone, and every row equals the reference exactly in n and top-1."""
    val, parsed, train = val_and_games
    model = _trained_model(train)
    labels = torch.cat([t["discard"][1] for t in val.tensors])
    a = int(torch.bincount(labels).argmax())  # the most frequent human discard: often the top logit
    b = (a + 1) % 34
    head = model.heads["discard"]
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        head.weight[b] = head.weight[a] + 1e-7 * torch.randn(head.weight.shape[1], generator=g)
        head.bias[b] = head.bias[a] + 1e-8
    before = val.tie_fallbacks
    got, ref = _evaluate_both(model, val, parsed)
    assert val.tie_fallbacks > before, "no near tie detected: the case is not constructed"
    _assert_rows_equal(got, ref)
    # the comparison sees a single top-1 change: flip one count by one and it fails
    flipped = {h: list(rows) for h, rows in got.items()}
    gid, n, nll, c = flipped["discard"][0]
    flipped["discard"][0] = (gid, n, nll, c + 1 if c < n else c - 1)
    with pytest.raises(AssertionError, match="top-1 counts"):
        _assert_rows_equal(flipped, ref)
