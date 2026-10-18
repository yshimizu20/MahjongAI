"""T-047 E1: ValSet.evaluate on PrebuiltGame objects, one game per forward call, gives the 468d71c evaluation (one
game per call, tensors built from the parsed game) bit for bit: game ids, sample counts, NLL sums and top-1 counts, at
the init and a trained state. A constructed tie that a two-game call breaks (Codex stage-5 probe) keeps its one-game
top-1 count."""
import pytest
import torch
import torch.nn.functional as F

from MahjongAI import supervised_learning as SL
from MahjongAI.agents.encoder_conv import EncoderConvModel, PrebuiltGame
from MahjongAI.utils.dataloader import DataLoader

N_VAL = 7


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
    """ids, n, NLL sums and top-1 counts bitwise equal, per game and per head."""
    assert got.keys() == ref.keys() == set(SL.HEADS)
    for head in SL.HEADS:
        assert [r[0] for r in got[head]] == [r[0] for r in ref[head]], f"{head}: game ids"
        assert [r[1] for r in got[head]] == [r[1] for r in ref[head]], f"{head}: sample counts"
        assert [r[3] for r in got[head]] == [r[3] for r in ref[head]], f"{head}: top-1 counts"
        assert [r[2] for r in got[head]] == [r[2] for r in ref[head]], f"{head}: NLL sums"


def _evaluate_both(model, val, parsed):
    ref = reference_evaluate(model, val.ids, [(h, t) for _, h, t in parsed], val.tensors)
    return val.evaluate(model), ref


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


def test_tie_broken_by_a_two_game_call_keeps_its_one_game_count():
    """Codex stage-5 probe: synthetic features of two games (11 and 23 discard samples); the discard head scores two
    actions exactly equal for one sample when that game is run alone (a large head coefficient on the feature that a
    two-game call moves). With every label 0 the tie is a hit alone (argmax takes the first) and a miss in a
    two-game call. A forward pre-hook on discard_head makes the two-game difference (feature 0 + 1e-3) so it does not
    depend on float rounding, which differs between platforms (Linux CI gave bit-identical features). evaluate must give the one-game counts."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        torch.manual_seed(0)
        m = EncoderConvModel(verbose=False).eval()
        ns = [11, 23]
        g = torch.Generator().manual_seed(9)
        x1 = torch.randn(sum(ns), 10, 37, 4, generator=g)
        x2 = torch.randn(sum(ns), 7, 4, generator=g)
        x3 = torch.randn(sum(ns), 7, generator=g)
        tokens = torch.zeros(sum(ns), 150, dtype=torch.long)
        y = torch.zeros(sum(ns), dtype=torch.long)
        widths = {"during": 72, "discard": 37, "post": 154}

        def prepare(ids, _):
            ix = torch.cat([torch.arange(sum(ns[:j]), sum(ns[:j + 1])) for j in ids])
            return tuple((tokens[ix], (x1[ix], x2[ix], x3[ix]), torch.ones(len(ix), w), y[ix], tokens[ix] != 0)
                         for w in widths.values())

        m.tensor_processor.prepare_batches = prepare
        eps = 1e-3

        def marker(mod, args):
            """Deterministic stand-in for rounding: feature 0 is 1.0 in a call of one game, 1.0 + eps in a larger one."""
            x = args[0].clone()
            x[:, 0] = 1.0 + (eps if len(x) > max(ns) else 0.0)
            return (x,) + tuple(args[1:])

        hook = m.discard_head.register_forward_pre_hook(marker)
        slope = -256.0
        with torch.no_grad():
            m.discard_head.weight.zero_()
            m.discard_head.bias.fill_(-100.0)
            m.discard_head.weight[0, 0] = slope
            m.discard_head.bias[0] = -slope  # logit 0 is exactly 0 at feature 1.0, negative at 1.0 + eps
            m.discard_head.bias[1] = 0.0
            single = [int((m([k], [None])["discard"].logits.argmax(1) == 0).sum()) for k in range(2)]
            batch = m([0, 1], [None, None])["discard"].logits.argmax(1) == 0
        assert single == [11, 23], f"one-game counts {single}: the singleton tie is not a hit"
        assert [int(batch[:ns[0]].sum()), int(batch[ns[0]:].sum())] != single, "a two-game call keeps the counts"

        val = SL.ValSet.__new__(SL.ValSet)
        val.games, val.ids = [0, 1], ["a", "b"]
        val.tensors = [{h: (torch.ones(n, w), torch.zeros(n, dtype=torch.long), None) for h, w in widths.items()}
                       for n in ns]
        calls = []
        spy = m.register_forward_pre_hook(lambda mod, args, kwargs: calls.append(len(args[0])), with_kwargs=True)
        out = val.evaluate(m)
        spy.remove()
        hook.remove()
        assert calls == [1, 1], f"evaluate made forward calls of {calls} games, not one game per call"
        assert [r[3] for r in out["discard"]] == single, f"top-1 counts {[r[3] for r in out['discard']]} vs {single}"
        assert m.training  # evaluate restores train mode
    finally:
        torch.set_num_threads(threads)
