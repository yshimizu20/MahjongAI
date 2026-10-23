"""Fixed-size position batches with carry-over (T-055, `--batch-mode positions`).

The training cluster serves ONE game per draw (batch_size 1). A game is admitted only once it is complete: every round
built (a worker's PrebuiltGame unpacked, so a deferred build error raises here) and no tensor-level skip. Its rows go
to three FIFO buffers, one per head, each row with its provenance (loader, epoch, file index in that epoch's order,
round, row in the round). A step takes exactly N_d discard rows; post / during take 2 x N_h rows when the buffer
holds that many, else N_h, else none (the head is absent from the step): the step shapes are fixed. Games are drawn
only while the discard buffer is short. The rest carries over to the next step and across epochs. FIFO, no shuffle,
no RNG.

Transaction: `peek` fills the buffers (admissions are permanent) and returns the step's rows without removing them;
`commit`, called after optimizer.step(), removes them. A failure between the two leaves the buffers whole, so a
stop checkpoint keeps every row an update has not trained on. An UnexpectedSkip or a dead worker in a draw leaves
every buffer, counter and provenance as it was (the loaders roll back or move past their own draw).
"""
import hashlib
import os

import torch

from MahjongAI.agents.encoder_conv import PrebuiltGame, TransformerTensorProcessor
from MahjongAI.utils.dataloader import UnexpectedSkip

HEADS = ("during", "discard", "post")
DEFAULT_SIZES = {"during": 96, "discard": 1024, "post": 320}  # rows per step: N_u, N_d, N_p (T-055 [R1])
SIDE_CAP = 8  # a post / during buffer above SIDE_CAP x its size raises (memory guard, never a silent drop)
PROVENANCE = ("loader", "epoch", "file", "round", "row")  # the int64 columns of a row's provenance


def side_take(buffered: int, size: int) -> int:
    """Rows a post / during step takes from a buffer of `buffered` rows: 2 x size, size or 0."""
    return 2 * size if buffered >= 2 * size else size if buffered >= size else 0


def emit_counts(buffered: dict, sizes: dict) -> dict:
    """{head: rows the step takes}: exactly sizes["discard"] discard rows (the buffer holds them), side_take for the
    side heads."""
    assert buffered["discard"] >= sizes["discard"], f"discard buffer {buffered['discard']} < {sizes['discard']}"
    return {h: sizes[h] if h == "discard" else side_take(buffered[h], sizes[h]) for h in HEADS}


def sample_digest(provenance: dict) -> str:
    """12 hex characters of the sha256 of the provenance of a step's rows, per head in HEADS order: the per-step
    log's record of exactly which positions an update trained on."""
    h = hashlib.sha256()
    for head in HEADS:
        p = provenance.get(head)
        h.update(head.encode() + b":")
        if p is not None:
            h.update(p.contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


def _flatten(t) -> tuple:
    """(tokens, (x1, x2, x3), mask, y, padding) -> the 7 tensors in that order."""
    return (t[0], *t[1], t[2], t[3], t[4])


def _unflatten(cols) -> tuple:
    return cols[0], (cols[1], cols[2], cols[3]), cols[4], cols[5], cols[6]


def check_rows(head: str, mask: torch.Tensor, y: torch.Tensor, where: str):
    """The forward's row checks, made once at admission on the CPU: a binary mask, a legal action in every row, a
    legal target."""
    if len(y) == 0:
        return
    assert torch.all((mask == 0) | (mask == 1)), f"{where}: {head} mask is not binary"
    assert torch.any(mask, dim=1).all(), f"{where}: a {head} row has no allowed action"
    assert torch.all(mask[torch.arange(len(y)), y] == 1), f"{where}: a {head} target is masked out"


class RowBuffer:
    """One head's FIFO of rows: a chunk (7 tensors, provenance (n, 5) int64) per admitted game, and the offset of
    the first unconsumed row in the first chunk."""

    def __init__(self):
        self.chunks = []
        self.offset = 0
        self.n = 0

    def __len__(self):
        return self.n

    def append(self, cols: tuple, provenance: torch.Tensor):
        assert all(len(c) == len(provenance) for c in cols)
        if len(provenance):
            self.chunks.append((cols, provenance))
            self.n += len(provenance)

    def first(self, k: int):
        """(7 tensors, provenance) of the first k rows, concatenated (new tensors); the buffer is unchanged."""
        assert 0 < k <= self.n, (k, self.n)
        parts, offset = [], self.offset
        for cols, prov in self.chunks:
            take = min(k, len(prov) - offset)
            parts.append((tuple(c[offset:offset + take] for c in cols), prov[offset:offset + take]))
            k -= take
            offset = 0
            if k == 0:
                break
        return tuple(torch.cat([p[0][i] for p in parts]) for i in range(7)), torch.cat([p[1] for p in parts])

    def drop(self, k: int):
        assert 0 <= k <= self.n, (k, self.n)
        self.n -= k
        while k:
            take = min(k, len(self.chunks[0][1]) - self.offset)
            k -= take
            self.offset += take
            if self.offset == len(self.chunks[0][1]):
                self.chunks.pop(0)
                self.offset = 0

    def state_dict(self) -> dict:
        if not self.n:
            return {"cols": None, "provenance": torch.empty(0, len(PROVENANCE), dtype=torch.int64)}
        cols, prov = self.first(self.n)
        return {"cols": list(cols), "provenance": prov}

    def load_state_dict(self, state: dict):
        self.chunks, self.offset, self.n = [], 0, 0
        if state["cols"] is not None:
            self.append(tuple(state["cols"]), state["provenance"])


class PositionBatcher:
    """Fixed-size per-head steps from a batch_size-1 DataLoaderCluster (the module docstring has the rules).

    Counters (all in the state): games admitted per source epoch; rows in / out per head and source epoch (rows
    buffered = in - out); steps per head and rows taken (0, N or 2 N); the largest buffer per head since the last
    `reset_max_buffered`. read_since_update: the files admitted since the last commit (the per-step `read` hash).
    """

    def __init__(self, cluster, sizes: dict = None, processor: TransformerTensorProcessor = None):
        sizes = dict(DEFAULT_SIZES if sizes is None else sizes)
        assert set(sizes) == set(HEADS) and all(n > 0 for n in sizes.values()), f"position sizes {sizes}"
        assert all(dl.batch_size == 1 for dl in cluster.dataloaders), "positions mode reads one game per draw"
        self.cluster = cluster
        self.sizes = sizes
        self.processor = processor or TransformerTensorProcessor()
        self.buffers = {h: RowBuffer() for h in HEADS}
        self.games_admitted = {}  # {epoch: games}
        self.rows_in = {h: {} for h in HEADS}  # {head: {epoch: rows}}
        self.rows_out = {h: {} for h in HEADS}
        self.taken = {h: {} for h in HEADS}  # {head: {rows taken in a step: steps}}
        self.max_buffered = {h: 0 for h in HEADS}
        self.read_since_update = []  # basenames
        self._peeked = None  # (counts, provenance) of the step between peek and commit; never in the state

    def buffered(self) -> dict:
        return {h: len(self.buffers[h]) for h in HEADS}

    # ---------------------------------------------------------------- admission
    def _build(self, game, tokens) -> list:
        """The game's rounds [(during, discard, post)], every one built; raises a build error before any admission."""
        if isinstance(game, PrebuiltGame):
            return list(game.unpack(self.processor))
        return [self.processor.build_tensors(h, t) for h, t in zip(game, tokens)]

    def _admit(self, game, tokens, file: str, loader: int, epoch: int, index: int):
        rounds = self._build(game, tokens)
        if self.processor.skipped:
            raise UnexpectedSkip(f"tensor-level skip {dict(self.processor.skipped)} in the training game "
                                 f"{os.path.basename(file)}")
        where = os.path.basename(file)
        per_head = {h: [] for h in HEADS}
        for r, built in enumerate(rounds):
            for h, t in zip(HEADS, built):
                n = len(t[3])
                check_rows(h, t[2], t[3], f"{where} round {r}")
                if n:
                    prov = torch.tensor([[loader, epoch, index, r, 0]], dtype=torch.int64).repeat(n, 1)
                    prov[:, 4] = torch.arange(n)
                    per_head[h].append((_flatten(t), prov))
        for h in HEADS:  # every check passed: the game enters all three buffers together
            parts = per_head[h]
            if parts:
                cols = tuple(torch.cat([p[0][i] for p in parts]) for i in range(7))
                self.buffers[h].append(cols, torch.cat([p[1] for p in parts]))
                n = sum(len(p[1]) for p in parts)
                self.rows_in[h][epoch] = self.rows_in[h].get(epoch, 0) + n
        self.games_admitted[epoch] = self.games_admitted.get(epoch, 0) + 1
        self.read_since_update.append(where)

    def _snapshot(self):
        """Everything one draw and its admission change: the cluster's and its loaders' positions, the buffers'
        lengths, the admission counters and the processor's skip counts."""
        c = self.cluster
        return ((c.idx, c.draws, c.epoch, [dl._snapshot() for dl in c.dataloaders]),
                {h: (len(b.chunks), b.n) for h, b in self.buffers.items()},
                dict(self.games_admitted), {h: dict(self.rows_in[h]) for h in HEADS}, len(self.read_since_update),
                self.processor.skipped.copy())

    def _restore(self, snapshot):
        (idx, draws, epoch, loaders), buffers, games, rows_in, n_read, skipped = snapshot
        self.cluster.idx, self.cluster.draws, self.cluster.epoch = idx, draws, epoch
        for dl, s in zip(self.cluster.dataloaders, loaders):
            dl._restore(s)
        for h, (n_chunks, n) in buffers.items():  # admission only appends chunks
            del self.buffers[h].chunks[n_chunks:]
            self.buffers[h].n = n
        self.games_admitted, self.rows_in = games, rows_in
        del self.read_since_update[n_read:]
        self.processor.skipped = skipped

    def _draw(self):
        """One cluster draw; the game it returns (none when it was skipped) is admitted. An interrupt (not an
        Exception: KeyboardInterrupt, SystemExit) inside the draw or the admission undoes both, so a stop never
        loses a drawn game; a rejected game (build error, tensor-level skip, a failed row check) stays drawn: the
        stop names it and the loader is past it, as in games mode."""
        snapshot = self._snapshot()
        try:
            games, tokens = next(self.cluster)
            epoch = self.cluster.epoch
            if epoch > 0 and self.rows_in["discard"].get(epoch - 1, 0) == 0:
                raise RuntimeError(f"a full pass over the training files (epoch {epoch - 1}) admitted no discard "
                                   f"row (skipped games: {dict(self.cluster.skipped())}); check the data")
            if not games:
                return
            assert len(games) == 1
            loader = self.cluster.dataloaders[self.cluster.idx]
            file, index = self.cluster.last_batch_files[0], loader.current_index - 1
            assert loader.file_list[index] == file, f"{file} is not file {index} of the epoch order"
            self._admit(games[0], tokens[0], file, self.cluster.idx, loader.epoch, index)
        except Exception:
            raise
        except BaseException:
            self._restore(snapshot)
            raise

    # ---------------------------------------------------------------- the step transaction
    def peek(self) -> dict:
        """{head: (tokens, (x1, x2, x3), mask, y, padding)} of the next step, absent heads left out. Draws games
        until the discard buffer holds a step; removes nothing (commit does)."""
        while len(self.buffers["discard"]) < self.sizes["discard"]:
            self._draw()
        buffered = self.buffered()
        over = {h: n for h, n in buffered.items() if h != "discard" and n > SIDE_CAP * self.sizes[h]}
        if over:
            raise RuntimeError(f"position buffers {buffered} exceed {SIDE_CAP} x the step size {self.sizes} on "
                               f"{sorted(over)}; games admitted {sum(self.games_admitted.values())}")
        for h in HEADS:
            self.max_buffered[h] = max(self.max_buffered[h], buffered[h])
        counts = emit_counts(buffered, self.sizes)
        rows = {h: self.buffers[h].first(k) for h, k in counts.items() if k}
        self._peeked = (counts, {h: prov for h, (_, prov) in rows.items()})
        return {h: _unflatten(cols) for h, (cols, _) in rows.items()}

    def peeked(self) -> dict:
        """{head: provenance (k, 5) int64} of the rows `peek` returned, absent heads left out."""
        assert self._peeked is not None, "no step peeked"
        return dict(self._peeked[1])

    def commit(self) -> tuple:
        """Removes the peeked step's rows (called after optimizer.step()); returns (the files admitted since the last
        commit, the step's sample_digest)."""
        assert self._peeked is not None, "commit without peek"
        counts, provenance = self._peeked
        for h, k in counts.items():
            self.buffers[h].drop(k)
            self.taken[h][k] = self.taken[h].get(k, 0) + 1
            if k:
                for epoch, n in zip(*torch.unique(provenance[h][:, 1], return_counts=True)):
                    self.rows_out[h][int(epoch)] = self.rows_out[h].get(int(epoch), 0) + int(n)
        read, self.read_since_update, self._peeked = self.read_since_update, [], None
        for h in HEADS:
            n_in, n_out = sum(self.rows_in[h].values()), sum(self.rows_out[h].values())
            assert n_in - n_out == len(self.buffers[h]), f"{h}: rows in {n_in} - out {n_out} != {len(self.buffers[h])}"
        return read, sample_digest(provenance)

    # ---------------------------------------------------------------- reporting and state
    def report(self) -> dict:
        """The counters for a metrics row (JSON keys: strings)."""
        def by_key(d):
            return {str(k): v for k, v in sorted(d.items())}

        return {"sizes": dict(self.sizes), "games_admitted": sum(self.games_admitted.values()),
                "games_by_epoch": by_key(self.games_admitted), "buffered": self.buffered(),
                "max_buffered": dict(self.max_buffered),
                "rows_in": {h: by_key(self.rows_in[h]) for h in HEADS},
                "rows_out": {h: by_key(self.rows_out[h]) for h in HEADS},
                "taken": {h: by_key(self.taken[h]) for h in HEADS}}

    def reset_max_buffered(self):
        """After an eval row is on disk: the next row's maximum starts from the buffers now."""
        self.max_buffered = self.buffered()

    def state_dict(self) -> dict:
        """Buffers (rows and provenance), counters and the files read since the last commit (weights_only-safe)."""
        return {"sizes": dict(self.sizes), "buffers": {h: self.buffers[h].state_dict() for h in HEADS},
                "games_admitted": dict(self.games_admitted),
                "rows_in": {h: dict(self.rows_in[h]) for h in HEADS},
                "rows_out": {h: dict(self.rows_out[h]) for h in HEADS},
                "taken": {h: dict(self.taken[h]) for h in HEADS},
                "max_buffered": dict(self.max_buffered), "read_since_update": list(self.read_since_update)}

    def load_state_dict(self, state: dict):
        assert state["sizes"] == self.sizes, f"checkpoint position sizes {state['sizes']} != {self.sizes}"
        for h in HEADS:
            self.buffers[h].load_state_dict(state["buffers"][h])
        self.games_admitted = dict(state["games_admitted"])
        self.rows_in = {h: dict(state["rows_in"][h]) for h in HEADS}
        self.rows_out = {h: dict(state["rows_out"][h]) for h in HEADS}
        self.taken = {h: dict(state["taken"][h]) for h in HEADS}
        self.max_buffered = dict(state["max_buffered"])
        self.read_since_update = list(state["read_since_update"])
        self._peeked = None
