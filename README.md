# MahjongAI
## Game viewer

Replay a Tenhou log in the browser, all four hands open, with the model's decision at every decision point next to
what the player did (`MahjongAI/viewer/`, T-052).

```bash
conda activate mahjong
python -m MahjongAI.viewer.server --ckpt task/plans/T-051-long-run/run/checkpoints/step_112600.pt --port 8765
# then open http://127.0.0.1:8765/
```

- **Pick a game.** Type a day (`20220101`) and press 一覧 to list that day's logs under `--data-root`
  (default `MahjongAI/data/processed`), or pick a sample / fixture log, or open a JSON written by the exporter:
  `python -m MahjongAI.viewer.export <log> --ckpt <pt> --out game.json`.
- **Table.** The point-of-view seat (視点 0-3) sits at the bottom; ponds in discard order (riichi tile sideways with
  a gold edge; a called tile stays as a grey sideways marker, the meld holds the tile); melds at the right end of
  each hand (a kakan shows all four tiles); the drawn tile apart; the centre shows 東1局, 本場,
  供託, 残り (draws left), the scores, the dealer (親) and a riichi stick per seat in riichi; dora indicators top-left.
- **Decision panel (right).** One block per seat that decides at this point: the model's distribution over the legal
  actions (top 5 + other), its choice in blue, the logged action in red, ✓ / ✗ when the log shows the choice
  (`? (pre-empted)` when a higher-priority call hid it). A discard point also draws a probability bar under each tile
  of that hand; the model's pick has a blue outline, the logged discard a red one (green = both).
- **Controls.** ← → one decision point, ↑ ↓ one discard, space play / pause (speed slider), Home / End; 局 picks the
  round; the last position of a round is its result (winner, yaku, score deltas).

- **Self-play (T-053).** The 対局 row starts a game between four players: pick `model` (the server's `--ckpt`, or
  the checkpoint typed in the ckpt box: it must be in `--ckpt`'s directory or under a `--ckpt-root <dir>`),
  `greedy` (shanten-greedy, never calls) or `uniform` per seat, a seed and a temperature T (0 = argmax), then Start.
  With `follow` on, the table jumps to each new decision point as it is played (the finished rounds plus the round
  in play; a round that just finished shows its result until the next poll); after the end, step it with the
  controls above like a log. At a self-play decision the panel shows that seat's own player: its distribution at
  play time (the model's softmax(logits / T) at T > 0, softmax(logits) at T = 0 where it plays the argmax;
  greedy's one-hot; uniform's flat), its choice (`(sampled)` for a model seat at T > 0 and for a uniform seat at
  every T) and `(pre-empted)` when a higher-priority call won. Every finished game is saved to `--selfplay-dir`
  (default `MahjongAI/viewer/selfplay_games/`, git-ignored) as `<time>_<seed>/` (`game.log` + `decisions.json` +
  `meta.json`) and listed in the `(self-play)` picker, also after a server restart; a game that fails leaves
  `error.json` in its folder instead and is not listed. API: `POST /api/selfplay
  {"seats": ["model", "model", "greedy", "uniform"], "seed": 7, "temperature": 0}`, then
  `GET /api/selfplay/<id>?since=<k>`; `GET /api/selfplay` lists the saved games. Many games without the page:
  `python -m MahjongAI.engine.match --seats model:<pt> greedy model:<pt> greedy --games 20 --out-dir <dir>`
  (four space-separated specs; `--workers N` for N processes).

Checks: `python -m MahjongAI.viewer.viewer_lib --test` (display-state invariants: hand counts, tile ownership, the
gates replay, logged actions, pond order / flags / callers; every corruption raises in the check it names);
`python -m MahjongAI.viewer.viewer_lib --parity --ckpt <pt> <log>...` (the viewer's per-sample inference equals the
training forward: both paths in float64 within 1e-8, the float32 delta reported);
`python -m MahjongAI.viewer.smoke_playwright --ckpt <pt>` (headless browser, needs `playwright` and
`playwright install chromium`). The page validates a trajectory JSON before it shows it and renders every string
as text. Tile images: `MahjongAI/viewer/static/tiles/NOTICE.md` (CC0).
