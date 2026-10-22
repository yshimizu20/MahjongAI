"""Self-play engine (T-053): four players play a Tenhou 4-player hanchan; the engine writes a Tenhou-format log
(`<mjloggm>`) that the production parser, the gates and the viewer read unchanged.

Legality comes from the parser: at each decision point the engine parses the current round's event prefix in the
unfinished-round mode (`process_xml.process_events(pending=True, admission=False)`) and offers the pending
halfturn's decisions. The engine owns only what the parser does not: the wall (wall.py), meld codes
(meld_codes.py), scoring and payments (scoring.py), the round's event loop (round.py), round progression and the
game file set (game.py), the players (players.py) and the match driver (match.py). engine_lib.py --test holds the
rule fixtures and the bookkeeping checks (gate G2)."""
