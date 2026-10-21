"""Game viewer (T-052): replay a Tenhou log with the model's decision at every decision point.

export.py   Tenhou log + checkpoint -> trajectory JSON (the display replay zipped with the parser's halfturns)
infer.py    label-free sample construction and single-sample inference on EncoderConvModel
viewer_lib.py  the G2 display-state checks, with a `--test` self-test
server.py   stdlib HTTP server: the static page and the JSON API
static/     the page (vanilla JS) and the vendored tile SVGs
"""
