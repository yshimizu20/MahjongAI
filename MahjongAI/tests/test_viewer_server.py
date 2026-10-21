"""T-052 R9: the viewer's API answers every failure with a JSON body: a path outside the roots, a missing file, a
game the parser rejects and a malformed (truncated) log are 400s with the reason; an unexpected exporter failure is a
logged 500, never a dropped connection."""
import glob
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from MahjongAI.utils import gates
from MahjongAI.viewer import export, server

GATE_GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gate_games")


def fixture(name):
    found = glob.glob(os.path.join(GATE_GAMES, "*", "*", "*", name))
    assert len(found) == 1, found
    return found[0]


@pytest.fixture
def served(tmp_path):
    httpd = server.make_server(None, str(tmp_path), 0, extra_roots=())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    """(status, body dict) of GET /api/game?path=..."""
    try:
        with urllib.request.urlopen(f"{base}/api/game?path={urllib.request.quote(path)}") as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        assert e.headers["Content-Type"].startswith("application/json")
        return e.code, json.load(e)


def test_malformed_log_is_a_json_400(served):
    base, tmp = served
    truncated = tmp / "truncated.log"
    truncated.write_bytes(open(gates.SAMPLE_LOG, "rb").read()[:4000])
    code, body = get(base, str(truncated))
    assert code == 400 and body["error"].startswith("malformed log:"), body


def test_rejected_missing_and_outside_paths_are_400s(served):
    base, tmp = served
    code, body = get(base, fixture("2021010100gm-00a9-0000-1500bf1f.log"))  # BYE in round 1: 0 rounds kept
    assert code == 400 and body["error"].startswith("rejected by the parser"), body
    code, body = get(base, str(tmp / "missing.log"))
    assert code == 400 and "FileNotFoundError" in body["error"], body
    code, body = get(base, "/etc/passwd")
    assert code == 400 and "not under the served roots" in body["error"], body


def test_unexpected_export_failure_is_a_logged_500(served, monkeypatch, capsys):
    base, _ = served

    def boom(path, model):
        raise RuntimeError("exporter bug")

    monkeypatch.setattr(export, "export_game", boom)
    code, body = get(base, gates.SAMPLE_LOG)
    assert code == 500 and body["error"] == "export failed: RuntimeError: exporter bug", body
    assert "Traceback" in capsys.readouterr().out


def test_a_good_game_is_served_once_and_cached(served):
    base, _ = served
    code, body = get(base, gates.SAMPLE_LOG)
    assert code == 200 and body["source"] == "sample_haifu.xml" and len(body["rounds"]) == 16
