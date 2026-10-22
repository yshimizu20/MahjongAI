"""The viewer's HTTP server (T-052, plan section B): stdlib http.server, no new dependency.

    python -m MahjongAI.viewer.server --ckpt <pt> [--port 8765] [--data-root <dir>] [--selfplay-dir <dir>]
                                      [--ckpt-root <dir> ...]

GET /                      the page (static/index.html, with viewer.js, tiles.css and the tile SVGs beside it)
GET /api/games?date=YYYYMMDD   the .log files of that day under <data-root>/<year>/<date>/*/
GET /api/fixtures          the sample log and the committed fixture games under tests/data
GET /api/game?path=<log>   the trajectory JSON (exported on demand, cached in RAM for the session); the path must be
                           under the data root, data/sample or tests/data [C8]; a rejected game and a malformed log
                           are a 400 with the reason; an unexpected export failure is a logged 500
POST /api/selfplay {seats, seed, temperature}   queue a self-play game (selfplay.py); 201 {id}; a bad seat /
                           checkpoint / seed / temperature is a 400 with the reason; a `model:<path>` checkpoint must
                           be under the --ckpt directory or a --ckpt-root
GET /api/selfplay/<id>?since=k   the game's status and its decision points from the k-th on (live while it plays)
GET /api/selfplay          the saved self-play games under --selfplay-dir (each served by /api/game as well)
"""
import argparse
import glob
import json
import os
import sys
import threading
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from functools import partial
from typing import Optional
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from MahjongAI.process_xml import InvalidGameException
from MahjongAI.viewer import export, infer, selfplay

VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(VIEWER_DIR)
STATIC_DIR = os.path.join(VIEWER_DIR, "static")
DEFAULT_DATA_ROOT = os.path.join(PACKAGE_DIR, "data", "processed")
SAMPLE_DIR = os.path.join(PACKAGE_DIR, "data", "sample")
FIXTURE_DIR = os.path.join(PACKAGE_DIR, "tests", "data")
DEFAULT_SELFPLAY_DIR = os.path.join(VIEWER_DIR, "selfplay_games")  # git-ignored
MAX_BODY = 1 << 16  # a POST body over this is a 400
MAX_DRAIN = 1 << 20  # a body over MAX_BODY up to this is read and dropped (keep-alive goes on); longer: the 400 closes


class GameStore:
    """Exports games on demand under a lock and keeps the JSON bytes; serves only paths under its roots."""

    def __init__(self, model, data_root: str, extra_roots: tuple = ()):
        self.model = model
        self.roots = [os.path.realpath(r) for r in (data_root, SAMPLE_DIR, FIXTURE_DIR, *extra_roots)]
        self.data_root = data_root
        self._cache = {}
        self._lock = threading.Lock()

    def allowed(self, path: str) -> bool:
        real = os.path.realpath(path)
        return any(real == r or real.startswith(r + os.sep) for r in self.roots)

    def games_of(self, date: str) -> list:
        if not (len(date) == 8 and date.isdigit()):
            raise ValueError(f"date must be YYYYMMDD, got {date!r}")
        return sorted(glob.glob(os.path.join(self.data_root, date[:4], date, "*", "*.log")))

    def fixtures(self) -> list:
        return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.xml"))) + sorted(
            glob.glob(os.path.join(FIXTURE_DIR, "gate_games", "*", "*", "*", "*.log")))

    def game_json(self, path: str) -> bytes:
        if not self.allowed(path):
            raise PermissionError(f"{path} is not under the served roots")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        real = os.path.realpath(path)
        with self._lock:
            if real not in self._cache:
                game = export.export_game(real, self.model)
                self._cache[real] = json.dumps(game, ensure_ascii=False).encode("utf-8")
            return self._cache[real]


class ViewerHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: the page re-requests ~60 tile images per step

    def __init__(self, *args, store: GameStore, runner: Optional[selfplay.SelfPlayRunner] = None, **kwargs):
        self.store = store
        self.runner = runner
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "max-age=3600")  # the static files and tile images
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter than the default: one line per request
        print(f"{self.address_string()} {fmt % args}")

    def _json(self, status: int, body: bytes, close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close:  # the request body was not read: the connection cannot carry another request
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _api(self, route: str, query: dict) -> None:
        try:
            if route == "/api/games":
                games = self.store.games_of(query.get("date", [""])[0])
                body = json.dumps({"games": [{"path": p, "name": os.path.basename(p)} for p in games]})
            elif route == "/api/fixtures":
                body = json.dumps({"games": [{"path": p, "name": os.path.basename(p)} for p in self.store.fixtures()]})
            elif route == "/api/game":
                self._json(200, self.store.game_json(query.get("path", [""])[0]))
                return
            elif route == "/api/selfplay" or route.startswith("/api/selfplay/"):
                self._selfplay_get(route, query)
                return
            else:
                self._json(404, json.dumps({"error": f"no route {route}"}).encode())
                return
            self._json(200, body.encode("utf-8"))
        except InvalidGameException as e:
            self._json(400, json.dumps({"error": f"rejected by the parser: {e.msg}"}).encode("utf-8"))
        except ET.ParseError as e:
            self._json(400, json.dumps({"error": f"malformed log: {e}"}).encode("utf-8"))
        except selfplay.BadRequest as e:
            self._json(400, json.dumps({"error": str(e)}).encode("utf-8"))
        except (PermissionError, FileNotFoundError, ValueError) as e:
            self._json(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            return  # the page abandoned this request (a newer selection): nothing to answer
        except Exception as e:  # an exporter / parser bug: logged with its traceback, reported as a 500
            print(f"{route} {query}: export failed\n{traceback.format_exc()}")
            self._json(500, json.dumps({"error": f"export failed: {type(e).__name__}: {e}"}).encode("utf-8"))

    def _runner(self) -> selfplay.SelfPlayRunner:
        if self.runner is None:
            raise selfplay.BadRequest("self-play is off on this server (no --selfplay-dir)")
        return self.runner

    def _selfplay_get(self, route: str, query: dict) -> None:
        if route == "/api/selfplay":
            body = {"games": self._runner().saved()}
            self._json(200, json.dumps(body, ensure_ascii=False).encode("utf-8"))
            return
        since = query.get("since", ["0"])[0]
        if not since.isdigit():
            raise selfplay.BadRequest(f"since must be a non-negative integer, got {since!r}")
        snap = self._runner().get(route[len("/api/selfplay/"):], int(since))
        if snap is None:
            self._json(404, json.dumps({"error": "no such self-play game"}).encode())
            return
        self._json(200, json.dumps(snap, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path.startswith("/api/"):
            self._api(url.path, urllib.parse.parse_qs(url.query))
        else:
            super().do_GET()

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        unread = False  # a body left in the stream (a bad length, a body over the cap): answer, then close
        try:
            raw = (self.headers.get("Content-Length") or "0").strip()
            if not (raw.isascii() and raw.isdigit()):
                unread = True
                raise selfplay.BadRequest(f"Content-Length must be a non-negative integer, got {raw!r}")
            length = int(raw)
            if length > MAX_BODY:
                unread = not self._drain(length)
                raise selfplay.BadRequest("request body too large")
            data = self.rfile.read(length)
            if url.path != "/api/selfplay":
                self._json(404, json.dumps({"error": f"no route POST {url.path}"}).encode())
                return
            try:
                body = json.loads(data or b"null")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise selfplay.BadRequest(f"the body is not JSON: {e}")
            job_id = self._runner().submit(body)
            self._json(201, json.dumps({"id": job_id}).encode())
        except selfplay.BadRequest as e:
            self._json(400, json.dumps({"error": str(e)}).encode("utf-8"), close=unread)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            print(f"POST {url.path}: failed\n{traceback.format_exc()}")
            self._json(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8"))

    def _drain(self, length: int) -> bool:
        """Reads and drops a body of up to MAX_DRAIN bytes (the connection stays usable); False for a longer one."""
        if length > MAX_DRAIN:
            return False
        while length > 0:
            chunk = self.rfile.read(min(length, 1 << 16))
            if not chunk:
                return False
            length -= len(chunk)
        return True


class ViewerServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return  # the page abandoned the connection (a superseded load): not an error
        super().handle_error(request, client_address)


def make_server(model, data_root: str, port: int, host: str = "127.0.0.1", extra_roots: tuple = (),
                selfplay_dir: Optional[str] = None, ckpt: Optional[str] = None, ckpt_roots: tuple = ()) -> ViewerServer:
    """selfplay_dir: where self-play games are saved (and served from); None turns the self-play routes off. ckpt:
    the checkpoint of a `model` seat. ckpt_roots: where a `model:<path>` checkpoint may lie besides ckpt's
    directory."""
    runner = selfplay.SelfPlayRunner(selfplay_dir, ckpt, ckpt_roots) if selfplay_dir else None
    store = GameStore(model, data_root, extra_roots + ((runner.out_dir,) if runner else ()))
    return ViewerServer((host, port), partial(ViewerHandler, store=store, runner=runner))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="serve the game viewer")
    ap.add_argument("--ckpt", help="checkpoint; without it the page shows legal / actual actions only")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--selfplay-dir", default=DEFAULT_SELFPLAY_DIR, help="where self-play games are saved")
    ap.add_argument("--ckpt-root", action="append", default=[],
                    help="a directory a self-play `model:<path>` seat may load from (besides --ckpt's); repeatable")
    args = ap.parse_args(argv)
    model = infer.load_model(args.ckpt) if args.ckpt else None
    server = make_server(model, args.data_root, args.port, args.host, selfplay_dir=args.selfplay_dir, ckpt=args.ckpt,
                         ckpt_roots=tuple(args.ckpt_root))
    print(f"viewer at http://{args.host}:{args.port}/  (data root {args.data_root}; self-play games in "
          f"{args.selfplay_dir}; Ctrl-C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
