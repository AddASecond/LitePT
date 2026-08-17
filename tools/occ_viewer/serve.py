#!/usr/bin/env python3
"""Serve occ_viewer + a Robotruck occ scene directory over HTTP."""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, scene_dir: Path, **kwargs):
        self.scene_dir = scene_dir.resolve()
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        # /scene/... → scene_dir
        from urllib.parse import unquote, urlparse

        parsed = urlparse(path)
        p = unquote(parsed.path)
        if p.startswith("/scene/") or p == "/scene":
            rel = p[len("/scene/") :] if p.startswith("/scene/") else ""
            target = (self.scene_dir / rel).resolve()
            if not str(target).startswith(str(self.scene_dir)):
                return str(VIEWER_DIR / "index.html")
            return str(target)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene",
        required=True,
        help="Path to exported scene root (contains index.json)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    scene = Path(args.scene).resolve()
    if not (scene / "index.json").is_file():
        raise SystemExit(f"missing index.json under {scene}")

    handler = functools.partial(Handler, scene_dir=scene)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/?scene=/scene"
    print(f"scene: {scene}")
    print(f"open:  {url}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
