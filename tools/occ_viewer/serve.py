#!/usr/bin/env python3
"""Serve occ_viewer + scene dir, with background scene-video export API.

Endpoints:
  GET  /                     viewer
  GET  /scene/...            scene assets
  POST /api/video/export     start export {mode,fps,tile_w,tile_h,max_frames}
  GET  /api/video/status     job status JSON
  GET  /api/video/list       list videos under scene/videos/
  GET  /videos/<file>        download exported mp4 / meta
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = Path(__file__).resolve().parent
EXPORT_SCRIPT = ROOT / "tools" / "export_robotruck_scene_video.py"


class VideoJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.job_id: str | None = None
        self.state = "idle"  # idle|running|done|error
        self.message = ""
        self.progress = 0.0
        self.frame = 0
        self.n = 0
        self.path: str | None = None
        self.relpath: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.proc: subprocess.Popen | None = None
        self.params: dict = {}

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "state": self.state,
                "message": self.message,
                "progress": self.progress,
                "frame": self.frame,
                "n": self.n,
                "path": self.path,
                "relpath": self.relpath,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "params": self.params,
            }


JOB = VideoJob()


def _python_bin() -> str:
    # Prefer venv used by the project if present
    for cand in (
        ROOT / ".venv_smoke" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ):
        if Path(cand).is_file():
            return str(cand)
    return sys.executable


def _run_export(scene_dir: Path, params: dict) -> None:
    out_dir = scene_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "job_status.json"
    log_path = out_dir / "job_export.log"

    mode = params.get("mode", "occ")
    fps = float(params.get("fps", 5))
    tile_w = int(params.get("tile_w", 960))
    tile_h = int(params.get("tile_h", 540))
    max_frames = int(params.get("max_frames", 0))

    cmd = [
        _python_bin(),
        str(EXPORT_SCRIPT),
        "--scene",
        str(scene_dir),
        "--mode",
        str(mode),
        "--fps",
        str(fps),
        "--tile-w",
        str(tile_w),
        "--tile-h",
        str(tile_h),
        "--max-frames",
        str(max_frames),
    ]

    with JOB.lock:
        JOB.state = "running"
        JOB.message = "starting exporter"
        JOB.progress = 0.0
        JOB.frame = 0
        JOB.n = 0
        JOB.path = None
        JOB.relpath = None
        JOB.started_at = time.time()
        JOB.finished_at = None
        JOB.params = dict(params)
        status_path.write_text(json.dumps(JOB.snapshot(), indent=2))

    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with JOB.lock:
            JOB.proc = proc
            JOB.message = "export running"

        # Poll log lightly for progress hints
        while proc.poll() is None:
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                # lines like: [3/10] done
                import re

                ms = list(re.finditer(r"\[(\d+)/(\d+)\]", text))
                if ms:
                    a, b = int(ms[-1].group(1)), int(ms[-1].group(2))
                    with JOB.lock:
                        JOB.frame = a
                        JOB.n = b
                        JOB.progress = (a / b) if b else 0.0
                        JOB.message = f"frame {a}/{b}"
                        status_path.write_text(json.dumps(JOB.snapshot(), indent=2))
            except Exception:
                pass
            time.sleep(0.8)

        rc = proc.returncode
        # Find newest mp4
        mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        with JOB.lock:
            JOB.finished_at = time.time()
            JOB.proc = None
            if rc == 0 and mp4s:
                JOB.state = "done"
                JOB.path = str(mp4s[0])
                JOB.relpath = f"videos/{mp4s[0].name}"
                JOB.progress = 1.0
                JOB.message = f"done → {mp4s[0].name}"
            else:
                JOB.state = "error"
                JOB.message = f"exporter failed rc={rc}; see videos/job_export.log"
            status_path.write_text(json.dumps(JOB.snapshot(), indent=2))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, scene_dir: Path, **kwargs):
        self.scene_dir = scene_dir.resolve()
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def _send_json(self, obj: dict, code: int = 200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/video/export":
            body = self._read_json_body()
            with JOB.lock:
                if JOB.state == "running":
                    self._send_json({"ok": False, "error": "job already running", **JOB.snapshot()}, 409)
                    return
                JOB.job_id = uuid.uuid4().hex[:10]
            t = threading.Thread(
                target=_run_export,
                args=(self.scene_dir, body),
                daemon=True,
            )
            t.start()
            self._send_json({"ok": True, **JOB.snapshot()})
            return
        self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/video/status":
            # Prefer live JOB; fall back to on-disk status
            snap = JOB.snapshot()
            disk = self.scene_dir / "videos" / "job_status.json"
            if snap["state"] == "idle" and disk.is_file():
                try:
                    snap = json.loads(disk.read_text())
                except Exception:
                    pass
            self._send_json(snap)
            return

        if path == "/api/video/list":
            vdir = self.scene_dir / "videos"
            items = []
            if vdir.is_dir():
                for p in sorted(vdir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
                    items.append(
                        {
                            "name": p.name,
                            "relpath": f"videos/{p.name}",
                            "url": f"/videos/{p.name}",
                            "bytes": p.stat().st_size,
                            "mtime": p.stat().st_mtime,
                        }
                    )
            self._send_json({"videos": items})
            return

        if path.startswith("/videos/"):
            name = path[len("/videos/") :]
            if "/" in name or name.startswith(".") or ".." in name:
                self.send_error(400)
                return
            target = (self.scene_dir / "videos" / name).resolve()
            if not str(target).startswith(str(self.scene_dir / "videos")) or not target.is_file():
                self.send_error(404)
                return
            # Serve file
            ctype = "video/mp4" if target.suffix == ".mp4" else "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()

    def translate_path(self, path: str) -> str:
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
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
    print(f"video export API: POST /api/video/export  GET /api/video/status|/api/video/list")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
