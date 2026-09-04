#!/usr/bin/env python3
"""Serve occ_viewer + one or more scene packages.

Endpoints:
  GET  /                     viewer
  GET  /api/clips            list available clips
  GET  /scenes/<clip>/...    scene assets (multi-clip)
  GET  /scene/...            alias of default / first clip (compat)
  POST /api/video/export     start export {clip_id?, mode,fps,...}
  GET  /api/video/status
  GET  /api/video/list?clip_id=
  GET  /videos/<clip>/<file> download mp4
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
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = Path(__file__).resolve().parent
EXPORT_SCRIPT = ROOT / "tools" / "occ" / "scene_video.py"


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
        self.clip_id: str | None = None
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
                "clip_id": self.clip_id,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "params": self.params,
            }


JOB = VideoJob()


def _python_bin() -> str:
    for cand in (
        ROOT / ".venv_smoke" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ):
        if Path(cand).is_file():
            return str(cand)
    return sys.executable


def discover_clips(scenes_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not scenes_root.is_dir():
        return out
    for p in sorted(scenes_root.iterdir()):
        if p.is_dir() and (p / "index.json").is_file():
            out[p.name] = p.resolve()
    return out


def clip_info(clip_id: str, scene_dir: Path) -> dict:
    n_frames = 0
    schema = None
    try:
        idx = json.loads((scene_dir / "index.json").read_text())
        n_frames = len(idx.get("frames") or [])
        schema = idx.get("schema_version")
    except Exception:
        pass
    return {
        "id": clip_id,
        "clip_id": clip_id,
        "n_frames": n_frames,
        "schema_version": schema,
        "url": f"/scenes/{clip_id}",
    }


def _run_export(scene_dir: Path, clip_id: str, params: dict) -> None:
    import os
    import re

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
        JOB.clip_id = clip_id
        JOB.started_at = time.time()
        JOB.finished_at = None
        JOB.params = dict(params)
        status_path.write_text(json.dumps(JOB.snapshot(), indent=2))

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

        while proc.poll() is None:
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
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
    def __init__(self, *args, clips: dict[str, Path], default_clip: str, **kwargs):
        self.clips = clips
        self.default_clip = default_clip
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def _scene(self, clip_id: str | None = None) -> Path | None:
        cid = clip_id or self.default_clip
        return self.clips.get(cid)

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
            clip_id = body.get("clip_id") or self.default_clip
            scene_dir = self._scene(clip_id)
            if scene_dir is None:
                self._send_json({"ok": False, "error": f"unknown clip_id={clip_id}"}, 400)
                return
            with JOB.lock:
                if JOB.state == "running":
                    self._send_json(
                        {"ok": False, "error": "job already running", **JOB.snapshot()},
                        409,
                    )
                    return
                JOB.job_id = uuid.uuid4().hex[:10]
            t = threading.Thread(
                target=_run_export,
                args=(scene_dir, clip_id, body),
                daemon=True,
            )
            t.start()
            self._send_json({"ok": True, **JOB.snapshot()})
            return
        self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)

        if path == "/api/clips":
            items = [clip_info(cid, p) for cid, p in self.clips.items()]
            self._send_json(
                {
                    "default_clip": self.default_clip,
                    "clips": items,
                }
            )
            return

        if path == "/api/video/status":
            snap = JOB.snapshot()
            clip_id = qs.get("clip_id", [self.default_clip])[0]
            scene = self._scene(clip_id)
            if snap["state"] == "idle" and scene is not None:
                disk = scene / "videos" / "job_status.json"
                if disk.is_file():
                    try:
                        snap = json.loads(disk.read_text())
                    except Exception:
                        pass
            self._send_json(snap)
            return

        if path == "/api/video/list":
            clip_id = qs.get("clip_id", [self.default_clip])[0]
            scene = self._scene(clip_id)
            items = []
            if scene is not None:
                vdir = scene / "videos"
                if vdir.is_dir():
                    for p in sorted(
                        vdir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True
                    ):
                        items.append(
                            {
                                "name": p.name,
                                "relpath": f"videos/{p.name}",
                                "url": f"/videos/{clip_id}/{p.name}",
                                "bytes": p.stat().st_size,
                                "mtime": p.stat().st_mtime,
                                "clip_id": clip_id,
                            }
                        )
            self._send_json({"videos": items, "clip_id": clip_id})
            return

        if path.startswith("/videos/"):
            rest = path[len("/videos/") :]
            parts = rest.split("/", 1)
            if len(parts) == 1:
                clip_id, name = self.default_clip, parts[0]
            else:
                clip_id, name = parts[0], parts[1]
            if "/" in name or name.startswith(".") or ".." in name or ".." in clip_id:
                self.send_error(400)
                return
            scene = self._scene(clip_id)
            if scene is None:
                self.send_error(404)
                return
            target = (scene / "videos" / name).resolve()
            if not str(target).startswith(str(scene / "videos")) or not target.is_file():
                self.send_error(404)
                return
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

        # /scenes/<clip_id>/...
        if p.startswith("/scenes/"):
            rest = p[len("/scenes/") :]
            if not rest:
                return str(VIEWER_DIR / "index.html")
            parts = rest.split("/", 1)
            clip_id = parts[0]
            rel = parts[1] if len(parts) > 1 else ""
            scene = self._scene(clip_id)
            if scene is None:
                return str(VIEWER_DIR / "index.html")
            target = (scene / rel).resolve() if rel else scene
            if not str(target).startswith(str(scene)):
                return str(VIEWER_DIR / "index.html")
            return str(target)

        # /scene/... → default clip (backward compatible)
        if p.startswith("/scene/") or p == "/scene":
            rel = p[len("/scene/") :] if p.startswith("/scene/") else ""
            scene = self._scene(self.default_clip)
            if scene is None:
                return str(VIEWER_DIR / "index.html")
            target = (scene / rel).resolve() if rel else scene
            if not str(target).startswith(str(scene)):
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
        "--scenes-root",
        default="",
        help="Parent directory containing multiple scene packages",
    )
    ap.add_argument(
        "--scene",
        default="",
        help="Single scene root (compat). If set with --scenes-root, becomes default clip.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    clips: dict[str, Path] = {}
    if args.scenes_root:
        clips = discover_clips(Path(args.scenes_root).resolve())
    if args.scene:
        scene = Path(args.scene).resolve()
        if not (scene / "index.json").is_file():
            raise SystemExit(f"missing index.json under {scene}")
        clips[scene.name] = scene
    if not clips:
        raise SystemExit("provide --scenes-root and/or --scene with valid packages")

    default_clip = Path(args.scene).resolve().name if args.scene else next(iter(clips))
    if default_clip not in clips:
        default_clip = next(iter(clips))

    handler = functools.partial(Handler, clips=clips, default_clip=default_clip)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"clips ({len(clips)}): {', '.join(clips)}")
    print(f"default: {default_clip}")
    print(f"open:  {url}")
    print(f"api:   GET /api/clips")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
