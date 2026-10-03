#!/usr/bin/env python3
"""
H200 direct-URL downloader.

Watches .urlqueue/<id>.url files (written by the local residential-IP resolver
and rsynced over). Each file holds a direct googlevideo media URL. Downloads it
with aria2c (multi-connection) straight into videos/, bypassing YouTube's
datacenter-IP bot block, then appends to archive.txt.

The frame extractor picks up the completed videos/ files automatically.
"""

import os
import time
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_PROJECT_DIR = os.environ.get("AYEKOO_PROJECT_DIR", "/mnt/volume_d2wey28/projects/ayekoo-videos")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def load_archive(archive_path: Path):
    done = set()
    if archive_path.exists():
        for line in archive_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                done.add(parts[-1])
    return done


class Downloader:
    def __init__(self, project_dir, workers, poll, conn, timeout):
        self.P = Path(project_dir)
        self.qdir = self.P / ".urlqueue"
        self.fdir = self.P / ".urlqueue_failed"
        self.videos = self.P / "videos"
        self.archive = self.P / "archive.txt"
        for d in (self.qdir, self.fdir, self.videos):
            d.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.poll = poll
        self.conn = conn
        self.timeout = timeout
        self.inflight = set()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def download(self, url, vid):
        tmp = self.videos / f".{vid}.part.mp4"
        final = self.videos / f"{vid}.mp4"
        qf = self.qdir / f"{vid}.url"
        tmp.unlink(missing_ok=True)
        cmd = [
            "aria2c", "--dir", str(self.videos), "-o", f".{vid}.part.mp4",
            "-x", str(self.conn), "-s", str(self.conn), "-k", "1M",
            "--file-allocation=none", "--summary-interval=0",
            "--max-tries=5", "--retry-wait=5", "--allow-overwrite=true",
            "--header", f"User-Agent: {UA}", url,
        ]
        r = None
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            pass
        ok = (r is not None and r.returncode == 0
              and tmp.exists() and tmp.stat().st_size > 65536)
        if ok:
            try:
                v = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "default=nw=1", str(tmp)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=120)
                ok = (v.returncode == 0)
            except subprocess.TimeoutExpired:
                ok = False
        if ok:
            os.replace(tmp, final)
            with self.lock:
                with open(self.archive, "a") as f:
                    f.write(f"youtube {vid}\n")
            qf.unlink(missing_ok=True)
            print(f"[{time.strftime('%F %T')}] OK   {vid} "
                  f"({final.stat().st_size / 1e6:.1f} MB)", flush=True)
        else:
            tmp.unlink(missing_ok=True)
            err = (r.stderr.decode("utf-8", "replace")[-200:] if r
                   else "timeout")
            if qf.exists():
                qf.replace(self.fdir / qf.name)
            print(f"[{time.strftime('%F %T')}] FAIL {vid}: {err}", flush=True)
        self.inflight.discard(vid)

    def run(self):
        print(f"URL downloader started (workers={self.workers}, "
              f"conn={self.conn}, poll={self.poll}s)", flush=True)
        while True:
            done = load_archive(self.archive)
            submitted = 0
            for uf in sorted(self.qdir.glob("*.url")):
                vid = uf.stem
                if vid in self.inflight:
                    continue
                if vid in done or (self.videos / f"{vid}.mp4").exists():
                    uf.unlink(missing_ok=True)
                    continue
                if len(self.inflight) >= self.workers:
                    break
                url = uf.read_text().strip()
                if not url:
                    uf.unlink(missing_ok=True)
                    continue
                self.inflight.add(vid)
                self.pool.submit(self.download, url, vid)
                submitted += 1
            time.sleep(2 if submitted else self.poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--conn", type=int, default=16)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    Downloader(a.project_dir, a.workers, a.poll, a.conn, a.timeout).run()
