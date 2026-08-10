"""Rebuild watchable video from a run .h5: one mp4 per camera.

Frames stream straight from the h5 into ffmpeg's stdin (rawvideo rgb24), so
a full-size run never has to fit in memory. Frame rate comes from the run's
`raw_fps` attr; output is h264/yuv420p - the even-resolution rule in the
camera schema exists exactly for this. Purely a viewing tool: it reads the
h5 and touches neither the run nor its sidecar.
"""
import logging
import shutil
import subprocess
from pathlib import Path

import h5py

log = logging.getLogger(__name__)

BATCH_FRAMES = 32


class VideoError(RuntimeError):
    """ffmpeg missing/failed or a requested camera does not exist."""


def export_videos(h5_path, out_root, cameras=None, crf=18) -> list[Path]:
    """Encode one <out_root>/<run_id>/<CAM>.mp4 per camera. Returns the paths."""
    h5_path = Path(h5_path)
    outputs = []
    with h5py.File(h5_path, "r") as f:
        available = list(f["images"].keys())
        targets = list(cameras) if cameras else available
        missing = sorted(set(targets) - set(available))
        if missing:
            raise VideoError(f"no such camera(s) {missing}; "
                             f"this run has {sorted(available)}")
        if shutil.which("ffmpeg") is None:
            raise VideoError("ffmpeg not found on PATH; install it to export video")
        fps = int(f.attrs["raw_fps"])
        out_dir = Path(out_root) / str(f.attrs["run_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for cam in targets:
            ds = f["images"][cam]
            if ds.shape[0] == 0:
                log.warning("%s: no frames; skipping", cam)
                continue
            out = out_dir / f"{cam}.mp4"
            _encode(ds, out, fps, crf)
            log.info("%s: %d frames @ %d fps -> %s", cam, ds.shape[0], fps, out)
            outputs.append(out)
    return outputs


def _encode(ds, out: Path, fps: int, crf: int):
    n, height, width, _ = ds.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(0, n, BATCH_FRAMES):
            proc.stdin.write(ds[i:i + BATCH_FRAMES].tobytes())
    except BrokenPipeError:
        pass  # ffmpeg died; its stderr is reported below
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    stderr = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        out.unlink(missing_ok=True)  # no half-written mp4s
        raise VideoError(f"ffmpeg failed for {out.name}: {stderr.strip()[:500]}")
