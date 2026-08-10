"""export-video tests: a tiny synthetic run encodes to real mp4s via ffmpeg
(skipped if ffmpeg is absent); error paths need no encoder at all."""
import shutil

import h5py
import numpy as np
import pytest

from carla_data_pipeline.video import VideoError, export_videos

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def make_run(tmp_path, cams=("FRONT", "BACK"), n=8, w=48, h=32):
    path = tmp_path / "run99.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["run_id"] = "run99"
        f.attrs["raw_fps"] = 4
        g = f.create_group("images")
        for cam in cams:
            g.create_dataset(cam, data=rng.integers(0, 255, (n, h, w, 3),
                                                    dtype=np.uint8))
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_exports_one_mp4_per_camera(tmp_path):
    path = make_run(tmp_path)
    outs = export_videos(path, tmp_path / "videos")
    assert sorted(o.name for o in outs) == ["BACK.mp4", "FRONT.mp4"]
    for o in outs:
        assert o.parent.name == "run99"
        assert o.stat().st_size > 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_camera_selection(tmp_path):
    path = make_run(tmp_path)
    outs = export_videos(path, tmp_path / "videos", cameras=["BACK"])
    assert [o.name for o in outs] == ["BACK.mp4"]


def test_unknown_camera_is_named(tmp_path):
    path = make_run(tmp_path)
    with pytest.raises(VideoError, match="SIDE"):
        export_videos(path, tmp_path / "videos", cameras=["FRONT", "SIDE"])
