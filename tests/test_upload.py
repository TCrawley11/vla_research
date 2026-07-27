"""Stage-3 upload tests: all Hub interaction goes through a fake api object,
so nothing here touches the network."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_data_pipeline.config.schema import UploadConfig
from carla_data_pipeline.upload import UploadError, upload_cfg_from_sidecar, upload_run


class FakeApi:
    def __init__(self):
        self.remote = {}   # path_in_repo -> size
        self.created = []
        self.uploads = []  # path_in_repo, in call order

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created.append((repo_id, repo_type, private, exist_ok))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type,
                    commit_message):
        self.uploads.append(path_in_repo)
        self.remote[path_in_repo] = Path(path_or_fileobj).stat().st_size

    def get_paths_info(self, repo_id, paths, repo_type):
        return [SimpleNamespace(size=self.remote[p]) for p in paths
                if p in self.remote]


class LossyApi(FakeApi):
    """Stores one byte short, so post-upload verification must fail."""

    def upload_file(self, **kwargs):
        super().upload_file(**kwargs)
        self.remote[kwargs["path_in_repo"]] -= 1


UCFG = UploadConfig(enabled=True, repo_id="org/data", delete_local_h5=True)


def make_run(tmp_path, status="samples_built"):
    h5 = tmp_path / "run01.h5"
    h5.write_bytes(b"x" * 100)
    (tmp_path / "run01.json").write_text(json.dumps(
        {"run_id": "run01", "status": status,
         "config": {"upload": UCFG.model_dump()}}))
    return h5


def test_upload_verifies_stamps_and_deletes(tmp_path):
    h5 = make_run(tmp_path)
    api = FakeApi()
    record = upload_run(h5, UCFG, api=api)
    assert api.created == [("org/data", "dataset", True, True)]
    assert api.uploads == ["runs/run01.h5", "runs/run01.json"]
    assert record["verified_size"] == 100
    assert not h5.exists()  # delete_local_h5
    side = json.loads((tmp_path / "run01.json").read_text())
    assert side["status"] == "uploaded"
    assert side["upload"]["path_in_repo"] == "runs/run01.h5"


def test_keeps_local_h5_by_default(tmp_path):
    h5 = make_run(tmp_path)
    upload_run(h5, UploadConfig(enabled=True, repo_id="org/data"), api=FakeApi())
    assert h5.exists()


def test_skips_transfer_when_remote_matches(tmp_path):
    h5 = make_run(tmp_path)
    api = FakeApi()
    api.remote["runs/run01.h5"] = 100
    upload_run(h5, UCFG, api=api)
    assert api.uploads == ["runs/run01.json"]  # only the sidecar moved
    assert not h5.exists()  # still verified + pruned


def test_refuses_unbuilt_run(tmp_path):
    h5 = make_run(tmp_path, status="collected")
    with pytest.raises(UploadError, match="build-samples"):
        upload_run(h5, UCFG, api=FakeApi())
    assert h5.exists()


def test_size_mismatch_keeps_h5_and_status(tmp_path):
    h5 = make_run(tmp_path)
    with pytest.raises(UploadError, match="size"):
        upload_run(h5, UCFG, api=LossyApi())
    assert h5.exists()
    assert json.loads((tmp_path / "run01.json").read_text())["status"] == "samples_built"


def test_missing_h5_with_record_is_noop(tmp_path):
    h5 = make_run(tmp_path)
    api = FakeApi()
    record = upload_run(h5, UCFG, api=api)
    n_uploads = len(api.uploads)
    assert upload_run(h5, UCFG, api=api) == record  # h5 already deleted
    assert len(api.uploads) == n_uploads


def test_missing_h5_without_record_raises(tmp_path):
    h5 = make_run(tmp_path)
    h5.unlink()
    with pytest.raises(UploadError, match="missing"):
        upload_run(h5, UCFG, api=FakeApi())


def test_upload_cfg_from_sidecar(tmp_path):
    make_run(tmp_path)
    ucfg = upload_cfg_from_sidecar(tmp_path / "run01.json")
    assert ucfg == UCFG


def test_upload_cfg_defaults_for_pre_upload_sidecars(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps(
        {"run_id": "old", "status": "samples_built", "config": {}}))
    assert upload_cfg_from_sidecar(tmp_path / "old.json").enabled is False
