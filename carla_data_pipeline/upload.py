"""Stage 3: upload a finished run to a Hugging Face dataset repo.

Ships `<run_id>.h5` + `<run_id>.json` to `<path_prefix>/` in the configured
dataset repo (created private on first use), verifies the remote file size,
stamps the sidecar (`status` -> "uploaded", plus an `upload` record) and
optionally deletes the local .h5. Only runs with built samples may be
uploaded: with `delete_local_h5` an earlier upload would destroy the input
build-samples still needs. Re-running is idempotent - anything already on
the Hub with the right size is skipped.

Upload settings come from the run's own sidecar (`config.upload`), i.e. the
destination is fixed at collect time and recorded with the run.

Auth: `hf auth login` once, or set HF_TOKEN.
"""
import datetime
import json
import logging
from pathlib import Path

from .config.schema import UploadConfig

log = logging.getLogger(__name__)


class UploadError(RuntimeError):
    """A precondition or verification failure; the local run is left intact."""


def _utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _default_api():
    from huggingface_hub import HfApi
    return HfApi()


def _remote_size(api, repo_id: str, path_in_repo: str):
    infos = api.get_paths_info(repo_id, [path_in_repo], repo_type="dataset")
    return infos[0].size if infos else None


def upload_run(h5_path: Path, ucfg: UploadConfig, api=None) -> dict:
    """Upload one run's .h5 + sidecar. Returns the sidecar's `upload` record."""
    if ucfg.backend != "hf" or not ucfg.repo_id:
        raise UploadError("upload config has no hf repo_id; set upload.repo_id")
    api = api or _default_api()
    h5_path = Path(h5_path)
    sidecar_path = h5_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise UploadError(f"no sidecar at {sidecar_path}; nothing to describe the run")
    sidecar = json.loads(sidecar_path.read_text())
    run_id = sidecar.get("run_id", h5_path.stem)
    status = sidecar.get("status")
    record = sidecar.get("upload")
    remote_h5 = f"{ucfg.path_prefix}/{h5_path.name}"

    if not h5_path.exists():
        if record and record.get("verified_size"):
            log.info("%s: already uploaded (local h5 deleted); nothing to do", run_id)
            return record
        raise UploadError(f"{h5_path} is missing and the sidecar has no upload record")

    if status not in ("samples_built", "uploaded"):
        raise UploadError(
            f"run {run_id} has status '{status}'; run build-samples first - "
            "upload only ships finished runs")

    api.create_repo(repo_id=ucfg.repo_id, repo_type="dataset",
                    private=ucfg.private, exist_ok=True)

    local_size = h5_path.stat().st_size
    if _remote_size(api, ucfg.repo_id, remote_h5) == local_size:
        log.info("%s: %s already on %s with matching size; skipping transfer",
                 run_id, remote_h5, ucfg.repo_id)
    else:
        log.info("%s: uploading %.2f GiB to %s/%s ...",
                 run_id, local_size / 2 ** 30, ucfg.repo_id, remote_h5)
        api.upload_file(path_or_fileobj=str(h5_path), path_in_repo=remote_h5,
                        repo_id=ucfg.repo_id, repo_type="dataset",
                        commit_message=f"{run_id}: run data")
        remote = _remote_size(api, ucfg.repo_id, remote_h5)
        if remote != local_size:
            raise UploadError(
                f"{run_id}: remote size {remote} != local {local_size} after "
                "upload; local h5 kept")

    record = {"backend": "hf", "repo_id": ucfg.repo_id, "path_in_repo": remote_h5,
              "verified_size": local_size, "uploaded_utc": _utcnow()}
    sidecar["upload"] = record
    sidecar["status"] = "uploaded"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    api.upload_file(path_or_fileobj=str(sidecar_path),
                    path_in_repo=f"{ucfg.path_prefix}/{sidecar_path.name}",
                    repo_id=ucfg.repo_id, repo_type="dataset",
                    commit_message=f"{run_id}: sidecar")

    if ucfg.delete_local_h5:
        h5_path.unlink()
        log.info("%s: verified on %s; deleted local %s",
                 run_id, ucfg.repo_id, h5_path.name)
    return record


def upload_cfg_from_sidecar(sidecar_path: Path) -> UploadConfig:
    """The UploadConfig recorded in a run's sidecar (defaults if absent)."""
    data = json.loads(Path(sidecar_path).read_text())
    return UploadConfig.model_validate((data.get("config") or {}).get("upload") or {})
