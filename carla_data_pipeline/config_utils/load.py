"""Loads a scenario YAML into a validated CollectConfig.

Two reference kinds, both resolved relative to the file that declares them:
  extends: <path>      deep-merge this file over the referenced one (this file wins)
  camera_spec: <path>  load the referenced file as its own CameraSpecConfig model,
                       never merged key-by-key
"""
from pathlib import Path

import yaml

from .schema import CollectConfig


class ConfigError(ValueError):
    """A config file reference or format problem (as opposed to a schema violation)."""


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config file must be a YAML mapping: {path}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_with_extends(path: Path, seen: tuple[Path, ...]) -> dict:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ConfigError(f"extends cycle: {chain}")
    data = _read_yaml(path)
    # anchor the camera_spec path to the file that declared it, before any merge
    if isinstance(data.get("camera_spec"), str):
        data["camera_spec"] = str((path.parent / data["camera_spec"]).resolve())
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    parent = _load_with_extends(path.parent / parent_ref, (*seen, path))
    return _deep_merge(parent, data)


def load_collect_config(path: str | Path) -> CollectConfig:
    """Resolve extends/camera_spec references and validate into a CollectConfig."""
    raw = _load_with_extends(Path(path), seen=())
    spec_ref = raw.get("camera_spec")
    if isinstance(spec_ref, str):
        raw["camera_spec"] = _read_yaml(Path(spec_ref))
    return CollectConfig.model_validate(raw)
