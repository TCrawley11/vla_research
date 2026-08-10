"""CLI entry point: python -m carla_data_pipeline {collect,build-samples,man}.

`carla` is imported lazily inside the collect path only, so config validation
(--dry-run), build-samples and man all work without the CARLA wheel or server.
"""
import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from .config.load import ConfigError, load_collect_config

DEFAULT_DATA_DIR = Path("data/runs")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m carla_data_pipeline",
        description="Config-driven headless CARLA data collection. "
                    "See the `man` subcommand for the full manual.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Stage 1: capture a run from a scenario config")
    p_collect.add_argument("scenario", type=Path, help="path to a configs/scenarios/*.yaml")
    p_collect.add_argument("--run-id", help="override the auto-assigned runNN id")
    p_collect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                           help="output directory (default: data/runs)")
    mode = p_collect.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="validate and print the resolved config; never touches CARLA")
    mode.add_argument("--verify-only", action="store_true",
                      help="connect and check map/spawn/blueprints, spawn nothing")

    p_build = sub.add_parser("build-samples",
                             help="Stage 2: build sample groups into an existing run .h5")
    p_build.add_argument("run", help="run id (e.g. run01) or a path to a .h5")
    p_build.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="where run ids are looked up (default: data/runs)")
    p_build.add_argument("--no-upload", action="store_true",
                         help="skip the automatic stage-3 upload even if the run's "
                              "config enables it")

    p_upload = sub.add_parser(
        "upload", help="Stage 3: upload finished runs to the configured hub repo")
    p_upload.add_argument("run", nargs="?",
                          help="run id (e.g. run01) or a path to a .h5")
    p_upload.add_argument("--all", action="store_true",
                          help="upload every run in --data-dir whose samples are built")
    p_upload.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                          help="where run ids are looked up (default: data/runs)")

    p_video = sub.add_parser(
        "export-video", help="rebuild per-camera mp4s from a run .h5 (viewing tool)")
    p_video.add_argument("run", help="run id (e.g. run07) or a path to a .h5")
    p_video.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="where run ids are looked up (default: data/runs)")
    p_video.add_argument("--out-dir", type=Path, default=Path("data/videos"),
                         help="output root; videos land in <out-dir>/<run_id>/ "
                              "(default: data/videos)")
    p_video.add_argument("--camera", action="append", metavar="CAM",
                         help="camera name, e.g. FRONT; repeat for several "
                              "(default: all cameras)")
    p_video.add_argument("--crf", type=int, default=18,
                         help="x264 quality, lower is better (default: 18)")

    p_man = sub.add_parser("man", help="show the manual")
    p_man.add_argument("topic", nargs="?", choices=["usage", "config"], default="usage",
                       help="usage (default) or the generated config reference")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "man":
        from . import man
        print(man.render(args.topic))
        return 0

    if args.command == "collect":
        try:
            cfg = load_collect_config(args.scenario)
        except (ConfigError, ValidationError) as exc:
            print(f"invalid config {args.scenario}:\n{exc}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(cfg.model_dump_json(indent=2))
            return 0
        from . import collect
        collect.collect(cfg, run_id=args.run_id, data_dir=args.data_dir,
                        verify_only=args.verify_only)
        return 0

    if args.command == "build-samples":
        path = Path(args.run) if args.run.endswith(".h5") else args.data_dir / f"{args.run}.h5"
        if not path.is_file():
            print(f"no such run file: {path}", file=sys.stderr)
            return 2
        from .build_samples import build_samples
        build_samples(path)
        if args.no_upload:
            return 0
        return _auto_upload(path)

    if args.command == "export-video":
        path = Path(args.run) if args.run.endswith(".h5") else args.data_dir / f"{args.run}.h5"
        if not path.is_file():
            print(f"no such run file: {path} (uploaded runs are pruned locally - "
                  "`hf download` it first)", file=sys.stderr)
            return 2
        from .video import VideoError, export_videos
        try:
            for out in export_videos(path, args.out_dir, cameras=args.camera,
                                     crf=args.crf):
                print(out)
        except VideoError as exc:
            print(f"export-video: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "upload":
        from .upload import UploadError, upload_cfg_from_sidecar, upload_run
        if bool(args.run) == args.all:
            print("upload: give a run id/path or --all (not both)", file=sys.stderr)
            return 2
        if args.all:
            targets = sorted(p.with_suffix(".h5")
                             for p in args.data_dir.glob("run*.json"))
        else:
            targets = [Path(args.run) if args.run.endswith(".h5")
                       else args.data_dir / f"{args.run}.h5"]
        failures = 0
        for h5_path in targets:
            try:
                ucfg = upload_cfg_from_sidecar(h5_path.with_suffix(".json"))
                if not ucfg.enabled:
                    msg = f"{h5_path.stem}: upload not enabled in this run's config"
                    if args.all:
                        logging.info("%s; skipping", msg)
                        continue
                    print(msg, file=sys.stderr)
                    return 2
                upload_run(h5_path, ucfg)
            except (OSError, ValueError, UploadError) as exc:
                logging.error("%s: %s", h5_path.stem, exc)
                failures += 1
        return 1 if failures else 0

    return 2


def _auto_upload(h5_path: Path) -> int:
    """Stage-3 hook after build-samples: no-op unless the run's recorded config
    enables it. A failed upload fails the command; `upload --all` backfills."""
    from .upload import UploadError, upload_cfg_from_sidecar, upload_run
    sidecar = h5_path.with_suffix(".json")
    if not sidecar.is_file():
        return 0
    ucfg = upload_cfg_from_sidecar(sidecar)
    if not (ucfg.enabled and ucfg.auto):
        return 0
    try:
        upload_run(h5_path, ucfg)
        return 0
    except (OSError, ValueError, UploadError) as exc:
        logging.error("auto-upload failed (samples are built; retry with "
                      "`upload %s`): %s", h5_path.stem, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
