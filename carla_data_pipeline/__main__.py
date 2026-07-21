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
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
