#!/usr/bin/env python
"""Lightweight CLI that runs the class-based pipeline in `src`.

Usage examples:
  train:        docker-compose run --rm app train
  run all:      docker-compose run --rm app run
  customize:    docker-compose run --rm app run --config config/my.yaml
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path so imports from `src` work when running as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml_project.pipelines.pipeline import TaxiPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the class-based TaxiPipeline from `src`.")
    parser.add_argument("action", nargs="?", default="run", choices=["run"],
                        help="Action to perform (currently only 'run' which trains + runs inference).")
    parser.add_argument("--config", "-c", default="config/config.yaml",
                        help="Path to the config YAML file (default: config/config.yaml)")

    args = parser.parse_args()

    pipeline = TaxiPipeline(args.config)

    if args.action == "run":
        pipeline.run()


if __name__ == "__main__":
    main()
