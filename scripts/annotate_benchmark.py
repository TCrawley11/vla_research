"""CLI entry point for the hosted annotation benchmark."""
import sys
from pathlib import Path

# Support running this file directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.benchmark import main

if __name__ == "__main__":
    main()
