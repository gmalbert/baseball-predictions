"""Compatibility entrypoint for ``python -m src.pipeline.replay``."""

from src.pipelines.replay import *  # noqa: F403
from src.pipelines.replay import main

if __name__ == "__main__":
    main()
