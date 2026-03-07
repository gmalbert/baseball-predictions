# src/ingestion/config.py
from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class IngestionConfig:
    """Central config for all ingestion jobs."""

    # Date range
    start_year: int = 2021
    end_year: int = 2025

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # API keys (from environment variables)
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))

    # Rate limiting
    request_delay_sec: float = 1.0  # polite delay between API calls

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data_files" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data_files" / "processed"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for subdir in ["gamelogs", "batting", "pitching", "odds", "weather"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


config = IngestionConfig()
