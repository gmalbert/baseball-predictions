# src/ingestion/config.py
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class IngestionConfig:
    """Central config for all ingestion jobs."""

    # Date range
    start_year: int = 2020
    end_year: int = field(default_factory=lambda: date.today().year)

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    # API keys (from environment variables)
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    therundown_api_key: str = field(default_factory=lambda: os.getenv("THERUNDOWN_API_KEY", ""))

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
        for subdir in ["gamelogs", "batting", "pitching", "odds", "therundown", "weather"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


config = IngestionConfig()
