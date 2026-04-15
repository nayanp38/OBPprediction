from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    start_season: int = 2018
    end_season: int = 2023
    min_qual_pa: int = 50
    min_model_pa: int = 100
    include_season_dummies: bool = True

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42

    output_dir: Path = Path("outputs")

    @property
    def seasons(self) -> list[int]:
        return list(range(self.start_season, self.end_season + 1))
