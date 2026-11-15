"""実験設定(YAML)のスキーマ。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forget_bench.scenario.splits import seeds_for, split_of
from forget_bench.types import Lang, ScenarioParams, Split


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(_Strict):
    max_tokens: int = 256
    temperature: float = 0.0


class ScenarioConfig(_Strict):
    langs: list[Lang] = ["ja"]
    split: Split = "dev"
    n_scenarios: int | None = None  # seeds を省略したとき、分割の先頭から n 個
    seeds: list[int] | None = None
    sessions: int = 5
    turns_per_session: tuple[int, int] = (10, 20)
    n_attributes: int = 4
    max_changes: int = 4
    n_oneshot: int = 3
    n_sensitive: int = 3
    n_forget: int = 2
    aggregate_mentions: int = 4

    @model_validator(mode="after")
    def _check_seeds(self) -> "ScenarioConfig":
        if (self.seeds is None) == (self.n_scenarios is None):
            raise ValueError("seeds か n_scenarios のどちらか一方を指定する")
        for s in self.resolved_seeds():
            if split_of(s) != self.split:
                raise ValueError(f"seed {s} は {self.split} 分割ではない")
        return self

    def resolved_seeds(self) -> list[int]:
        if self.seeds is not None:
            return list(self.seeds)
        assert self.n_scenarios is not None
        return seeds_for(self.split, self.n_scenarios)

    def params(self, lang: Lang) -> ScenarioParams:
        return ScenarioParams(
            lang=lang,
            sessions=self.sessions,
            turns_per_session=tuple(self.turns_per_session),  # type: ignore[arg-type]
            n_attributes=self.n_attributes,
            max_changes=self.max_changes,
            n_oneshot=self.n_oneshot,
            n_sensitive=self.n_sensitive,
            n_forget=self.n_forget,
            aggregate_mentions=self.aggregate_mentions,
        )


class StrategyConfig(_Strict):
    name: str
    label: str | None = None  # 結果表での名前。同じ戦略を別パラメータで並べるとき用
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.label or self.name


class RunConfig(_Strict):
    name: str
    backend: BackendConfig
    generation: GenerationConfig = GenerationConfig()
    scenario: ScenarioConfig
    strategies: list[StrategyConfig]

    @model_validator(mode="after")
    def _unique_labels(self) -> "RunConfig":
        labels = [s.display for s in self.strategies]
        if len(labels) != len(set(labels)):
            raise ValueError(f"戦略の label が重複している: {labels}")
        return self


def load_config(path: str | Path) -> tuple[RunConfig, str]:
    """設定と、meta.json に残すための原文を返す。"""
    text = Path(path).read_text(encoding="utf-8")
    return RunConfig.model_validate(yaml.safe_load(text)), text
