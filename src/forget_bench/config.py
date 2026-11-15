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
    sessions: int | list[int] = 5  # リストなら規模スイープ(例: [5, 20, 50])
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

    def session_list(self) -> list[int]:
        return [self.sessions] if isinstance(self.sessions, int) else list(self.sessions)

    def params(self, lang: Lang, sessions: int | None = None) -> ScenarioParams:
        if sessions is None:
            if not isinstance(self.sessions, int):
                raise ValueError("sessions がリストのときは sessions を指定する")
            sessions = self.sessions
        return ScenarioParams(
            lang=lang,
            sessions=sessions,
            turns_per_session=tuple(self.turns_per_session),  # type: ignore[arg-type]
            n_attributes=self.n_attributes,
            max_changes=self.max_changes,
            n_oneshot=self.n_oneshot,
            n_sensitive=self.n_sensitive,
            n_forget=self.n_forget,
            aggregate_mentions=self.aggregate_mentions,
        )


class EmbedderConfig(_Strict):
    name: str = "ollama"
    params: dict[str, Any] = Field(default_factory=lambda: {"model": "bge-m3"})


class RetrievalConfig(_Strict):
    embedder: EmbedderConfig = EmbedderConfig()


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
    # 同じ入力を何回送るか。温度を指定できないバックエンド(claude)の揺れを測る(docs/protocol.md)。
    # ollama/hf は seed 固定で決定的なので 1 にし、揺れは別 seed で見る。
    repeats: int = Field(default=1, ge=1)
    # 見積もりリクエスト数の上限。超える設定は実行前に止める(利用枠のあるバックエンド用)。
    max_requests: int | None = None
    # (シナリオ × 戦略 × 繰り返し)の単位を何並列で回すか。利用枠は減らないが、待ち時間が減る。
    # ollama は1つのGPUで処理するので 1 のままにする(並列にしても速くならず、レイテンシの比較が壊れる)。
    concurrency: int = Field(default=1, ge=1)
    # rag / hybrid の検索器(回答バックエンドによらず共通)
    retrieval: RetrievalConfig = RetrievalConfig()

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
