"""メモリ戦略の契約(AGENTS.md「インターフェース」)とプラグイン登録。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from forget_bench.backends.base import Backend
from forget_bench.types import ForgetRequest, Lang, MemoryStats, Message, Question, Turn


@dataclass(frozen=True)
class StrategyContext:
    """戦略に渡す共通の材料。条件を揃えるため、system_prompt と max_tokens は全戦略で共通。"""

    lang: Lang
    system_prompt: str
    llm: Backend  # メモリ構築用。回答用と同じバックエンドを、記録用ラッパー越しに渡す
    max_tokens: int


@runtime_checkable
class MemoryStrategy(Protocol):
    name: str

    def observe(self, turn: Turn) -> None: ...

    def forget(self, request: ForgetRequest) -> None: ...

    def build_context(self, question: Question) -> list[Message]: ...

    def stats(self) -> MemoryStats: ...


StrategyFactory = Callable[..., MemoryStrategy]
_REGISTRY: dict[str, StrategyFactory] = {}


def register_strategy(name: str) -> Callable[[StrategyFactory], StrategyFactory]:
    def deco(factory: StrategyFactory) -> StrategyFactory:
        if name in _REGISTRY:
            raise ValueError(f"strategy {name} は登録済み")
        _REGISTRY[name] = factory
        return factory

    return deco


def make_strategy(name: str, ctx: StrategyContext, **params: Any) -> MemoryStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"未知の戦略: {name}(登録済み: {sorted(_REGISTRY)})")
    return _REGISTRY[name](ctx, **params)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


@dataclass(frozen=True)
class ScenarioShape:
    """見積もりに使うシナリオの形(LLMを呼ばずに生成器から数える)。"""

    n_user_turns: int
    n_sessions: int
    n_forget: int


def estimate_memory_calls(name: str, shape: ScenarioShape, params: dict[str, Any]) -> int:
    """メモリ構築のLLM呼び出し数の見積もり。戦略クラスが estimate_memory_calls を持っていればそれを使う。"""
    fn = getattr(_REGISTRY[name], "estimate_memory_calls", None)
    return int(fn(shape, **params)) if callable(fn) else 0
