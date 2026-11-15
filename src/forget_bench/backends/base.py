"""バックエンドの契約(AGENTS.md「インターフェース」)とプラグイン登録。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from forget_bench.types import Generation, Message


class BackendError(RuntimeError):
    """バックエンド呼び出しの失敗(ランナーはこれを記録して次へ進む)。"""


class BackendFatalError(RuntimeError):
    """続けても無駄な失敗(利用枠切れ・認証失敗など)。BackendError ではないので、ランナーは実行を止める。

    途中までのログとレスポンスキャッシュは残るので、--reuse-cache で続きから再実行できる。
    """


@runtime_checkable
class Backend(Protocol):
    name: str

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation: ...

    def info(self) -> dict[str, Any]:
        """meta.json に記録するモデル情報(名前・digest・量子化など)。"""
        ...


BackendFactory = Callable[..., Backend]
_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    def deco(factory: BackendFactory) -> BackendFactory:
        if name in _REGISTRY:
            raise ValueError(f"backend {name} は登録済み")
        _REGISTRY[name] = factory
        return factory

    return deco


def make_backend(name: str, **params: Any) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"未知のバックエンド: {name}(登録済み: {sorted(_REGISTRY)})")
    return _REGISTRY[name](**params)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
