"""バックエンドのプラグイン。import するとレジストリに登録される。"""

from forget_bench.backends import dummy, ollama  # noqa: F401  (登録のための import)
from forget_bench.backends.base import Backend, BackendError, available_backends, make_backend
from forget_bench.backends.metered import MeteredBackend

__all__ = ["Backend", "BackendError", "MeteredBackend", "available_backends", "make_backend"]
