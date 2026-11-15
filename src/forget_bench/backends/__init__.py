"""バックエンドのプラグイン。import するとレジストリに登録される。"""

from forget_bench.backends import claude, dummy, ollama  # noqa: F401  (登録のための import)
from forget_bench.backends.base import Backend, BackendError, BackendFatalError, available_backends, make_backend
from forget_bench.backends.metered import MeteredBackend

__all__ = ["Backend", "BackendError", "BackendFatalError", "MeteredBackend", "available_backends", "make_backend"]
