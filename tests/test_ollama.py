from __future__ import annotations

import json

import httpx
import pytest

from forget_bench.backends import BackendError
from forget_bench.backends.ollama import OllamaBackend
from forget_bench.types import Message


def _backend(handler, num_ctx: int = 4096) -> OllamaBackend:  # noqa: ANN001
    return OllamaBackend("m", num_ctx, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_request_options_and_parsing() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={
            "message": {"content": "金沢"}, "prompt_eval_count": 20, "eval_count": 2,
            "prompt_eval_duration": 100_000_000, "load_duration": 0, "done_reason": "stop",
        })

    g = _backend(handler).generate([Message("user", "質問")], max_tokens=64)
    assert seen["options"] == {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 64}
    assert seen["stream"] is False
    assert g.text == "金沢" and g.input_tokens == 20 and g.output_tokens == 2
    assert g.ttft_s == pytest.approx(0.1)


def test_silent_truncation_is_an_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "x"}, "prompt_eval_count": 258, "eval_count": 1})

    with pytest.raises(BackendError, match="truncation"):
        _backend(handler, 512).generate([Message("user", "あ" * 3000)], max_tokens=8)


def test_overflow_is_an_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "x"}, "prompt_eval_count": 4000, "eval_count": 1})

    with pytest.raises(BackendError, match="overflow"):
        _backend(handler, 4096).generate([Message("user", "a" * 4000)], max_tokens=256)


def test_http_error() -> None:
    with pytest.raises(BackendError):
        _backend(lambda r: httpx.Response(500, text="boom")).generate([Message("user", "x")], max_tokens=8)


def test_num_ctx_required() -> None:
    with pytest.raises(ValueError):
        OllamaBackend("m", 0)
