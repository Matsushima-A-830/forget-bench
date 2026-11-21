"""Claude Code ヘッドレス実行(`claude -p`)バックエンド。被験者としての Claude を呼ぶ。

隔離(CLAUDE.md「被験者としてのClaude」、docs/backends.md):
- 1呼び出しごとに **空の一時ディレクトリ** を作業ディレクトリにする(CLAUDE.md を自動で読ませない)
- `--system-prompt` で実験用プロンプトに置き換え、`--tools ""` でツール無効、`--strict-mcp-config` でMCP無効、
  `--no-session-persistence`、`--output-format json`
- 環境変数 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`(自動メモリ無効。念のため)
- auth="token"(既定): `claude setup-token` で作った `CLAUDE_CODE_OAUTH_TOKEN` と、空の `CLAUDE_CONFIG_DIR` で呼ぶ。
  ログイン中アカウントの設定・プラグイン・スキル・メールアドレス(OAuthプロフィール由来で文脈に注入される)を
  被験者に渡さないため(2026-11-15 に、通常ログインでは userEmail が system-reminder として入ることを確認)。
- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` は子プロセスに渡さない(APIキー課金に切り替わらないように)。
- プロンプトは標準入力で渡す(長い文脈が引数長の上限に当たらないように)。

ローカル側と揃えられない差(docs/backends.md に記載):
- 温度・seed は指定できない。同じ入力を repeats 回送って揺れを測る。
- 最大出力トークンは指定しない(CLI に指定がなく、思考トークンも出力上限に数えられるため)。出力トークン数は記録する。
- chat の役割分けはできないので、system 以外の messages を `<conversation>` タグ付きの1文字列に整形する。
- `--system-prompt` の先頭に Claude Code が1文を足し、作業ディレクトリ・日付・モデル名の環境情報も入る。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from forget_bench.backends.base import BackendError, BackendFatalError, register_backend
from forget_bench.types import Generation, Message

TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CONFIG_DIR", TOKEN_ENV)


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE 形式の .env を読む(値は表示・記録しない)。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


def render_prompt(messages: list[Message]) -> tuple[str, str]:
    """messages を (system, prompt) に整形する。

    最後のメッセージ(質問)はそのまま末尾に置き、それより前のやり取りを <conversation> で囲む。
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [m for m in messages if m.role != "system"]
    if not rest:
        raise ValueError("system 以外のメッセージがない")
    *history, last = rest
    if not history:
        return system, last.content
    lines = ["<conversation>"]
    for m in history:
        lines.append(f"<{m.role}>\n{m.content}\n</{m.role}>")
    lines.append("</conversation>")
    return system, "\n".join(lines) + "\n\n" + last.content


@register_backend("claude")
class ClaudeCodeBackend:
    name = "claude"

    def __init__(
        self,
        model: str,
        *,
        effort: str = "medium",
        auth: str = "token",
        env_file: str = ".env",
        claude_bin: str = "claude",
        timeout_s: float = 600.0,
        max_retries: int = 2,
        retry_wait_s: float = 30.0,
        **_: Any,
    ) -> None:
        if auth not in ("token", "login"):
            raise ValueError("auth は token か login")
        self.model = model
        self.effort = effort
        self.auth = auth
        self.claude_bin = claude_bin
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self._env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        self._env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        self._config_dir: str | None = None
        if auth == "token":
            token = os.environ.get(TOKEN_ENV) or read_env_file(Path(env_file)).get(TOKEN_ENV)
            if not token:
                raise BackendFatalError(
                    f"{TOKEN_ENV} が環境変数にも {env_file} にもない。別の端末で `claude setup-token` を実行し、"
                    f"{env_file} に {TOKEN_ENV}=... を書く(git管理外・chmod 600)"
                )
            self._config_dir = tempfile.mkdtemp(prefix="cc-config-")
            self._env[TOKEN_ENV] = token
            self._env["CLAUDE_CONFIG_DIR"] = self._config_dir

    def __del__(self) -> None:
        if self._config_dir:
            shutil.rmtree(self._config_dir, ignore_errors=True)

    def command(self, system: str) -> list[str]:
        return [
            self.claude_bin, "-p",
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--output-format", "json",
            "--model", self.model,
            "--effort", self.effort,
        ]

    def _call(self, system: str, prompt: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="subject-") as cwd:
            try:
                proc = subprocess.run(
                    self.command(system), input=prompt, capture_output=True, text=True,
                    cwd=cwd, env=self._env, timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as e:
                raise BackendError(f"claude -p timed out after {self.timeout_s}s") from e
            except OSError as e:
                raise BackendFatalError(f"claude を起動できない: {e}") from e
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise BackendError(
                f"claude -p の出力がJSONでない (exit {proc.returncode}): {proc.stderr.strip()[-300:]}"
            ) from None
        if not isinstance(data, dict):
            raise BackendError("claude -p の出力が想定外の形式")
        return data

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        system, prompt = render_prompt(messages)
        last_error = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            data = self._call(system, prompt)
            latency = time.perf_counter() - t0
            status = data.get("api_error_status")
            if not data.get("is_error") and data.get("subtype") == "success":
                return self._to_generation(data, latency)
            last_error = f"subtype={data.get('subtype')} api_error_status={status} result={str(data.get('result'))[:200]}"
            if status in (401, 403):
                raise BackendFatalError(f"claude 認証エラー: {last_error}")
            retryable = status in (429, 500, 502, 503, 504, 529) or status is None
            if not retryable or attempt == self.max_retries:
                break
            time.sleep(self.retry_wait_s * (attempt + 1))
        if "limit" in last_error.lower() and "429" in last_error:
            raise BackendFatalError(f"claude の利用枠に達した可能性: {last_error}")
        raise BackendError(f"claude -p failed: {last_error}")

    @staticmethod
    def _to_generation(data: dict[str, Any], latency: float) -> Generation:
        usage = data.get("usage") or {}
        input_tokens = sum(
            int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        ttft_ms = data.get("ttft_ms")
        models = sorted((data.get("modelUsage") or {}).keys())
        raw = {
            "duration_ms": data.get("duration_ms"),
            "duration_api_ms": data.get("duration_api_ms"),
            "ttft_ms": ttft_ms,
            "stop_reason": data.get("stop_reason"),
            "total_cost_usd": data.get("total_cost_usd"),
            "usage": {k: usage.get(k) for k in (
                "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens",
            )},
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            "models": models,
            "num_turns": data.get("num_turns"),
        }
        warnings = []
        if data.get("num_turns") not in (None, 1):
            warnings.append(f"num_turns={data.get('num_turns')}")
        return Generation(
            text=str(data.get("result") or ""),
            input_tokens=input_tokens if usage else None,
            output_tokens=usage.get("output_tokens"),
            latency_s=latency,
            ttft_s=ttft_ms / 1000 if isinstance(ttft_ms, (int, float)) else None,
            raw=raw,
            warnings=warnings,
        )

    def info(self) -> dict[str, Any]:
        version = None
        try:
            version = subprocess.run([self.claude_bin, "--version"], capture_output=True, text=True,
                                     timeout=30, env=self._env).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return {
            "backend": "claude",
            "model": self.model,
            "effort": self.effort,
            "auth": self.auth,
            "isolated_config_dir": self._config_dir is not None,
            "claude_code_version": version,
            "command": self.command("<system>"),
            "env_flags": {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
            "temperature": "not supported (repeats で揺れを測る)",
            "max_tokens": "not enforced",
        }
