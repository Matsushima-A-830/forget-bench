from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from forget_bench.backends import BackendError, BackendFatalError
from forget_bench.backends.claude import TOKEN_ENV, ClaudeCodeBackend, read_env_file, render_prompt
from forget_bench.types import Message

# 偽の claude: 呼ばれ方を記録し、FAKE_PLAN の順に応答を返す
FAKE = r'''#!{python}
import json, os, sys
log = os.environ["FAKE_LOG"]
calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
plan = json.loads(os.environ["FAKE_PLAN"])
step = plan[min(len(calls), len(plan) - 1)]
rec = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
       "cwd_entries": os.listdir("."), "env": {{k: os.environ.get(k) for k in
       ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_DISABLE_AUTO_MEMORY"]}}}}
open(log, "a").write(json.dumps(rec) + "\n")
if step == "version":
    print("9.9.9 (Claude Code)"); sys.exit(0)
print(json.dumps(step))
'''

OK = {"type": "result", "subtype": "success", "is_error": False, "result": "金沢", "num_turns": 1,
      "usage": {"input_tokens": 3, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 7},
      "modelUsage": {"claude-opus-5-5": {}}, "ttft_ms": 1500, "duration_ms": 2000, "api_error_status": None}


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-pass")
    monkeypatch.setenv(TOKEN_ENV, "tok-123")

    def setup(plan: list) -> tuple[ClaudeCodeBackend, Path]:
        monkeypatch.setenv("FAKE_PLAN", json.dumps(plan))
        return ClaudeCodeBackend("claude-opus-5-5", effort="low", claude_bin=str(exe), retry_wait_s=0), log

    return setup


def _calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_isolation_flags_env_and_stdin(fake) -> None:  # noqa: ANN001
    backend, log = fake([OK])
    msgs = [Message("system", "SYS"), Message("user", "u1"), Message("assistant", "a1"), Message("user", "Q?")]
    g = backend.generate(msgs, max_tokens=256)
    backend.generate(msgs, max_tokens=256)
    c1, c2 = _calls(log)
    argv = c1["argv"]
    assert argv[:3] == ["-p", "--system-prompt", "SYS"]
    for flag in ("--strict-mcp-config", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "low"
    assert "Q?" not in argv and c1["stdin"].endswith("Q?")  # プロンプトは標準入力
    assert c1["cwd_entries"] == [] and c1["cwd"] != c2["cwd"]  # 毎回、空の別ディレクトリ
    assert "forget-bench" not in c1["cwd"]
    env = c1["env"]
    assert env["ANTHROPIC_API_KEY"] is None
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-123"
    assert env["CLAUDE_CONFIG_DIR"] and os.path.isdir(env["CLAUDE_CONFIG_DIR"])
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert g.text == "金沢" and g.input_tokens == 153 and g.output_tokens == 7 and g.ttft_s == 1.5


def test_retry_then_success(fake) -> None:  # noqa: ANN001
    err = dict(OK, subtype="error_during_execution", is_error=True, api_error_status=529, result="overloaded")
    backend, log = fake([err, OK])
    assert backend.generate([Message("user", "x")], max_tokens=8).text == "金沢"
    assert len(_calls(log)) == 2


def test_persistent_error_is_backend_error(fake) -> None:  # noqa: ANN001
    err = dict(OK, subtype="error_during_execution", is_error=True, api_error_status=500, result="boom")
    backend, log = fake([err])
    with pytest.raises(BackendError):
        backend.generate([Message("user", "x")], max_tokens=8)
    assert len(_calls(log)) == 3  # 1回 + リトライ2回


def test_auth_error_is_fatal(fake) -> None:  # noqa: ANN001
    backend, _ = fake([dict(OK, is_error=True, subtype="error", api_error_status=401, result="unauthorized")])
    with pytest.raises(BackendFatalError):
        backend.generate([Message("user", "x")], max_tokens=8)


def test_missing_token_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(BackendFatalError, match="setup-token"):
        ClaudeCodeBackend("m", env_file=str(tmp_path / "missing.env"))


def test_token_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comment\nexport {TOKEN_ENV}='abc'\nOTHER=1\n")
    assert read_env_file(env)[TOKEN_ENV] == "abc"
    b = ClaudeCodeBackend("m", env_file=str(env))
    assert b._env[TOKEN_ENV] == "abc"


def test_render_prompt() -> None:
    system, prompt = render_prompt([Message("system", "S"), Message("user", "only")])
    assert (system, prompt) == ("S", "only")
    _, prompt = render_prompt([Message("system", "S"), Message("user", "u"), Message("assistant", "a"), Message("user", "Q")])
    assert prompt == "<conversation>\n<user>\nu\n</user>\n<assistant>\na\n</assistant>\n</conversation>\n\nQ"
