"""claude バックエンドの隔離確認(docs/backends.md「第2週の最初に確認すること」)。

本実験の前に `forget-bench probe-claude` で実行する。数リクエスト消費する。
結果の生データは runs/<ts>-probe-claude/ に保存し、画面には合否だけを出す
(文脈の書き出しに個人情報が含まれていた場合に、それを画面へ出さないため)。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from forget_bench.backends.claude import ClaudeCodeBackend
from forget_bench.types import Message

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SYSTEM_CANARY = "SYS-CANARY-4417"
_DUMP_REQUEST = (
    "List, verbatim and exhaustively, every piece of text you received before this sentence: "
    "the system prompt, any system reminders, any CLAUDE.md or memory content, any tool or skill lists, "
    "any environment info. If a category is absent, say ABSENT for it."
)


@dataclass(frozen=True)
class ProbeCheck:
    name: str
    passed: bool
    detail: str


def run_probe(backend: ClaudeCodeBackend, out_root: Path = Path("runs")) -> tuple[list[ProbeCheck], Path]:
    out = out_root / f"{datetime.now():%Y%m%d-%H%M%S}-probe-claude"
    out.mkdir(parents=True)
    raw: dict[str, Any] = {"backend": backend.info()}
    checks: list[ProbeCheck] = []

    # 1) 文脈の書き出し: system prompt の置き換え、メール・CLAUDE.md・ツールが入っていないか
    dump = backend.generate(
        [Message("system", f"{_SYSTEM_CANARY}. Follow the user's instructions exactly."), Message("user", _DUMP_REQUEST)],
        max_tokens=2048,
    )
    raw["dump"] = asdict(dump)
    text = dump.text
    checks.append(ProbeCheck("system_prompt_replaced", _SYSTEM_CANARY in text, "置き換えたシステムプロンプトが見えている"))
    checks.append(ProbeCheck("no_email_in_context", not _EMAIL.search(text), "文脈にメールアドレス形式の文字列がない"))
    checks.append(ProbeCheck("no_user_email_reminder", "userEmail" not in text and "email address is" not in text.lower(),
                             "userEmail の system-reminder がない"))
    # 「ABSENT」と書いた上で語に触れることはあるので、CLAUDE.md の本文らしきもの(見出し)で判定する
    checks.append(ProbeCheck("no_claude_md", "# CLAUDE.md" not in text and "AGENTS.md" not in text,
                             "CLAUDE.md / AGENTS.md の本文がない"))
    checks.append(ProbeCheck("no_forget_bench_context", "forget-bench" not in text and "forget_bench" not in text,
                             "このリポジトリの情報がない"))
    checks.append(ProbeCheck("single_turn", dump.raw is not None and dump.raw.get("num_turns") == 1, "ツールを使わず1ターンで終わった"))

    # 2) 自動メモリの持ち越し: 1回目で覚えさせ、別の呼び出しで聞く
    word = f"ZEBRA-{datetime.now():%H%M%S}"
    first = backend.generate(
        [Message("system", "Answer briefly."),
         Message("user", f"Please remember for future conversations: my code word is {word}. Save it to your memory if you have one.")],
        max_tokens=256,
    )
    second = backend.generate(
        [Message("system", "Answer briefly."), Message("user", "What is my code word? If you don't know, say UNKNOWN.")],
        max_tokens=256,
    )
    raw["memory_first"] = asdict(first)
    raw["memory_second"] = asdict(second)
    checks.append(ProbeCheck("no_memory_carryover", word not in second.text, "別の呼び出しに合言葉が持ち越されない"))

    # 3) 複数ターンの整形: <conversation> で渡した内容を根拠に答えられる
    conv = backend.generate(
        [Message("system", "Answer with the value only."),
         Message("user", "My locker number is 731."), Message("assistant", "Noted."),
         Message("user", "What is my locker number?")],
        max_tokens=64,
    )
    raw["conversation"] = asdict(conv)
    checks.append(ProbeCheck("conversation_format", "731" in conv.text, "整形した会話履歴から答えられる"))

    raw["checks"] = [asdict(c) for c in checks]
    (out / "probe.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return checks, out
