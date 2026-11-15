"""文脈の整形。全戦略・全バックエンドで同じ規則を使う(バックエンド間で条件を揃えるため)。"""

from __future__ import annotations

from collections.abc import Iterable

from forget_bench.types import Lang, Message, Question, Turn

_SESSION_HEADER = {"ja": "[セッション {n}]", "en": "[Session {n}]"}
_QUESTION = {"ja": "質問: {q}", "en": "Question: {q}"}


def session_header(session: int, lang: Lang) -> str:
    return _SESSION_HEADER[lang].format(n=session)


def render_turns(turns: Iterable[Turn], lang: Lang) -> list[Message]:
    """ターン列を chat messages にする。セッションの境目ではユーザー発話の先頭に見出しを付ける。"""
    out: list[Message] = []
    current: int | None = None
    for t in turns:
        text = t.text
        if t.role == "user" and t.session != current:
            text = f"{session_header(t.session, lang)}\n{text}"
            current = t.session
        out.append(Message(t.role, text))
    return out


def question_text(question: Question, lang: Lang) -> str:
    return _QUESTION[lang].format(q=question.text)
