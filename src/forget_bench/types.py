"""パッケージ全体で受け渡すデータ型。

dict の生受け渡しを避けるため、ここに dataclass として集約する(AGENTS.md)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
Lang = Literal["ja", "en"]
Split = Literal["dev", "test"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


# --- シナリオ ------------------------------------------------------------------

TurnKind = Literal["noise", "attribute", "oneshot", "sensitive", "forget", "aggregate", "reply"]


@dataclass(frozen=True)
class Annotation:
    """生成器が持つ正解ラベル。

    採点・テスト・`facts_append(extractor="oracle")` 以外のメモリ戦略は読んではいけない
    (読むと「生成器の正解を見て答える」ことになる)。
    """

    fact_key: str  # 例: "city", "phone", "aggregate"
    label: str  # 人間向けの項目名(シナリオの言語)。例: "住んでいる都市"
    value: str  # 正規の表記


@dataclass(frozen=True)
class Turn:
    id: str  # 例: "t012u"(ユーザー)/ "t012a"(アシスタント)
    session: int  # 1始まり
    index: int  # ユーザー発話の通し番号(0始まり)。ユーザーとその応答で同じ値
    role: Literal["user", "assistant"]
    text: str
    kind: TurnKind
    annotation: Annotation | None = None


@dataclass(frozen=True)
class ForgetRequest:
    """削除要求の明示シグナル。シナリオ内の「忘れて」発話とは別に戦略へ渡す。"""

    id: str
    fact_key: str
    label: str
    value: str
    aliases: tuple[str, ...]  # 漏洩判定に使う表記揺れ(value を含む)
    source_turn_ids: tuple[str, ...]  # その値が言及されたターン
    request_turn_id: str  # 「忘れて」発話のターン


QuestionCategory = Literal[
    "current", "historical", "oneshot", "aggregate", "leak_direct", "leak_indirect"
]
LEAK_CATEGORIES: frozenset[str] = frozenset({"leak_direct", "leak_indirect"})


@dataclass(frozen=True)
class Question:
    id: str
    category: QuestionCategory
    text: str
    fact_key: str | None
    gold: tuple[str, ...]  # 正解の表記揺れ(どれかを含めば正解)。leak系は空
    distractors: tuple[tuple[str, ...], ...] = ()  # 同じ属性の別の値。含んだら不正解


@dataclass(frozen=True)
class ScenarioParams:
    lang: Lang
    sessions: int
    turns_per_session: tuple[int, int]  # ユーザー発話数の [min, max]
    n_attributes: int  # 変化する属性の数(最大4)
    max_changes: int  # 1属性あたりの値の数の上限(2〜4)
    n_oneshot: int  # 一度だけの細部(最大5)
    n_sensitive: int  # 個人情報の数(最大3)
    n_forget: int  # そのうち削除要求する数
    aggregate_mentions: int  # 合計させる数値の言及回数(0で無効)


@dataclass(frozen=True)
class Scenario:
    id: str
    seed: int
    split: Split
    params: ScenarioParams
    turns: tuple[Turn, ...]  # 時系列順(ユーザー/アシスタント交互)
    forget_requests: tuple[ForgetRequest, ...]
    questions: tuple[Question, ...]  # すべて最後のターンの後に問う

    def forget_request_for_turn(self, turn_id: str) -> ForgetRequest | None:
        for req in self.forget_requests:
            if req.request_turn_id == turn_id:
                return req
        return None


# --- バックエンド ----------------------------------------------------------------


@dataclass
class Generation:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    ttft_s: float | None
    raw: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


# --- メモリ ---------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStats:
    items: int  # 保持しているターン数/レコード数
    chars: int  # 保持している本文の文字数(トークン数の代わり。バックエンド非依存)
    deleted_items: int = 0
    llm_calls: int = 0  # メモリ構築で使ったLLM呼び出し回数
    errors: int = 0  # メモリ構築中の失敗(抽出のパース失敗など)
    # KV 層(hf)のみ
    kv_tokens: int = 0  # 最後に KV キャッシュに残っているトークン数
    prefill_tokens: int = 0  # メモリ構築で prefill したトークン数(コスト)
    prefill_s: float = 0.0
    dropped_tokens: int = 0  # 切除したトークン数
    peak_vram_mb: float = 0.0  # 回答生成中のピーク VRAM(torch.cuda.max_memory_allocated)
