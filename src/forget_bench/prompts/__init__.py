"""実験用プロンプト。全バックエンド共通(docs/backends.md「揃える条件」)。"""

from __future__ import annotations

from importlib import resources

from forget_bench.types import Lang


def load_prompt(name: str, lang: Lang) -> str:
    return resources.files(__package__).joinpath(f"{name}_{lang}.md").read_text(encoding="utf-8").strip()
