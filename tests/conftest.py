from __future__ import annotations

import pytest

from forget_bench.types import ScenarioParams


def make_params(lang: str = "ja", **kw: object) -> ScenarioParams:
    base: dict[str, object] = dict(
        lang=lang, sessions=5, turns_per_session=(10, 20), n_attributes=4, max_changes=4,
        n_oneshot=3, n_sensitive=3, n_forget=2, aggregate_mentions=4,
    )
    base.update(kw)
    return ScenarioParams(**base)  # type: ignore[arg-type]


@pytest.fixture
def params() -> ScenarioParams:
    return make_params()
