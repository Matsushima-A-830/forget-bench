from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from forget_bench.backends.hf import chatml  # noqa: E402
from forget_bench.backends.kv_ops import drop_tokens, rotate_keys  # noqa: E402
from forget_bench.types import Message  # noqa: E402
from tests.kv_fixtures import tiny_backend  # noqa: E402

SYS = Message("system", "You are a test.")
T1 = Message("user", "My phone is 090-1234-5678.")
T2 = Message("assistant", "Noted.")
T3 = Message("user", "I live in Kobe.")
Q = [Message("user", "Where do I live?")]


def _layer0_keys(kv) -> torch.Tensor:  # noqa: ANN001
    return kv.cache.layers[0].keys.clone()


def test_rotate_by_zero_is_bit_identical() -> None:
    b = tiny_backend()
    kv = b.prefill([SYS, T1, T2, T3])
    for layer in kv.cache.layers:
        k = layer.keys
        assert torch.equal(rotate_keys(b.model, k, torch.zeros(k.shape[2])), k)


def test_rotate_composes() -> None:
    b = tiny_backend()
    kv = b.prefill([SYS, T1])
    k = kv.cache.layers[1].keys
    n = k.shape[2]
    back = rotate_keys(b.model, rotate_keys(b.model, k, torch.full((n,), 37.0)), torch.full((n,), -37.0))
    assert torch.allclose(back, k, atol=1e-5)


def test_zero_rotation_keeps_logits_exactly_under_eager() -> None:
    """Day 025 の delta=0 恒等テスト: eager では全キーを0回転しても次トークンのロジットが完全に一致する。"""
    b = tiny_backend()
    kv1 = b.prefill([SYS, T1, T2, T3])
    kv2 = b.prefill([SYS, T1, T2, T3])
    for layer in kv2.cache.layers:
        layer.keys = rotate_keys(b.model, layer.keys, torch.zeros(layer.keys.shape[2]))
    ids = b.encode(chatml(Q, add_generation_prompt=True))
    l1 = b._forward(kv1.cache, ids, kv1.length)
    l2 = b._forward(kv2.cache, ids, kv2.length)
    assert torch.equal(l1, l2)


def test_drop_repos_matches_from_scratch_layer0_but_naive_does_not() -> None:
    """層0のキーは文脈に依存しない(埋め込み→線形→RoPE)ので、drop+repos 後は「最初からその範囲が無かった」キャッシュと一致する。"""
    b = tiny_backend()
    scratch = b.prefill([SYS, T2, T3])  # T1 が最初から無い
    for repos, should_match in ((True, True), (False, False)):
        kv = b.prefill([SYS, T1, T2, T3])
        b.drop(kv, ["m1"], repos=repos)
        assert kv.length == scratch.length
        same = torch.allclose(_layer0_keys(kv), _layer0_keys(scratch), atol=1e-5)
        assert same is should_match, f"repos={repos}"
        # 値(value)は回転しないので、どちらでも層0は一致する
        assert torch.allclose(kv.cache.layers[0].values, scratch.cache.layers[0].values, atol=1e-5)


def test_drop_multiple_spans_and_bookkeeping() -> None:
    b = tiny_backend()
    kv = b.prefill([SYS, T1, T2, T3, T1])
    before = kv.length
    lens = {s.id: s.length for s in kv.spans}
    b.drop(kv, ["m1", "m4"], repos=True)
    assert [s.id for s in kv.spans] == ["m0", "m2", "m3"]
    assert kv.length == before - lens["m1"] - lens["m4"] == kv.cache.get_seq_length()
    assert kv.dropped_tokens == lens["m1"] + lens["m4"]
    starts = [s.start for s in kv.spans]
    assert starts == [0, lens["m0"], lens["m0"] + lens["m2"]]
    # 存在しない範囲は何もしない
    b.drop(kv, ["nope"], repos=True)
    assert kv.length == before - lens["m1"] - lens["m4"]


def test_generate_from_restores_cache_and_is_deterministic() -> None:
    b = tiny_backend()
    kv = b.prefill([SYS, T1, T2, T3])
    n = kv.length
    g1 = b.generate_from(kv, Q, max_tokens=5)
    assert kv.length == n == kv.cache.get_seq_length()
    g2 = b.generate_from(kv, Q, max_tokens=5)
    assert g1.text == g2.text and g1.output_tokens == g2.output_tokens
    assert g1.raw["context_tokens"] == n + g1.input_tokens


def test_incremental_kv_matches_text_prefill() -> None:
    """発話ごとに追記した KV と、全体を一度に(チャンク単位で)prefill した場合の次トークンのロジットが一致する。"""
    b = tiny_backend(prefill_chunk=7)
    msgs = [SYS, T1, T2, T3]
    kv = b.prefill(msgs)
    ids = b.encode(chatml(Q, add_generation_prompt=True))
    inc = b._forward(kv.cache, ids, kv.length)
    whole = b.encode(chatml(msgs + Q, add_generation_prompt=True))
    assert whole == b.encode(chatml(msgs)) + ids  # 整形を分けても同じトークン列
    one = b._forward(b.new_cache(), whole, 0)
    assert torch.allclose(inc, one, atol=1e-4)


def test_non_eager_requires_opt_in() -> None:
    from forget_bench.backends.hf import HFBackend
    from tests.kv_fixtures import ByteTokenizer, tiny_model

    with pytest.raises(ValueError):
        HFBackend("x", model_obj=tiny_model(), tokenizer_obj=ByteTokenizer(), device="cpu", attn_implementation="sdpa")
