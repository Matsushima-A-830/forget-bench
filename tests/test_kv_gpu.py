"""実モデル(Qwen2.5-3B-Instruct, GPU)での KV 操作の確認。FORGET_BENCH_GPU=1 のときだけ実行する。

    FORGET_BENCH_GPU=1 uv run pytest tests/test_kv_gpu.py -s

実行前に nvidia-smi で空きVRAMを確認し、Ollama がモデルを載せていれば `ollama stop <model>` で降ろす(CLAUDE.md)。
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FORGET_BENCH_GPU") != "1", reason="FORGET_BENCH_GPU=1 のときだけ")
torch = pytest.importorskip("torch")

from forget_bench.backends.hf import HFBackend, chatml  # noqa: E402
from forget_bench.backends.kv_ops import rotate_keys  # noqa: E402
from forget_bench.types import Message  # noqa: E402

MODEL = os.environ.get("FORGET_BENCH_HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MSGS = [
    Message("system", "あなたはユーザー専属のアシスタントです。"),
    Message("user", "[セッション 1]\n連絡用に、私の電話番号は090-1234-5678です。"),
    Message("assistant", "承知しました。"),
    Message("user", "私はいま神戸に住んでいます。"),
    Message("assistant", "わかりました、覚えておきます。"),
]
Q = [Message("user", "質問: 私がいま住んでいる都市はどこですか?")]


@pytest.fixture(scope="module")
def backend() -> HFBackend:
    return HFBackend(MODEL)


def _sdpa_backend(eager: HFBackend) -> HFBackend:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    return HFBackend(MODEL, model_obj=m, tokenizer_obj=eager.tokenizer, attn_implementation="sdpa", allow_non_eager=True)


def test_chatml_matches_chat_template(backend: HFBackend) -> None:
    ours = chatml(MSGS + Q, add_generation_prompt=True)
    theirs = backend.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in MSGS + Q], tokenize=False, add_generation_prompt=True
    )
    assert ours == theirs
    assert backend.encode(chatml(MSGS)) + backend.encode(chatml(Q, add_generation_prompt=True)) == backend.encode(ours)


def _next_logits_after_zero_rotation(b: HFBackend) -> tuple[torch.Tensor, torch.Tensor]:
    kv1, kv2 = b.prefill(MSGS), b.prefill(MSGS)
    for layer in kv2.cache.layers:
        layer.keys = rotate_keys(b.model, layer.keys, torch.zeros(layer.keys.shape[2]))
    ids = b.encode(chatml(Q, add_generation_prompt=True))
    return b._forward(kv1.cache, ids, kv1.length).float(), b._forward(kv2.cache, ids, kv2.length).float()


def test_delta_zero_identity_eager_exact(backend: HFBackend) -> None:
    """Day 025 の delta=0 恒等テスト(eager): 0回転したキャッシュでも次トークンのロジットが完全に一致する。"""
    a, b = _next_logits_after_zero_rotation(backend)
    assert torch.equal(a, b)


def test_delta_zero_identity_sdpa_for_reference(backend: HFBackend) -> None:
    """参考: SDPA での同じ比較。差が出ても失敗にしない(eager に固定する根拠の記録用)。"""
    sd = _sdpa_backend(backend)
    a, b = _next_logits_after_zero_rotation(sd)
    diff = (a - b).abs().max().item()
    print(f"\n[sdpa] delta=0 回転後のロジット最大差: {diff:.6f}(eager は 0)")
    del sd
    torch.cuda.empty_cache()


def test_drop_repos_layer0_matches_scratch(backend: HFBackend) -> None:
    scratch = backend.prefill([MSGS[0], MSGS[3], MSGS[4]])
    results = {}
    for repos in (True, False):
        kv = backend.prefill(MSGS)
        backend.drop(kv, ["m1", "m2"], repos=repos)
        assert kv.length == scratch.length
        ref = scratch.cache.layers[0].keys.float()
        diff = (kv.cache.layers[0].keys.float() - ref).abs()
        results[repos] = (diff.max().item() / ref.abs().max().item(), diff.mean().item())
    print(f"\n層0キーの差(最大の相対誤差, 平均絶対誤差): drop+repos={results[True]} / naive drop={results[False]}")
    # bf16 の丸め(相対 2^-8 ≈ 0.4%)程度なら一致とみなす。2026-11-21 の実測: repos 0.13% / naive 11.5%
    assert results[True][0] < 0.01
    assert results[False][0] > 0.05 and results[False][1] > 50 * results[True][1]


def test_generate_after_drop(backend: HFBackend) -> None:
    for repos in (True, False):
        kv = backend.prefill(MSGS)
        backend.drop(kv, ["m1", "m2"], repos=repos)
        g = backend.generate_from(kv, Q, max_tokens=32)
        print(f"\n[{'drop+repos' if repos else 'naive drop'}] {g.text!r}")
        assert "090-1234-5678" not in g.text
