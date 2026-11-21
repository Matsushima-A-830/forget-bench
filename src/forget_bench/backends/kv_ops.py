"""KVキャッシュの直接操作: drop(トークン範囲の切除)と repos(残したキーの RoPE 再回転)。

出典: ai-100days リポジトリ `experiments/day-025/akv_poc_gpu.py`(commit 59983b5、同じ作者の100日チャレンジ。
リポジトリにライセンスファイルはない)の `slice_out_span()` / `rotate_keys_by_delta()` を移植した。
Day 025 自体は論文 AKV(https://huggingface.co/papers/2610.28577、公式コード https://github.com/thu-nics/AKV, Apache-2.0)
のアイデアを、公式コードを使わずに自作実装したもの。

移植での変更点:
- 1つの範囲だけでなく、複数の範囲をまとめて切除する(残すトークンの添字で index_select)
- repos の回転量をトークンごとに変える(各トークンより前で切除したトークン数だけ巻き戻す)
- 回転は float32 で計算してから元の dtype に戻す(bf16 のまま掛けるより丸め誤差が小さい。delta=0 は元のキーとビット単位で一致)

RoPE は2次元回転の合成則 R(a)R(b) = R(a+b) が成り立つので、位置 p で回転済みのキーに R(-n) を掛ければ
位置 p-n のキーになる(作り直さずに付け替えられる)。cos/sin はモデル自身の rotary_emb で計算する(θ を独自実装しない)。

torch が必要(hf extra)。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen2.modeling_qwen2 import rotate_half


def _check_rope(model: Any) -> Any:
    rotary = model.model.rotary_emb
    scaling = float(getattr(rotary, "attention_scaling", 1.0))
    if scaling != 1.0:
        # cos/sin に attention_scaling が掛かっているモデルでは、追加回転で二重に掛かってしまう
        raise NotImplementedError(f"attention_scaling={scaling} の RoPE には repos を適用できない")
    return rotary


@torch.no_grad()
def rotate_keys(model: Any, keys: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """keys (batch, kv_heads, n, head_dim) の各トークンを deltas (n,) だけ追加回転する。"""
    rotary = _check_rope(model)
    n = keys.shape[2]
    if deltas.shape != (n,):
        raise ValueError("deltas の長さがキーのトークン数と合わない")
    ref = torch.zeros(1, n, keys.shape[-1], device=keys.device, dtype=torch.float32)
    cos, sin = rotary(ref, deltas.to(device=keys.device, dtype=torch.float32)[None, :])
    cos = cos.to(torch.float32).unsqueeze(1)
    sin = sin.to(torch.float32).unsqueeze(1)
    k = keys.to(torch.float32)
    return (k * cos + rotate_half(k) * sin).to(keys.dtype)


@torch.no_grad()
def drop_tokens(model: Any, cache: Any, keep: torch.Tensor, repos: bool) -> None:
    """cache(DynamicCache)から keep(bool, 長さ=キャッシュ長)が False のトークンを切除する(その場で書き換える)。

    repos=True なら、残したキーを「そのトークンより前で切除した数」だけ巻き戻して、詰めた後の連番位置に合わせる。
    repos=False(naive drop)では回転はそのまま(焼き込まれた位置と、詰めた後の位置が食い違う)。
    """
    if keep.dtype != torch.bool or keep.ndim != 1:
        raise ValueError("keep は1次元の bool テンソル")
    if bool(keep.all()):
        return
    device = cache.layers[0].keys.device
    keep = keep.to(device)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    shift = None
    first = 0
    if repos:
        dropped_before = torch.cumsum((~keep).to(torch.int64), dim=0)[idx]
        nz = torch.nonzero(dropped_before, as_tuple=False)
        if nz.numel():
            first = int(nz[0])
            shift = -dropped_before[first:]
    for layer in cache.layers:
        k = layer.keys.index_select(2, idx)
        v = layer.values.index_select(2, idx)
        if shift is not None:
            k = torch.cat([k[:, :, :first, :], rotate_keys(model, k[:, :, first:, :], shift)], dim=2)
        layer.keys = k
        layer.values = v
