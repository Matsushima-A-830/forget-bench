# 2026-11-23 第4週: 発表用の図

既存の3つの run の生ログから作った図。新しい実験は回していない。

## 作り方

```bash
uv sync --extra viz                       # matplotlib
uv run forget-bench figures               # configs/figures.yaml → results/figures/
```

- 元データ(`configs/figures.yaml`)
  - claude: `runs/20261116-073132-sweep-claude-ja1`(commit `f66d794`)。日本語 × seed 1000 × 5/20/50、繰り返し1
  - ollama: `runs/20261115-205757-sweep-ollama`(commit `7e481d1`)。`qwen2.5:7b-instruct`、日英 × seed 1000–1002 × 5/20/50
  - hf: `runs/20261121-183356-sweep-hf`(commit `69fd567`、results/2026-11-21-week3-kv.md の記録を参照)。Qwen2.5-3B bf16、日英 × seed 1000–1002 × 5/20/50
- 図に載せた数値はすべて `results/figures/figures.json` にある(点ごとの値・分母・出所の run と commit)
- 指標の定義(`src/forget_bench/figures.py`)
  - **覚える力** = current / historical / oneshot の質問をまとめた正答率。aggregate はローカルモデルでほぼ 0 なので含めない
  - **忘れる力** = 100 − 削除要求あたりの漏洩率(1つの削除要求について、その値が1回でも回答に出たら漏洩)
- `facts_append_oracle`(正解ラベルで抽出した参照値)は散布図に描かない(戦略ではないので)。数値は下の表にある

## 図

| ファイル | 内容 |
|---|---|
| `results/figures/tradeoff_all.{png,svg}` | 覚える力 × 忘れる力。バックエンドごとに1枚、各バックエンドの全シナリオ・全規模をまとめた値。色と形が層(テキスト = 青い丸、KV = オレンジのひし形) |
| `results/figures/tradeoff_common.{png,svg}` | 同じ図を、3つの run に共通するシナリオ(`dev-1000-ja-s5/s20/s50`)だけで作ったもの。バックエンド間の比較はこちら |
| `results/figures/scale_remember.{png,svg}` | セッション数(5 / 20 / 50)に対する覚える力。線は8戦略(色は戦略ごとに固定) |
| `results/figures/latency_kv.{png,svg}` | hf で、回答1回のレイテンシ(`full` と `kv_full`、対数軸) |

**分母の注意**: claude は1シナリオ × 3規模なので、1点あたり質問 43問・削除要求 6件しかない。
tradeoff_all でも claude の分母は同じ(claude の run にはそのシナリオしかない)。

## 図の数値

### 各バックエンドの全シナリオ(tradeoff_all)

| backend | strategy | 覚える力 % (質問数) | 忘れる力 % (削除要求数) | over-forget % |
|---|---|---:|---:|---:|
| claude opus-5.5 | facts_append | 95 (43) | 100 (6) | 0 |
| claude opus-5.5 | facts_append_oracle | 100 (43) | 100 (6) | 0 |
| claude opus-5.5 | full | 95 (43) | 100 (6) | 0 |
| claude opus-5.5 | full_redact | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | hybrid | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | rag | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | summary | 100 (43) | 100 (6) | 0 |
| claude opus-5.5 | window | 2 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | facts_append | 49 (268) | 100 (36) | 0 |
| ollama qwen2.5 7B | facts_append_oracle | 81 (268) | 100 (36) | 0 |
| ollama qwen2.5 7B | full | 71 (268) | 53 (36) | 3 |
| ollama qwen2.5 7B | full_redact | 74 (268) | 100 (36) | 2 |
| ollama qwen2.5 7B | hybrid | 79 (268) | 100 (36) | 0 |
| ollama qwen2.5 7B | rag | 80 (268) | 100 (36) | 1 |
| ollama qwen2.5 7B | summary | 22 (268) | 100 (36) | 0 |
| ollama qwen2.5 7B | window | 3 (268) | 100 (36) | 8 |
| hf qwen2.5 3B | facts_append | 37 (268) | 100 (36) | 4 |
| hf qwen2.5 3B | facts_append_oracle | 71 (268) | 100 (36) | 1 |
| hf qwen2.5 3B | full | 44 (268) | 42 (36) | 2 |
| hf qwen2.5 3B | full_redact | 44 (268) | 100 (36) | 2 |
| hf qwen2.5 3B | hybrid | 69 (268) | 100 (36) | 1 |
| hf qwen2.5 3B | kv_drop | 43 (268) | 100 (36) | 2 |
| hf qwen2.5 3B | kv_drop_repos | 46 (268) | 100 (36) | 2 |
| hf qwen2.5 3B | kv_evict_old | 14 (268) | 100 (36) | 16 |
| hf qwen2.5 3B | kv_full | 43 (268) | 44 (36) | 2 |
| hf qwen2.5 3B | rag | 65 (268) | 100 (36) | 3 |
| hf qwen2.5 3B | summary | 9 (268) | 100 (36) | 6 |
| hf qwen2.5 3B | window | 3 (268) | 100 (36) | 15 |

### 共通シナリオだけ(tradeoff_common)

| backend | strategy | 覚える力 % (質問数) | 忘れる力 % (削除要求数) | over-forget % |
|---|---|---:|---:|---:|
| claude opus-5.5 | facts_append | 95 (43) | 100 (6) | 0 |
| claude opus-5.5 | facts_append_oracle | 100 (43) | 100 (6) | 0 |
| claude opus-5.5 | full | 95 (43) | 100 (6) | 0 |
| claude opus-5.5 | full_redact | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | hybrid | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | rag | 98 (43) | 100 (6) | 0 |
| claude opus-5.5 | summary | 100 (43) | 100 (6) | 0 |
| claude opus-5.5 | window | 2 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | facts_append | 40 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | facts_append_oracle | 86 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | full | 70 (43) | 0 (6) | 2 |
| ollama qwen2.5 7B | full_redact | 74 (43) | 100 (6) | 2 |
| ollama qwen2.5 7B | hybrid | 74 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | rag | 81 (43) | 100 (6) | 2 |
| ollama qwen2.5 7B | summary | 14 (43) | 100 (6) | 0 |
| ollama qwen2.5 7B | window | 2 (43) | 100 (6) | 4 |
| hf qwen2.5 3B | facts_append | 49 (43) | 100 (6) | 4 |
| hf qwen2.5 3B | facts_append_oracle | 74 (43) | 100 (6) | 4 |
| hf qwen2.5 3B | full | 33 (43) | 67 (6) | 0 |
| hf qwen2.5 3B | full_redact | 33 (43) | 100 (6) | 2 |
| hf qwen2.5 3B | hybrid | 79 (43) | 100 (6) | 2 |
| hf qwen2.5 3B | kv_drop | 37 (43) | 100 (6) | 2 |
| hf qwen2.5 3B | kv_drop_repos | 44 (43) | 100 (6) | 2 |
| hf qwen2.5 3B | kv_evict_old | 7 (43) | 100 (6) | 35 |
| hf qwen2.5 3B | kv_full | 33 (43) | 67 (6) | 2 |
| hf qwen2.5 3B | rag | 65 (43) | 100 (6) | 4 |
| hf qwen2.5 3B | summary | 5 (43) | 100 (6) | 17 |
| hf qwen2.5 3B | window | 2 (43) | 100 (6) | 24 |

### 回答1回の平均レイテンシ(hf、秒。日英まとめ、latency_kv)

| strategy | 5 | 20 | 50 |
|---|---:|---:|---:|
| full | 1.16 | 4.57 | 57.55 |
| kv_full | 0.57 | 0.74 | 1.68 |

規模ごとの値は `figures.json` の `points_by_sessions`。

## 図から読めること

1. **README の仮説「追記のみメモリは忘れる力が最弱」は外れた。** `facts_append` は削除要求で墓標を立てるので、どのバックエンドでも
   忘れる力 100%。忘れる力が落ちたのは、削除要求を無視する `full`(ollama 53%、hf 42%)と `kv_full`(hf 44%)だけ。
   claude は `full` でも漏らさなかった(6件中0件)。
2. **忘れ方は層によらず、消し方の確実さで決まる。** テキストを伏字にする `full_redact`、索引から消す `rag`、KV を切り取る
   `kv_drop` / `kv_drop_repos` は、どれも忘れる力 100%。そのうえで覚える力は戦略とモデルの大きさで決まる
   (hf 3B で `hybrid` 69%・`rag` 65%、`kv_drop_repos` 46%・`full_redact` 44%)。
3. **3B では、長い文脈をそのまま渡す戦略が規模で崩れる。** 規模曲線で、hf 3B の `full` は 5 → 20 → 50 セッションで
   65 → 36 → 31%(`kv_full` も 65 → 33 → 32%)。検索で文脈を絞る `rag` は 71 → 67 → 59%、`hybrid` は 77 → 61 → 68%。
   ollama 7B の `full` は 76 → 68 → 70% でそこまで落ちない。claude は `window` 以外が 50セッションでも 88% 以上
   (ただし claude は1シナリオで、1点あたり最大16問)。
4. **KV を使い回すと、規模が大きいほど速い。** 50セッションで回答1回 57.55秒(`full`)と 1.68秒(`kv_full`)。
