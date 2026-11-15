# LLMバックエンド

3つのバックエンドは共通の `Backend.generate()` 契約(AGENTS.md)を満たす。KV層の実験は `hf` だけが `KVBackend` を実装する。

## 揃える条件

| 項目 | 方針 |
|---|---|
| システムプロンプト | 全バックエンド共通の実験用プロンプト(`prompts/system_*.md`、第1週で作成) |
| 温度 | 0(指定できない場合はその旨を結果に明記) |
| 最大出力トークン | 共通値(初期案 256) |
| 回答形式 | 「答えだけを1行で。削除済みの情報は『削除済み』と答える」 |
| 文脈の渡し方 | メモリ戦略が組み立てた `messages` をそのまま渡す。バックエンド側で足し引きしない |

## `claude` — Claude Code ヘッドレス実行

```bash
cd "$(mktemp -d)" && claude -p \
  --system-prompt "$SYSTEM_PROMPT" \
  --tools "" \
  --strict-mcp-config \
  --no-session-persistence \
  --output-format json \
  --model <model> \
  "<組み立てた文脈+質問>"
```

- 認証はサブスクリプション(OAuth)。`--bare` は `ANTHROPIC_API_KEY` 限定なので使わない。
- 複数ターンの `messages` は1つのプロンプト文字列に整形して渡す(整形ルールはローカル側と同じにする)。
- `--output-format json` の出力からトークン使用量・所要時間を取る(実際のフィールド名は第2週に実物で確認)。
- 温度は指定できない前提で扱い、同じ入力で3回呼んで揺れを測る(揺れ自体を結果に記録する)。

### 第2週の最初に確認すること(未確認)

- [ ] 空の一時ディレクトリから呼んだとき、ユーザー側の `~/.claude/CLAUDE.md` やメモリが文脈に入らないか。
      入る場合は `--setting-sources` 等で除外できるか
- [ ] Claude Code の自動メモリが、試行をまたいで情報を持ち越さないか(削除要求の漏洩判定に直結する)
- [ ] `--system-prompt` 置き換え時に、どの程度Claude Code固有の振る舞いが残るか
- [ ] 1リクエストあたりの所要時間と、利用枠に対する消費感(100リクエスト単位)

## `ollama` — ローカル Ollama

- HTTP API(`http://localhost:11434/api/chat`)を使う。`options.temperature=0`, `options.seed` 固定, `options.num_ctx` を明示。
- **`num_ctx` を明示しないと既定値で文脈が切り詰められ、`full` 戦略が実質 `window` になる。** 実行前に必ず設定値とプロンプト長を照合する。
  - 確認済み(Ollama 0.35.1, 2026-11-15): `num_ctx` を超えた入力は**エラーにならず黙って切り詰められる**(3236トークンの入力が `num_ctx=512` で `prompt_eval_count=258`)。
    バックエンドは「`prompt_eval_count` が入力文字数/4 の9割未満」または「入力+`max_tokens` > `num_ctx`」のときエラーにする(`backends/ollama.py`)。
  - 確認済み(同上): プレフィックスキャッシュが効いても `prompt_eval_count` は全入力トークン数を返す。短くなるのは `prompt_eval_duration` だけ。
  - 最初の呼び出しはモデルのロード(約10秒)を含むので、ランナーは本番前に1回 warmup してロード時間をTTFTに混ぜない。
- 返り値の `prompt_eval_count` / `eval_count` / `prompt_eval_duration` をトークン数・TTFTの近似に使う。
- 使うモデル(このPCにpull済み): `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`。`qwen2.5:32b-instruct` は16GBに載り切らずCPUオフロードになるので、参考値扱い。
- `meta.json` には `ollama show` で得たモデルのdigestと量子化レベルを記録する。

## `hf` — Hugging Face transformers(KV層)

- モデル: `Qwen/Qwen2.5-3B-Instruct`(bf16, 約6GB)を主、`Qwen/Qwen2.5-7B-Instruct` の4bit(bitsandbytes)を副。
- **`attn_implementation="eager"` に固定する。** Day 025 で、SDPA/flash系カーネルでは
  値が同一のKVでもメモリレイアウトの違いで損失が変わる(恒等変換のはずの delta=0 回転で loss 1.43→2.19)ことを確認済み。
- KV操作: ターン/セッションごとのトークン範囲を `KVHandle` に記録し、`drop(spans, repos=True)` で該当範囲を除去し後続キーを `-削除トークン数` だけ再回転する。
- 評価は自由生成(greedy)の回答を採点する。Day 025 で、指示チューニング済み3Bでは強制尤度(loss)が情報保持を正しく反映しなかったため、lossは補助指標に留める。
- Ollama と同時にGPUを使わない(実行前に `ollama stop`)。

## バックエンド間の比較で気をつけること

- 「Claude vs Qwen」はモデル規模も違う。比較の主眼は**同一バックエンド内での戦略間の差**とし、バックエンド間は「傾向が同じか違うか」を見る。
- テキスト層の同じ戦略を `ollama` と `hf` の両方で回し、ランタイム差(量子化・テンプレート)による違いを一度だけ確認する。
