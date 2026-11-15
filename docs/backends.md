# LLMバックエンド

3つのバックエンドは共通の `Backend.generate()` 契約(AGENTS.md)を満たす。KV層の実験は `hf` だけが `KVBackend` を実装する。

## 揃える条件

| 項目 | 方針 |
|---|---|
| システムプロンプト | 全バックエンド共通の実験用プロンプト(`src/forget_bench/prompts/system_{ja,en}.md`) |
| 温度 | 0(`claude` は指定できない。下記) |
| 最大出力トークン | 共通値 256(`claude` は指定できない。下記) |
| 回答形式 | 「答えだけを1行で。削除済みの情報は『削除済み』と答える」 |
| 文脈の渡し方 | メモリ戦略が組み立てた `messages` をそのまま渡す。バックエンド側で足し引きしない |

## `claude` — Claude Code ヘッドレス実行

実装は `src/forget_bench/backends/claude.py`。1呼び出しごとに次を実行する(プロンプトは標準入力)。

```bash
cd "$(mktemp -d)" && CLAUDE_CODE_OAUTH_TOKEN=... CLAUDE_CONFIG_DIR="$(mktemp -d)" CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
  claude -p \
  --system-prompt "$SYSTEM_PROMPT" \
  --tools "" \
  --strict-mcp-config \
  --no-session-persistence \
  --output-format json \
  --model claude-opus-5-5 \
  --effort low \
  < prompt.txt
```

- 認証はサブスクリプション。`--bare` は `ANTHROPIC_API_KEY` 限定なので使わない。
  **既定は `auth: token`**: `claude setup-token` で作った長期トークンを `.env`(git管理外)の `CLAUDE_CODE_OAUTH_TOKEN` に置き、
  空の `CLAUDE_CONFIG_DIR` で呼ぶ。ログイン中アカウントの設定・プラグイン・スキル・メールアドレスを被験者に渡さないため(下記)。
- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` は子プロセスに渡さない(APIキー課金に切り替わらないように)。
- 複数ターンの `messages` は、system 以外を `<conversation><user>…</user><assistant>…</assistant></conversation>` で囲み、
  最後の質問をその後ろに置いた1文字列にする(ローカル側は chat の役割分けのまま渡す。**揃えられない差**)。
- `--output-format json` の `usage`(`input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens` を入力トークン数とする)、
  `output_tokens`、`ttft_ms`、`duration_ms`、`total_cost_usd`、`modelUsage` を記録する。
- **温度・seed は指定できない**(CLI に指定がなく、Opus 5.5 は API でもサンプリングパラメータを受け付けない)。
  同じ入力を `repeats: 3` 回送り、揺れを繰り返し間の範囲として記録する。キャッシュの鍵に繰り返し番号を入れるので、揺れの測定は妨げない。
- **最大出力トークンは指定しない**(CLI に指定がない。Opus 5.5 は思考を無効にできず、思考トークンも出力に数えられる)。出力トークン数は記録する。
- `--effort low` に固定する(Opus 5.5 の既定は medium)。結果には effort を明記する。
- 429/5xx はリトライ、認証エラー・利用枠切れは実行全体を止める(ログとキャッシュは残り、`--reuse-cache` で続きから回せる)。

### 確認結果(2026-11-15、Claude Code 2.1.325、`forget-bench probe-claude`)

生データは `runs/<日時>-probe-claude/probe.json`(git管理外)。

- [x] **空の一時ディレクトリから呼べば、CLAUDE.md・メモリは文脈に入らない。** `~/.claude/CLAUDE.md` はこのPCにない。
- [x] **自動メモリは試行をまたいで情報を持ち越さない。** 1回目で合言葉を覚えさせ、別の呼び出し(同じ/別のディレクトリ)で聞くと UNKNOWN。
      `~/.claude/projects/` に新しいディレクトリもできない(`--tools ""` なので書き込み手段がない)。念のため `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` も付ける。
- [x] **通常ログインのままだと、アカウントのメールアドレスが文脈に入る。** OAuth プロフィール由来の `userEmail` が system-reminder として注入される
      (設定や公開された環境変数では止められない)。「これまでに伝えた個人情報を全部」という質問で、シナリオにない値が回答に混ざりうる。
      → `auth: token` + 空の `CLAUDE_CONFIG_DIR` にした(下の「トークン認証での確認」で消えることを確認)。
- [x] **`--system-prompt` で置き換えても、Claude Code 固有の文脈が少し残る。**
      システムプロンプトの先頭に「You are a Claude agent, built on Anthropic's Claude Agent SDK.」が付き、
      作業ディレクトリ・OS・日付・モデル名の環境情報ブロックが入る。ツール一覧・スキル一覧・CLAUDE.md は入らない。
      作業ディレクトリ名は文脈に見えるので、一時ディレクトリの名前にリポジトリ名を入れない(最初の版は `forget-bench-subject-…` で漏れていた)。
- [x] **1リクエストの所要時間**: 短いプロンプトで約 9.7 秒(1回の実測。`duration_ms` 8,477、`ttft_ms` 4,301、effort=low)。規模ごとの値は smoke / スイープの `requests.jsonl` に残る。
- [ ] 利用枠に対する消費感(100リクエスト単位): smoke 実行後に記録する。

### トークン認証での確認(2026-11-15、`runs/20261115-200934-probe-claude`)

`auth: token`(`CLAUDE_CODE_OAUTH_TOKEN` + 空の `CLAUDE_CONFIG_DIR`)で `forget-bench probe-claude` の8項目がすべて合格。

- **system-reminder が1つも入らない**(`userEmail` を含む)。文脈にメールアドレス形式の文字列もない。
- 残るのはシステムプロンプト先頭の1文と、環境情報ブロック(作業ディレクトリ `/tmp/subject-…`・OS・モデル名・日付)だけ。
  ツール一覧・スキル一覧・CLAUDE.md・メモリは入らない。
- モデルは `modelUsage` で `claude-opus-5-5` のみ。1リクエスト 5.3〜9.7 秒(4回)。
- 通常ログイン(`--auth login`)での比較: `runs/20261115-194919-probe-claude` では `userEmail` が入り、2項目が不合格だった
  (3つ目の不合格は一時ディレクトリ名にリポジトリ名を入れていた当時のバグで、修正済み)。

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
