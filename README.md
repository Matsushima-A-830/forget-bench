# forget-bench — LLMは「忘れるべきものを忘れ、覚えるべきものを覚えて」いられるか

> [!WARNING]
> **このリポジトリは実験中です。** 設計・コード・数値は作業途中のもので、予告なく変わります。
> 仮説はまだ検証しておらず、ここにある記述を結論として引用しないでください。
> 確定した結果は `results/` に、実行条件と一緒に置いていきます。

LLMエージェントの**メモリ戦略**(要約・追記・RAG・KVキャッシュ操作など)を、
同一の正解付きシナリオで比較する実験基盤。
クラウドLLM(Claude、`claude -p` 経由)とローカルLLM(Ollama / Hugging Face transformers)の両方で動かし、
「テキストを操作するメモリ」と「推論中のKVキャッシュそのものを操作するメモリ」を同じ物差しで測る。

> ステータス: **第1週(基盤 + テキスト層・ローカル)を実装中**。シナリオ生成器・採点器・`ollama` バックエンド・
> 戦略 `full` / `window` / `facts_append` とランナーが動く。進め方は [docs/roadmap.md](docs/roadmap.md) を参照。

## 使い方

```bash
uv sync                                          # 依存インストール(テキスト層のみ)
uv run pytest                                    # テスト(LLM不要)
uv run forget-bench run configs/smoke.yaml --backend dummy   # LLMなしで全経路を通す
uv run forget-bench run configs/smoke.yaml       # Ollama で smoke
uv run forget-bench run configs/small.yaml       # 5セッション × 3シナリオ × 日英
uv run forget-bench report runs/<run_id>         # 表を再表示
uv run forget-bench scenario --seed 1000         # dev 分割のシナリオを1つ見る
```

## 何を測るか

メモリには相反する2つの能力が要る。

| 能力 | 問い | 指標 |
|---|---|---|
| **覚える力** | 変化した事実の「今の値」と「以前の値」を答えられるか。一度しか出てこない細部を保持できるか | current正答率 / historical正答率 / one-shot細部の正答率 |
| **忘れる力** | ユーザーが「忘れて」と頼んだ情報を、その後の応答で漏らさないか | 削除要求後の漏洩率(leak rate) |
| **コスト** | それを何トークン・何秒・何GBで実現したか | 入力トークン数 / TTFT / 総レイテンシ / ピークVRAM |

仮説(検証前。結果で覆ってよい):

- 追記のみメモリは「覚える力」最強・「忘れる力」最弱になり、トレードオフ図の両端に来る。
- 要約メモリは削除要求後も要約文に情報が残り、漏洩しやすい。
- KVキャッシュの該当範囲を切除する方法は、テキストを経由しないぶん「本当に忘れる」が、位置補正(RoPE再回転)を誤ると全体が崩れる。

## 比較するメモリ戦略

| 層 | 戦略 | 使えるバックエンド |
|---|---|---|
| テキスト層 | full(全文保持) / window(直近N件) / rolling summary(上書き要約) / append-only facts(追記のみの事実ストア) / RAG(密検索+再ランキング) / hybrid(facts + RAG) | Claude・Ollama・HF |
| KV層 | full KV / naive drop / drop + repos(RoPE再回転) / 削除要求範囲のKV切除 | **HFのみ**(KVに直接触れるのはローカルだけ) |

詳細は [docs/design.md](docs/design.md)。

## バックエンド

| 名前 | 実体 | 用途 |
|---|---|---|
| `claude` | Claude Code ヘッドレス実行(`claude -p`) | テキスト層の基準線(クラウド・大規模モデル) |
| `ollama` | ローカル Ollama(qwen2.5 7B/14B 等) | テキスト層のローカル版 |
| `hf` | Hugging Face transformers(Qwen2.5-3B bf16 / 7B 4bit、attention は eager 固定) | KV層の実験 |

各バックエンドの契約・注意点は [docs/backends.md](docs/backends.md)。

## 想定環境

- WSL2 Ubuntu / NVIDIA GeForce RTX 5060 Ti 16GB(Blackwell, sm_120)/ RAM 30GB
- Python 3.11+(`uv` で venv 管理予定)、Ollama 0.35 系、Claude Code 2.1 系

環境の確認:

```bash
bash scripts/check_env.sh
```

## ディレクトリ構成(予定)

```
forget-bench/
├── README.md
├── LICENSE                # MIT
├── CLAUDE.md              # Claude Code で開発するときの指示
├── AGENTS.md              # エージェント共通の開発ルール(Claude Code / Codex / ローカルLLMエージェント)
├── docs/
│   ├── design.md          # システム設計
│   ├── backends.md        # LLMバックエンドの契約と注意点
│   ├── protocol.md        # 実験プロトコル(再現性・記録ルール)
│   ├── roadmap.md         # 4週間の計画
│   └── prior-work.md      # 下敷きにした過去の検証(ai-100days)と出典
├── scripts/check_env.sh   # 環境チェック
├── src/forget_bench/      # 実装
│   ├── scenario/          #   シナリオ生成器(vocab.py が語彙・テンプレート、splits.py が dev/test 分割)
│   ├── memory/            #   メモリ戦略プラグイン
│   ├── backends/          #   LLMバックエンド(ollama / dummy)
│   ├── scoring/           #   採点器と集計
│   ├── prompts/           #   実験用プロンプト(全バックエンド共通)
│   └── runner.py          #   実験ランナー
├── configs/               # 実験設定(YAML)
├── tests/
└── runs/                  # 実験出力(git管理外。要約だけ results/ に残す)
```

## 背景

LT会(2026年12月中旬予定)の発表用。
「生成AIをそのまま使う」話ではなく、「LLMの記憶を自分で設計し、壊し、測る」話をする。
下敷きは 100日チャレンジ(ai-100days)の Day 025 / 028 / 031 / 042 の検証([docs/prior-work.md](docs/prior-work.md))。

## ライセンス

[MIT](LICENSE)。他リポジトリから移植したコードは、該当ファイルの先頭に元のライセンスと出典を記載する(docs/prior-work.md)。
