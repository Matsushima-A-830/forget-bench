# CLAUDE.md

@AGENTS.md

以下は Claude Code で開発するときの補足。共通ルールは上の AGENTS.md が正。

## Claude Code が守ること

- **実装は roadmap の該当週の範囲だけ。** 先の週の機能を先回りして作らない(docs/roadmap.md)。
- 実験結果を書くときは、`runs/` の生ログを開いて数値を写す。記憶や推測で数値を書かない。
- GPUジョブ(`hf` バックエンド)は実行前に `nvidia-smi` で空きVRAMを確認し、Ollamaがモデルを載せたままなら
  `ollama stop <model>` で降ろしてから走らせる(16GBを取り合うとOOMや速度低下で比較が壊れる)。
- 長時間の実験はバックグラウンド実行し、`runs/<run_id>/progress.jsonl` で進捗を見る。

## 「被験者としてのClaude」と「開発者としてのClaude」を分ける

このリポジトリでは Claude が2つの役割を持つ。混ぜないこと。

| 役割 | 実体 | 注意 |
|---|---|---|
| 開発者 | このセッションの Claude Code | コードを書く。実験結果を見て設計を変えるのはOKだが、test分割の正解を見ながらプロンプトを調整しない |
| 被験者 | `claude` バックエンドが起動する `claude -p` | **このリポジトリの CLAUDE.md・メモリ・ツールを一切見せない**状態で呼ぶ(下記) |

被験者側の `claude -p` は、次の条件で呼ぶ(詳細と確認状況は docs/backends.md)。

- 作業ディレクトリは**空の一時ディレクトリ**(このリポジトリの CLAUDE.md を自動で読ませないため)
- `--system-prompt` で実験用のシステムプロンプトに**置き換える**(Claude Code既定のプロンプトを使わない)
- `--tools ""` でツールを無効化、`--strict-mcp-config` でMCPを無効化
- `--no-session-persistence`、`--output-format json`
- Claude Code自身の自動メモリ(auto-memory)が試行間で情報を持ち越さないことを、第2週の最初に確認してから本実験に入る

> `--bare` は auth が `ANTHROPIC_API_KEY` 限定になる(OAuth/サブスクリプションでは使えない)ため使わない。

## 利用枠

`claude` バックエンドはサブスクリプションの利用枠を消費する。

- 本実験の前に必ず `configs/smoke.yaml` 相当(数問)で1回通す。
- 実行前に「問題数×戦略数×(要約呼び出し含む)想定リクエスト数」を見積もって表示し、100リクエストを超える場合は人間に確認する。
- レスポンスは `runs/<run_id>/cache/` にキャッシュし、同じ入力を二度送らない(再採点はキャッシュから)。

## よく使うコマンド(第1週以降に整備)

```bash
bash scripts/check_env.sh          # 環境確認
uv sync                            # 依存インストール(テキスト層のみ)
uv sync --extra hf                 # KV層用(torch, transformers)
uv run pytest                      # テスト
uv run forget-bench run configs/smoke.yaml
```
