# AGENTS.md — エージェント共通の開発ルール

このファイルは、このリポジトリで作業するすべてのコーディングエージェント
(Claude Code、Codex系CLI、ローカルLLMを使うエージェント等)に共通のルール。
Claude Code 固有の補足は CLAUDE.md にある(CLAUDE.md はこのファイルを読み込む)。

## プロジェクトの目的(1行)

メモリ戦略(テキスト層・KV層)ごとの「覚える力・忘れる力・コスト」を、
正解付きシナリオでClaudeとローカルLLMの両方について比較する。設計は docs/design.md。

## 絶対ルール

1. **動かしていないものを「動いた」と書かない。**
   数値・グラフ・スライドに載せるものは、`runs/` に生ログがある実測値だけ。
   失敗した実行も消さずに残す(失敗も発表のネタになる)。
2. **採点はLLMにさせない。** 正解は生成器が持つ短い固有値で、採点は決定的なコード(正規化+完全一致/包含)で行う。
   LLM-as-judge を補助指標として足す場合は、主指標と混ぜずに別列にする。
3. **実験は再現可能にする。** すべての実行は `seed` / モデル名とバージョン(Ollamaはdigest、HFはrevision)/
   設定ファイル / git commit を `runs/<run_id>/meta.json` に記録する。記録のない数値は使わない。
4. **評価用シナリオを学習・プロンプト調整にリークさせない。** プロンプトを調整するときは `dev` 分割のseedだけを使い、
   `test` 分割のseedで最終評価する。
5. **バックエンド間で条件を揃える。** 同じ戦略なら、システムプロンプト・メモリに渡す材料・温度(0)・最大出力長を揃える。
   揃えられない差(例: Claudeは温度を指定できない場合がある)は docs/backends.md と結果に明記する。
6. **秘密情報をコミットしない。** APIキー・トークンはファイルに書かない。`.env` は git 管理外。
7. **mainへ直接pushしない。** 変更はブランチを切ってPRにする(リモート作成後は main を保護する)。

## コーディング規約

- Python 3.11+。依存管理は `uv`(`pyproject.toml`)。パッケージは `src/forget_bench/`。
- 型ヒント必須。データの受け渡しは `dataclass` か `pydantic` で、dictの生受け渡しを避ける。
- メモリ戦略・バックエンドはプラグイン(後述のインターフェース)として実装し、ランナーに `if backend == ...` を書かない。
- 重い依存(torch, transformers)は `hf` extra に分け、テキスト層だけなら torch なしで動くようにする。
- テストは `pytest`。シナリオ生成器と採点器は**LLMなしで**テストできること(決定性テスト必須)。
- ログは JSONL(1リクエスト1行: 入力トークン数・出力・レイテンシ・エラー)。

## インターフェース(第1週で実装する契約)

```python
class Backend(Protocol):
    name: str
    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation: ...
    # Generation: text, input_tokens, output_tokens, latency_s, ttft_s | None, raw(任意)

class KVBackend(Backend, Protocol):          # hf のみ
    def prefill(self, messages: list[Message]) -> KVHandle: ...
    def drop(self, kv: KVHandle, spans: list[TokenSpan], *, repos: bool) -> KVHandle: ...
    def generate_from(self, kv: KVHandle, prompt: list[Message], *, max_tokens: int) -> Generation: ...

class MemoryStrategy(Protocol):
    name: str
    def observe(self, turn: Turn) -> None: ...          # 会話を1ターン受け取る
    def forget(self, request: ForgetRequest) -> None: ...  # 削除要求(シナリオ内の「忘れて」発話とは別に明示シグナルも受ける)
    def build_context(self, question: Question) -> list[Message]: ...  # 質問時にLLMへ渡す文脈
    def stats(self) -> MemoryStats: ...                 # 保持トークン数など
```

要約生成など、メモリ戦略自身がLLMを呼ぶ場合は、**回答用と同じバックエンド**を使う(Claudeの要約をローカルの回答に流用しない)。
この呼び出しのトークンもコストに計上する。

## 作業の進め方

- 1つのPRは1つの関心事(生成器、ある戦略、あるバックエンド…)に絞る。
- 実験を回したら、`results/<日付>-<名前>.md` に「設定・実行コマンド・表・気づき」を残す。生ログは `runs/`(git管理外)。
- GPUを使う長い実行の前に、小さい設定(`configs/smoke.yaml`)で全経路が通ることを確認する。
