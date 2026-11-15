# 下敷きにした過去の検証と出典

このプロジェクトは 100日チャレンジ(ai-100days リポジトリ)での以下の検証を発展させたもの。
コードを移植する場合は元ファイルとライセンスを該当モジュールの先頭に記載する。

| Day | テーマ | 引き継ぐもの | 主な結果(当時の実測) |
|---|---|---|---|
| 025 | AKV: KVキャッシュの drop / repos | `akv_poc_gpu.py` の RoPE再回転実装、eager固定の知見 | Qwen2.5-3B: drop単体は自由生成が崩壊、drop+reposは数値を保持したが存在しない棚を幻視。summarizeは自然な文で誤答(14→9)。SDPAで恒等回転が loss を変える問題を発見 |
| 028 | 上書きメモリ vs 追記のみ vs RAG | 会話データの設計、`memory_methods.py` の戦略構成 | historical正答率: 上書き 16.67% / 追記のみ 100%。dense+rerank RAG は全体 75.00% |
| 031 | Memory Tax(メモリの税と利益) | 「RAGで答えられるのにメモリが負ける」の分解 | 縮小実験では税1件・利益3件 |
| 042 | billion-context | 長文脈の扱い | — |

## 論文・リポジトリ

- AKV (Cache Engineering for Agents): https://huggingface.co/papers/2610.28577 / https://github.com/thu-nics/AKV
- Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought: https://huggingface.co/papers/2610.30755
- Memory Tax: Dissecting the Net Value of Agent Memory: https://huggingface.co/papers/2610.34073
