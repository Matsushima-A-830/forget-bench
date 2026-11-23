# ロードマップ(LT: 2026年12月中旬予定)

## 第0週(2026-11-15〜): 事前準備

- [x] リポジトリ作成、README / AGENTS.md / CLAUDE.md / docs 整備
- [x] 環境チェックスクリプト(`scripts/check_env.sh`)
- [x] GitHubリモート作成(公開リポジトリ)・main保護設定(PR必須・管理者にも適用・force push/削除禁止、承認数0)
- [x] ライセンス決定(MIT)

## 第1週: 基盤 + テキスト層(ローカル)

- [x] `pyproject.toml`(uv)、パッケージ骨格、pytest
- [x] シナリオ生成器(日英、規模パラメータ、dev/test分割)+ 決定性テスト
- [x] 採点器(正規化+一致、漏洩判定)+ テスト
- [x] `ollama` バックエンド
- [x] 戦略: `full`, `window`, `facts_append`
- [x] ランナー + `runs/` 出力 + smoke設定
- 完了条件: `qwen2.5:7b-instruct` で smoke → 小規模(5セッション)の表が出る
  - 2026-11-15 達成: results/2026-11-15-week1-small.md

## 第2週: テキスト層の完成 + Claude

- [x] `claude` バックエンド + docs/backends.md の未確認事項の確認
- [x] 戦略: `full_redact`, `summary`, `rag`, `hybrid`
- [x] 規模スイープ(5/20/50セッション)を ollama で実行(results/2026-11-15-week2-sweep.md)
- [x] Claude で同じスイープ(利用枠を見積もってから)
  - フル設計(12,426リクエスト)は1,110リクエストで5時間枠に達して停止。縮小版(日本語 × seed 1000 × 5/20/50 × 繰り返し1)で完了(2026-11-16)。
    英語・複数シナリオ・繰り返しは未実施

## 第3週: KV層(hf)

- [x] `hf` バックエンド(eager固定、KVHandle、トークン範囲管理)
- [x] Day 025 の drop / repos を移植し、delta=0 恒等テストを自動テスト化
- [x] 戦略: `kv_full`, `kv_drop`, `kv_drop_repos`, `kv_evict_old`
- [x] テキスト層と同じシナリオで比較(results/2026-11-21-week3-kv.md)
  - Qwen2.5-3B(bf16)だけで比較。7B 4bit・test 分割・繰り返しは未実施

## 第4週: 見せる ← いまここ

- [x] 図: 覚える力×忘れる力の散布図、規模曲線(`forget-bench figures`、results/2026-11-23-week4-figures.md)
- [x] デモチャットUI(メモリ状態の可視化、KV切除のライブデモ)(`forget-bench demo`、docs/design.md「5. デモUI」)
- [ ] スライド、発表練習(5分/10分版)
  - スライドの下書き(`talk/slides.md` 10分版・`talk/slides-5min.md` 5分版)と台本(`talk/script.md`)まで。
    PDF への書き出しと練習は未(人間がやる)

## 第4週の追加検証(発表の主張を支えるもの)

- [ ] test 分割での最終評価(ollama・hf。設定は dev のまま `configs/sweep_*_test.yaml`。Claude は利用枠の確認待ち)
- [ ] 揺れの幅: 問題単位のブートストラップで信頼区間を付ける(新たな生成なし)
- [ ] 漏洩判定の取りこぼし点検: 表記を崩した漏洩・部分漏洩を既存の回答ログで洗う(新たな生成なし)

## 予備(時間があれば)

- KV切除を Ollama(llama.cpp)側で再現できるか
- 削除要求を自然文から検出する部分もLLMにやらせた場合の取りこぼし率
