# ロードマップ(LT: 2026年12月中旬予定)

## 第0週(2026-11-15〜): 事前準備

- [x] リポジトリ作成、README / AGENTS.md / CLAUDE.md / docs 整備
- [x] 環境チェックスクリプト(`scripts/check_env.sh`)
- [x] GitHubリモート作成(公開リポジトリ)・main保護設定(PR必須・管理者にも適用・force push/削除禁止、承認数0)
- [x] ライセンス決定(MIT)

## 第1週: 基盤 + テキスト層(ローカル) ← いまここ

- [x] `pyproject.toml`(uv)、パッケージ骨格、pytest
- [x] シナリオ生成器(日英、規模パラメータ、dev/test分割)+ 決定性テスト
- [x] 採点器(正規化+一致、漏洩判定)+ テスト
- [x] `ollama` バックエンド
- [x] 戦略: `full`, `window`, `facts_append`
- [x] ランナー + `runs/` 出力 + smoke設定
- 完了条件: `qwen2.5:7b-instruct` で smoke → 小規模(5セッション)の表が出る
  - 2026-11-15 達成: results/2026-11-15-week1-small.md

## 第2週: テキスト層の完成 + Claude

- [ ] `claude` バックエンド + docs/backends.md の未確認事項の確認
- [ ] 戦略: `full_redact`, `summary`, `rag`, `hybrid`
- [ ] 規模スイープ(5/20/50セッション)を ollama で実行
- [ ] Claude で同じスイープ(利用枠を見積もってから)

## 第3週: KV層(hf)

- [ ] `hf` バックエンド(eager固定、KVHandle、トークン範囲管理)
- [ ] Day 025 の drop / repos を移植し、delta=0 恒等テストを自動テスト化
- [ ] 戦略: `kv_full`, `kv_drop`, `kv_drop_repos`, `kv_evict_old`
- [ ] テキスト層と同じシナリオで比較

## 第4週: 見せる

- [ ] 図: 覚える力×忘れる力の散布図、規模曲線
- [ ] デモチャットUI(メモリ状態の可視化、KV切除のライブデモ)
- [ ] スライド、発表練習(5分/10分版)

## 予備(時間があれば)

- KV切除を Ollama(llama.cpp)側で再現できるか
- 削除要求を自然文から検出する部分もLLMにやらせた場合の取りこぼし率
