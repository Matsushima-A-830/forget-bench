#!/usr/bin/env bash
# 実験環境の確認(読み取りのみ。何もインストールしない)
set -u

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader | sed 's/^/  /'
else
  warn "nvidia-smi が見つからない(hf バックエンドは使えない)"
fi

echo "== Python / uv"
command -v python3 >/dev/null && ok "$(python3 --version)" || warn "python3 なし"
command -v uv >/dev/null && ok "$(uv --version)" || warn "uv なし(https://docs.astral.sh/uv/)"

echo "== Ollama"
if command -v ollama >/dev/null; then
  ok "$(ollama --version 2>&1 | tail -1)"
  if curl -s -m 3 http://localhost:11434/api/tags >/dev/null; then
    ok "サーバー応答あり"
    ollama list | sed 's/^/  /'
    echo "  -- GPUに載っているモデル(hf実行前は空にする)"
    ollama ps | sed 's/^/  /'
  else
    warn "サーバー応答なし(ollama serve を起動)"
  fi
else
  warn "ollama なし"
fi

echo "== Claude Code"
if command -v claude >/dev/null; then
  ok "$(claude --version)"
else
  warn "claude なし(claude バックエンドは使えない)"
fi

echo "== Disk (HFキャッシュ)"
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | tail -1 | sed 's/^/  /' || df -h "$HOME" | tail -1 | sed 's/^/  /'
