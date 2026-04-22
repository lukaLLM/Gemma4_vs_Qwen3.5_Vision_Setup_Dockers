#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# FP8 model downloader for the local compare-lab setup.
#
# Default downloads in this repo:
# - Qwen/Qwen3.5-27B-FP8
# - RedHatAI/gemma-4-31B-it-FP8-block
#
# The script reads HF_TOKEN from the environment or a local .env file.
# Re-running resumes partial downloads automatically.
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]] && grep -qE '^HF_TOKEN=' ".env"; then
    # Load HF_TOKEN from local project env file if present.
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing dependency: 'hf' command not found."
  echo "Install with uv:"
  echo "  uv init"
  echo "  uv venv .venv"
  echo "  source .venv/bin/activate"
  echo "  uv add hf-transfer huggingface-hub"
  echo "Then activate your venv and run this script again."
  exit 127
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN found (from env or .env). Using token for downloads."
else
  echo "HF_TOKEN not found in environment or .env. Continuing without login."
  echo "For private/gated repos, set HF_TOKEN in .env or run:"
  echo "export HF_TOKEN=\"your_hf_token_here\""
fi

download_or_exit() {
  local repo="$1"
  shift

  if hf download "$repo" --type model "$@"; then
    return 0
  fi

  local rc=$?
  if [[ $rc -eq 130 ]]; then
    echo "Interrupted by Ctrl+C."
    exit 130
  fi

  echo "Failed: $repo (exit $rc)"
  exit "$rc"
}

# Kept for compatibility with older README guidance.
# This repo currently uses FP8 Hugging Face model repos only.
GGUF_MODELS=()

HF_MODELS=(
  "Qwen/Qwen3.5-27B-FP8"
  "RedHatAI/gemma-4-31B-it-FP8-block"
)

for REPO in "${GGUF_MODELS[@]}"; do
  echo "Downloading Q4_K_M for $REPO..."
  download_or_exit "$REPO" --include "*Q4_K_M*.gguf"

  echo "Downloading mmproj for $REPO..."
  download_or_exit "$REPO" --include "*mmproj*.gguf"

  echo "Done: $REPO"
  echo "----------------------------------------"
done

if [[ ${#HF_MODELS[@]} -eq 0 ]] && [[ ${#GGUF_MODELS[@]} -eq 0 ]]; then
  echo "No models selected. Edit download_qwen_models.sh and add entries to HF_MODELS."
  exit 0
fi

for REPO in "${HF_MODELS[@]}"; do
  echo "Downloading full repo for $REPO..."
  download_or_exit "$REPO"

  echo "Done: $REPO"
  echo "----------------------------------------"
done
