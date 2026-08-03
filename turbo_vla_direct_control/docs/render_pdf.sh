#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-turbo-vla-direct-control-tex:ubuntu24}"
DOC_DIR="$ROOT_DIR/turbo_vla_direct_control/docs"
DOC_NAME="${1:-turbo_vla_direct_control_report.tex}"
DOC_BASENAME="${DOC_NAME%.tex}"

if command -v pdflatex >/dev/null 2>&1; then
  (cd "$DOC_DIR" && pdflatex -interaction=nonstopmode "$DOC_NAME" && pdflatex -interaction=nonstopmode "$DOC_NAME")
elif command -v docker >/dev/null 2>&1; then
  docker build -t "$IMAGE_NAME" "$DOC_DIR"
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$ROOT_DIR:/workspace" \
    "$IMAGE_NAME" \
    bash -lc "cd /workspace/turbo_vla_direct_control/docs && pdflatex -interaction=nonstopmode '$DOC_NAME' && pdflatex -interaction=nonstopmode '$DOC_NAME'"
else
  echo "Neither pdflatex nor docker is available; cannot render PDF." >&2
  exit 1
fi

echo "Rendered $DOC_DIR/${DOC_BASENAME}.pdf"
