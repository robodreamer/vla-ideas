#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-recap-pi-tex:ubuntu24}"
DOC_DIR="$ROOT_DIR/recap_pi/docs"
DOC_NAME="${1:-recap_concept_writeup.tex}"
DOC_BASENAME="${DOC_NAME%.tex}"

docker build -t "$IMAGE_NAME" "$DOC_DIR"
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$ROOT_DIR:/workspace" \
  "$IMAGE_NAME" \
  bash -lc "cd /workspace/recap_pi/docs && pdflatex -interaction=nonstopmode '$DOC_NAME' && pdflatex -interaction=nonstopmode '$DOC_NAME'"

echo "Rendered $DOC_DIR/${DOC_BASENAME}.pdf"
