#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: tools/render_latex_pdf.sh path/to/report.tex

Builds a LaTeX report from the repository root. Uses local pdflatex when
available, otherwise builds/runs the shared Docker image in tools/latex/.
Set FORCE_DOCKER=1 to force Docker even when local pdflatex exists.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEX_PATH="$1"

if [[ "$TEX_PATH" != /* ]]; then
  TEX_PATH="$ROOT_DIR/$TEX_PATH"
fi

if [[ ! -f "$TEX_PATH" ]]; then
  echo "LaTeX source not found: $TEX_PATH" >&2
  exit 1
fi

DOC_DIR="$(cd "$(dirname "$TEX_PATH")" && pwd)"
DOC_NAME="$(basename "$TEX_PATH")"
DOC_BASENAME="${DOC_NAME%.tex}"
REL_DOC_DIR="${DOC_DIR#"$ROOT_DIR/"}"
IMAGE_NAME="${IMAGE_NAME:-vla-ideas-latex:ubuntu24}"

run_pdflatex() {
  (cd "$DOC_DIR" && pdflatex -interaction=nonstopmode "$DOC_NAME" && pdflatex -interaction=nonstopmode "$DOC_NAME")
}

if [[ "${FORCE_DOCKER:-0}" != "1" ]] && command -v pdflatex >/dev/null 2>&1; then
  run_pdflatex
elif command -v docker >/dev/null 2>&1; then
  docker build -t "$IMAGE_NAME" "$ROOT_DIR/tools/latex"
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$ROOT_DIR:/workspace" \
    "$IMAGE_NAME" \
    bash -lc "cd '/workspace/$REL_DOC_DIR' && pdflatex -interaction=nonstopmode '$DOC_NAME' && pdflatex -interaction=nonstopmode '$DOC_NAME'"
else
  echo "Neither pdflatex nor docker is available; cannot render PDF." >&2
  exit 1
fi

echo "Rendered $DOC_DIR/${DOC_BASENAME}.pdf"
