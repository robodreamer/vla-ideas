#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_DIR="$ROOT_DIR/streaming_action_denoising/docs"
rm -f "$DOC_DIR"/streaming_action_denoising_report.{aux,log,out}
"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$DOC_DIR/streaming_action_denoising_report.tex"
