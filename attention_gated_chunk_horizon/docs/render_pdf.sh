#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$ROOT_DIR/attention_gated_chunk_horizon/docs/attention_gated_chunk_horizon_report.tex"
