#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-temporal_robustness_transfer_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$ROOT_DIR/temporal_robustness_transfer/docs/$DOC_NAME"
