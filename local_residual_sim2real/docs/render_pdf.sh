#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-local_residual_sim2real_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/local_residual_sim2real/docs/$DOC_NAME"
