#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-bspline_action_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/bspline_action_parameterization/docs/$DOC_NAME"
