#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-prediction_error_policy_state_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$ROOT_DIR/prediction_error_policy_state/docs/$DOC_NAME"
