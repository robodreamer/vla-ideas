#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-force_feedback_demonstration_quality_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$ROOT_DIR/force_feedback_demonstration_quality/docs/$DOC_NAME"
