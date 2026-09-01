#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/skill_reset_diagnostics/docs/skill_reset_diagnostics_report.tex"
