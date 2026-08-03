#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-turbo_vla_direct_control_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/turbo_vla_direct_control/docs/$DOC_NAME"
