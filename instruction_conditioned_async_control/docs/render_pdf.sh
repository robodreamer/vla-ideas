#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_DIR="$ROOT_DIR/instruction_conditioned_async_control/docs"
rm -f "$DOC_DIR"/instruction_conditioned_async_control_report.{aux,log,out}
"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$DOC_DIR/instruction_conditioned_async_control_report.tex"
