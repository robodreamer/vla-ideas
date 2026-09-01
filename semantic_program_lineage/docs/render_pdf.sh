#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/semantic_program_lineage/docs/semantic_program_lineage_report.tex"
