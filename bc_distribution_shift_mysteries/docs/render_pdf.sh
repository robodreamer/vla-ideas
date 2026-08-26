#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/bc_distribution_shift_mysteries/docs/bc_distribution_shift_mysteries_report.tex"
