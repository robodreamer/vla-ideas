#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-prefix_rl_chunking_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/prefix_rl_chunking/docs/$DOC_NAME"
