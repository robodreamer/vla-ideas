#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-pacs_toy_report.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/path_consistent_safety_filtering/docs/$DOC_NAME"
