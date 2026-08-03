#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC_NAME="${1:-recap_concept_writeup.tex}"

"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/recap_pi/docs/$DOC_NAME"
