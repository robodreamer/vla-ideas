#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" "$ROOT_DIR/causal_interaction_memory/docs/causal_interaction_memory_report.tex"
