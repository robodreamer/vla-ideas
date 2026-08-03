#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" "$SCRIPT_DIR/explorative_policy_toy_report.tex"
