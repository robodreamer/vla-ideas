#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT_DIR/tools/render_latex_pdf.sh" \
  "$ROOT_DIR/video_prompt_shortcut_resistance/docs/video_prompt_shortcut_resistance_report.tex"
