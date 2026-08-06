# Shared tooling

## Report scaffold

Start every new experiment report from [`templates/vla_experiment_report.tex`](templates/vla_experiment_report.tex), then adapt it to the experiment. Do not create custom report scaffolds; the template preserves the shared report structure used by `path_consistent_safety_filtering`, `bspline_action_parameterization`, and `turbo_vla_direct_control`.

## LaTeX report rendering

Use the shared renderer for every idea report:

```bash
./tools/render_latex_pdf.sh path/to/report.tex
```

Each idea folder keeps a small compatibility wrapper at `*/docs/render_pdf.sh`, but the Docker image and compile logic live here:

- `tools/render_latex_pdf.sh` — local `pdflatex` first, Docker fallback, `FORCE_DOCKER=1` supported.
- `tools/latex/Dockerfile` — shared Ubuntu/TeX Live image for all report subfolders.

This avoids each idea folder carrying its own duplicated Dockerfile and render logic.
