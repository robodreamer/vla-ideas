# VLA experiment report template

Every new experiment report must start by copying [`vla_experiment_report.tex`](vla_experiment_report.tex), then adapting its existing scaffold. Do not create a custom report scaffold.

The template is the repository-native skeleton copied from the peer reports: its preamble, title/author/date convention, section progression, `booktabs` table pattern, `[H]` figure pattern, and inline bibliography pattern are the house style. The closest reference tracks are:

- [`path_consistent_safety_filtering`](../../path_consistent_safety_filtering/docs/pacs_toy_report.tex)
- [`bspline_action_parameterization`](../../bspline_action_parameterization/docs/bspline_action_report.tex)
- [`turbo_vla_direct_control`](../../turbo_vla_direct_control/docs/turbo_vla_direct_control_report.tex)

Keep each track's thin `docs/render_pdf.sh` wrapper and render through the shared tool:

```bash
./your_track/docs/render_pdf.sh
# or
./tools/render_latex_pdf.sh your_track/docs/your_report.tex
```
