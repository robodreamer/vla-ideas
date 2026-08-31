#!/usr/bin/env python3
"""Fail-fast artifact verification for the async-control mechanism package."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DOC = ROOT / "docs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    expected = [
        OUT / "metrics.json",
        OUT / "training_metrics.json",
        OUT / "sanity_checks.json",
        OUT / "trial_metrics.csv",
        OUT / "summary_metrics.csv",
        OUT / "sweep_trial_metrics.csv",
        OUT / "sweep_summary.csv",
        OUT / "instruction_relabeling_segments.csv",
        OUT / "method_comparison.png",
        OUT / "mechanism_sweeps.png",
        OUT / "representative_rollout.png",
        DOC / "instruction_conditioned_async_control_report.tex",
        DOC / "instruction_conditioned_async_control_report.pdf",
    ]
    for path in expected:
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty: {path.relative_to(ROOT)}")

    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    sanity = json.loads((OUT / "sanity_checks.json").read_text(encoding="utf-8"))
    require(sanity.get("all_passed") is True, "sanity checks did not pass")
    require("not a Dreamer" in metrics["claim_boundary"], "claim boundary is missing")
    cfg = metrics["config"]
    with (OUT / "trial_metrics.csv").open(newline="", encoding="utf-8") as handle:
        trials = list(csv.DictReader(handle))
    require(len(trials) == int(cfg["trials"]) * 5, "unexpected default trial row count")
    require({r["method"] for r in trials} == set(metrics["method_summary"]), "method coverage mismatch")

    for csv_path in (OUT / "trial_metrics.csv", OUT / "summary_metrics.csv", OUT / "sweep_trial_metrics.csv", OUT / "sweep_summary.csv"):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if key.endswith(("_mean", "_sem", "_fraction", "_steps", "_errors", "_violations", "_calls")) and value:
                        try:
                            require(math.isfinite(float(value)), f"non-finite {key} in {csv_path.name}")
                        except ValueError:
                            pass

    for png in OUT.glob("*.png"):
        require(png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"invalid PNG: {png.name}")
    pdf = DOC / "instruction_conditioned_async_control_report.pdf"
    require(pdf.read_bytes()[:5] == b"%PDF-", "report is not a real PDF")
    tex = (DOC / "instruction_conditioned_async_control_report.tex").read_text(encoding="utf-8")
    require("not a Dreamer" in tex and "2608.26788" in tex, "report scope/reference check failed")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("mechanism toy" in readme.lower() and "2608.26788" in readme, "README scope/reference check failed")

    print(f"verified {len(expected)} required artifacts, {len(trials)} paired default trial rows, and a real PDF")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise
