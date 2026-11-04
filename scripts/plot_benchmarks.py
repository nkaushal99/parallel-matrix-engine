#!/usr/bin/env python3
"""Generate SVG benchmark plots and update the README plot section."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "benchmarks.csv"
PLOTS_DIR = ROOT / "docs" / "plots"
README_PATH = ROOT / "README.md"
PLOT_START = "<!-- benchmark-plots:start -->"
PLOT_END = "<!-- benchmark-plots:end -->"


def load_rows() -> list[dict[str, object]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("data/benchmarks.csv has no measurements")

    parsed: list[dict[str, object]] = []
    for row in rows:
        try:
            parsed.append(
                {
                    "implementation": row["implementation"],
                    "matrix_size": int(row["matrix_size"]),
                    "block_size": int(row["block_size"]),
                    "workers": int(row["workers"]),
                    "time_ms": float(row["time_ms"]),
                    "result_sum": float(row["result_sum"]),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid row in {DATA_PATH}: {row}") from error
    return parsed


def validate_checksums(rows: list[dict[str, object]]) -> None:
    sums = [row["result_sum"] for row in rows]
    reference = sums[0]
    tolerance = max(1e-3, abs(reference) * 1e-6)
    if any(abs(value - reference) > tolerance for value in sums):
        raise SystemExit(
            "result_sum values disagree; investigate correctness before plotting "
            f"(reference={reference}, tolerance={tolerance})"
        )


def median_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["implementation"], row["matrix_size"], row["block_size"], row["workers"])
        groups[key].append(row["time_ms"])
    return [
        {
            "implementation": key[0],
            "matrix_size": key[1],
            "block_size": key[2],
            "workers": key[3],
            "time_ms": statistics.median(values),
        }
        for key, values in groups.items()
    ]


def plot_metric(results: list[dict[str, object]], metric: str, ylabel: str, filename: str) -> None:
    sizes = sorted({row["matrix_size"] for row in results})
    figure, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4.5), squeeze=False)
    for axis, matrix_size in zip(axes[0], sizes):
        subset = [row for row in results if row["matrix_size"] == matrix_size and metric in row]
        for implementation in sorted({row["implementation"] for row in subset}):
            points = sorted(
                (row["workers"], row[metric])
                for row in subset
                if row["implementation"] == implementation
            )
            axis.plot(*zip(*points), marker="o", linewidth=2, label=implementation)
        axis.set_title(f"Matrix size {matrix_size} × {matrix_size}")
        axis.set_xlabel("Workers (threads or ranks)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / filename, format="svg")
    plt.close(figure)


def add_speedup(results: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines = {
        row["matrix_size"]: row["time_ms"]
        for row in results
        if row["implementation"] == "naive" and row["workers"] == 1
    }
    if not baselines:
        raise SystemExit("record at least one single-worker naive measurement before plotting speedup")
    return [
        {**row, "speedup": baselines[row["matrix_size"]] / row["time_ms"]}
        for row in results
        if row["matrix_size"] in baselines
    ]


def update_readme() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    replacement = "\n".join(
        [
            PLOT_START,
            "![Median runtime by worker count](docs/plots/runtime-by-workers.svg)",
            "",
            "![Speedup relative to naive baseline](docs/plots/speedup-vs-naive.svg)",
            PLOT_END,
        ]
    )
    start = readme.find(PLOT_START)
    end = readme.find(PLOT_END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit("README benchmark plot markers are missing")
    README_PATH.write_text(readme[:start] + replacement + readme[end + len(PLOT_END):], encoding="utf-8")


def main() -> None:
    rows = load_rows()
    validate_checksums(rows)
    results = median_results(rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric(results, "time_ms", "Median runtime (ms)", "runtime-by-workers.svg")
    plot_metric(add_speedup(results), "speedup", "Speedup vs. naive (×)", "speedup-vs-naive.svg")
    update_readme()
    print("wrote docs/plots/runtime-by-workers.svg and docs/plots/speedup-vs-naive.svg")


if __name__ == "__main__":
    main()
