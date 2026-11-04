#!/usr/bin/env python3
"""Append one executable's key=value output to data/benchmarks.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "benchmarks.csv"
FIELDS = ["implementation", "matrix_size", "block_size", "workers", "run", "time_ms", "result_sum"]


def parse_metrics(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key and value:
            metrics[key.strip()] = value.strip()
    required = {"method", "size", "time_ms", "result_sum"}
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"missing executable output fields: {', '.join(sorted(missing))}")
    return metrics


def next_run(metrics: dict[str, str]) -> int:
    if not DATA_PATH.exists():
        return 1
    key = (
        metrics["method"],
        metrics["size"],
        metrics.get("block_size", "0"),
        metrics.get("threads", metrics.get("ranks", "1")),
    )
    existing_runs: list[int] = []
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row_key = (row["implementation"], row["matrix_size"], row["block_size"], row["workers"])
            if row_key == key:
                existing_runs.append(int(row["run"]))
    return max(existing_runs, default=0) + 1


def main() -> None:
    metrics = parse_metrics(sys.stdin.read())
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0
    row = {
        "implementation": metrics["method"],
        "matrix_size": metrics["size"],
        "block_size": metrics.get("block_size", "0"),
        "workers": metrics.get("threads", metrics.get("ranks", "1")),
        "run": str(next_run(metrics)),
        "time_ms": metrics["time_ms"],
        "result_sum": metrics["result_sum"],
    }
    with DATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"recorded {row['implementation']} run {row['run']} in {DATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
