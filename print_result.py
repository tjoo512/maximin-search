#!/usr/bin/env python3
"""
Summarize benchmark results from a directory of run folders.
Each run folder must contain an outputs.jsonl file.

Usage:
    python print_result.py --results_dir <path> [--min_lines <int>]
"""
import json
import os
import re
import glob
import argparse
from dataclasses import dataclass
from typing import Optional


# Matches run directory names produced by save_utils.make_run_id
RUN_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_(?P<model>.+?)_(?P<gen>[a-zA-Z0-9]+)_bs(?P<bs>\d+)_temp(?P<temp>[0-9.]+)$"
)


@dataclass
class RunSummary:
    run_name: str
    generator: str = ""
    batch_size: Optional[int] = None
    temperature: Optional[float] = None
    num_lines: int = 0
    unique_indices: int = 0
    duplicate_indices: int = 0
    accuracy: Optional[float] = None
    format_error_rate: Optional[float] = None
    avg_latency_sec: Optional[float] = None
    valid: bool = False
    reason: str = ""


def fast_count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def parse_run_name(run_name: str) -> dict:
    m = RUN_RE.match(run_name)
    if not m:
        return {}
    d = m.groupdict()
    return {"generator": d["gen"], "batch_size": int(d["bs"]), "temperature": float(d["temp"])}


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def dedup(records: list) -> tuple[list, int]:
    """Remove duplicate indices, keeping the first occurrence."""
    seen, duplicates = {}, 0
    for r in records:
        i = r.get("index")
        if i is None:
            continue
        if i in seen:
            duplicates += 1
        else:
            seen[i] = r
    return list(seen.values()), duplicates


def compute_metrics(records: list) -> tuple:
    """Return (accuracy,  format_error_rate, avg_latency)."""
    if not records:
        return None, None, None

    correct, fmt_errors, latencies = [], [], []

    for r in records:
        pred = r.get("prediction")
        gt = r.get("ground_truth")

        if pred is not None:
            fmt_errors.append(pred == "N/A")

        if "correct" in r:
            correct.append(bool(r["correct"]))
        elif pred is not None and gt is not None:
            try:
                match = float(pred) == float(gt)
                correct.append(match)
            except (TypeError, ValueError):
                pass

        if "latency_sec" in r:
            latencies.append(float(r["latency_sec"]))

    acc       = sum(correct) / len(correct)             if correct     else None
    fer       = sum(fmt_errors) / len(fmt_errors)       if fmt_errors  else None
    avg_lat   = sum(latencies) / len(latencies)         if latencies   else None

    return acc, fer, avg_lat


def summarize_one(results_dir: str, run_name: str, min_lines: int) -> RunSummary:
    s = RunSummary(run_name=run_name)
    for k, v in parse_run_name(run_name).items():
        setattr(s, k, v)

    # Match both single-GPU (outputs.jsonl) and data-parallel shards
    # (outputs.rank0.jsonl, outputs.rank1.jsonl, ...).
    out_files = sorted(glob.glob(os.path.join(results_dir, run_name, "outputs*.jsonl")))
    if not out_files:
        s.reason = "no outputs.jsonl"
        return s

    s.num_lines = sum(fast_count_lines(p) for p in out_files)
    if s.num_lines < min_lines:
        s.reason = f"<{min_lines} lines"
        return s

    records = []
    for p in out_files:
        records.extend(load_jsonl(p))
    if not records:
        s.reason = "no valid json"
        return s

    records, s.duplicate_indices = dedup(records)
    s.unique_indices = len(records)

    s.accuracy, s.format_error_rate, s.avg_latency_sec = compute_metrics(records)
    s.valid = True
    return s


def print_table(summaries: list[RunSummary]) -> None:
    header = (
        f"{'run_name':<120} | "
        f"{'lines':>7} | "
        f"{'uniq':>6} | "
        f"{'acc':>7} | "
        f"{'fmt_err':>7}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.run_name:<120} | "
            f"{(s.num_lines or 0):>7} | "
            f"{(s.unique_indices or 0):>6} | "
            f"{(s.accuracy or 0):>7.4f} | "
            f"{(s.format_error_rate or 0):>7.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="bench_outputs/ablation_bench_math500_qwen25_math_15b_prm",
        help="Path to the directory containing run subdirectories.",
    )
    parser.add_argument(
        "--min_lines", type=int, default=5,
        help="Skip runs with fewer than this many output lines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )
    print(f"Found {len(runs)} run(s) in {args.results_dir}\n")

    summaries = []
    for run_name in runs:
        s = summarize_one(args.results_dir, run_name, args.min_lines)
        if s.valid:
            summaries.append(s)

    summaries.sort(key=lambda x: (x.generator, x.batch_size or 0))
    print_table(summaries)