#!/usr/bin/env python3
"""
Run end-to-end invariant benchmarks case-by-case in fresh processes and combine results.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


_RESULT_COLS = [
    "tool",
    "case_id",
    "regime",
    "invariant_kind",
    "degree",
    "n_points",
    "ambient_dim",
    "max_dim",
    "cold_ms",
    "cold_alloc_kib",
    "warm_median_ms",
    "warm_p90_ms",
    "warm_alloc_median_kib",
    "output_term_count",
    "output_abs_mass",
    "output_measure_canonical",
    "output_rank_query_axes_canonical",
    "output_rank_table_canonical",
    "notes",
    "timestamp_utc",
]

_FAILURE_COLS = [
    "tool",
    "case_id",
    "regime",
    "invariant_kind",
    "degree",
    "run_status",
    "exit_code",
    "failure_reason",
    "log_path",
    "command",
]


def _configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
            if limit <= 0:
                raise


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    return raw


def _parse_requested_invariants(raw: str) -> list[str] | None:
    token = raw.strip().lower()
    if token in ("", "all"):
        return None
    out: list[str] = []
    for part in token.split(","):
        item = part.strip().lower().replace("-", "_")
        if item in ("rank", "rank_signed_measure"):
            out.append("rank_signed_measure")
        elif item in ("slice", "slice_barcodes"):
            out.append("slice_barcodes")
        elif item in ("euler", "euler_signed_measure"):
            out.append("euler_signed_measure")
        else:
            raise ValueError(f"Unsupported invariant token: {part}")
    return list(dict.fromkeys(out))


def _filter_cases_by_eligibility(
    cases: list[dict[str, Any]],
    invariant_eligibility: dict[str, Any],
    requested_invariants: list[str] | None,
) -> tuple[list[dict[str, Any]], int]:
    filtered: list[dict[str, Any]] = []
    skipped = 0
    for case in cases:
        regime = str(case["regime"])
        approved_raw = invariant_eligibility.get(regime)
        if approved_raw is None:
            filtered.append(case)
            continue
        if not isinstance(approved_raw, list):
            raise ValueError(f"invariant_eligibility[{regime!r}] must be an array.")
        approved = [str(v) for v in approved_raw]
        if requested_invariants is None:
            use = approved
        else:
            approved_set = set(approved)
            use = [inv for inv in requested_invariants if inv in approved_set]
        if use:
            filtered.append(case)
        else:
            skipped += 1
    return filtered, skipped


def _write_rows(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _last_reason(text: str, rc: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1][:400]
    return f"runner_exit_code={rc}"


def _build_cmd(tool: str, args, out_path: Path, case_id: str) -> list[str]:
    if tool == "tamer":
        cmd = [
            args.julia,
            "--startup-file=no",
            "--history-file=no",
            "--project=.",
            "benchmark/ingestion_compare_harness/run_tamer_invariants.jl",
            f"--manifest={args.manifest}",
            f"--out={out_path}",
            f"--profile={args.profile}",
            f"--invariants={args.invariants}",
            f"--degree={args.degree}",
            f"--case={case_id}",
        ]
        if args.reps is not None:
            cmd.append(f"--reps={args.reps}")
        return cmd
    if tool == "multipers":
        cmd = [
            args.python,
            "benchmark/ingestion_compare_harness/run_multipers_invariants.py",
            "--manifest",
            str(args.manifest),
            "--out",
            str(out_path),
            "--profile",
            args.profile,
            "--invariants",
            args.invariants,
            "--degree",
            str(args.degree),
            "--case",
            case_id,
        ]
        if args.reps is not None:
            cmd.extend(["--reps", str(args.reps)])
        return cmd
    raise ValueError(f"Unsupported tool={tool}")


def _run_tool(tool: str, cases: list[dict[str, Any]], args, result_out: Path, failure_out: Path, case_dir: Path, log_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_id = str(case["id"])
        regime = str(case["regime"])
        case_csv = case_dir / f"{case_id}.csv"
        log_path = log_dir / f"{tool}_{case_id}.log"
        cmd = _build_cmd(tool, args, case_csv, case_id)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        combined = (proc.stdout or "") + (proc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")

        if proc.returncode == 0 and case_csv.exists():
            rows = _read_rows(case_csv)
            if rows:
                successes.extend(rows)
                continue

        failures.append(
            {
                "tool": tool,
                "case_id": case_id,
                "regime": regime,
                "invariant_kind": args.invariants,
                "degree": args.degree,
                "run_status": "failed",
                "exit_code": proc.returncode,
                "failure_reason": _last_reason(combined, proc.returncode),
                "log_path": str(log_path),
                "command": " ".join(cmd),
            }
        )

    _write_rows(result_out, successes, _RESULT_COLS)
    _write_rows(failure_out, failures, _FAILURE_COLS)
    return successes, failures


def _default_failure_path(result_path: Path) -> Path:
    stem = result_path.stem
    if stem.startswith("results_"):
        stem = "failures_" + stem[len("results_") :]
    else:
        stem = stem + "_failures"
    return result_path.with_name(stem + result_path.suffix)


def main() -> None:
    _configure_csv_field_limit()
    p = argparse.ArgumentParser(description="Run invariant benchmark matrix case-by-case in fresh processes.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tools", choices=("tamer", "multipers", "both"), default="both")
    p.add_argument("--profile", default="desktop")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--invariants", default="all")
    p.add_argument("--degree", type=int, default=0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--julia", default="julia")
    p.add_argument("--work_dir", type=Path, default=Path("benchmark/ingestion_compare_harness/_run_invariant_supervisor"))
    p.add_argument("--tamer_out", type=Path, default=Path("benchmark/ingestion_compare_harness/results_tamer_invariants.csv"))
    p.add_argument("--multipers_out", type=Path, default=Path("benchmark/ingestion_compare_harness/results_multipers_invariants.csv"))
    p.add_argument("--tamer_failures_out", type=Path, default=None)
    p.add_argument("--multipers_failures_out", type=Path, default=None)
    p.add_argument("--comparison_out", type=Path, default=Path("benchmark/ingestion_compare_harness/comparison_invariants.csv"))
    p.add_argument("--summary_out", type=Path, default=Path("benchmark/ingestion_compare_harness/comparison_summary_invariants.csv"))
    p.add_argument("--comparison_failures_out", type=Path, default=Path("benchmark/ingestion_compare_harness/comparison_failures_invariants.csv"))
    args = p.parse_args()

    manifest = _load_manifest(args.manifest)
    cases = manifest["cases"]
    requested_invariants = _parse_requested_invariants(args.invariants)
    cases, skipped_cases = _filter_cases_by_eligibility(
        cases,
        manifest.get("invariant_eligibility", {}),
        requested_invariants,
    )
    args.tamer_failures_out = args.tamer_failures_out or _default_failure_path(args.tamer_out)
    args.multipers_failures_out = args.multipers_failures_out or _default_failure_path(args.multipers_out)

    if skipped_cases:
        print(f"[supervisor] skipped {skipped_cases} case(s) with no benchmark-approved invariants under request={args.invariants!r}")

    tool_set = ["tamer", "multipers"] if args.tools == "both" else [args.tools]
    work_dir = args.work_dir
    successes: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}

    for tool in tool_set:
        result_out = args.tamer_out if tool == "tamer" else args.multipers_out
        failure_out = args.tamer_failures_out if tool == "tamer" else args.multipers_failures_out
        case_dir = work_dir / "cases" / tool
        log_dir = work_dir / "logs"
        success_rows, failure_rows = _run_tool(tool, cases, args, result_out, failure_out, case_dir, log_dir)
        successes[tool] = success_rows
        failures[tool] = failure_rows
        print(f"[{tool}] successes={len(success_rows)} failures={len(failure_rows)} results={result_out}")

    if args.tools == "both":
        cmd = [
            args.python,
            "benchmark/ingestion_compare_harness/compare_invariants.py",
            "--tamer",
            str(args.tamer_out),
            "--multipers",
            str(args.multipers_out),
            "--tamer_failures",
            str(args.tamer_failures_out),
            "--multipers_failures",
            str(args.multipers_failures_out),
            "--out",
            str(args.comparison_out),
            "--summary_out",
            str(args.summary_out),
            "--failures_out",
            str(args.comparison_failures_out),
        ]
        subprocess.run(cmd, check=True, cwd=".")
        print(f"[compare] out={args.comparison_out} summary={args.summary_out} failures={args.comparison_failures_out}")

    requested_successes = sum(len(successes.get(tool, [])) for tool in tool_set)
    if requested_successes == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
