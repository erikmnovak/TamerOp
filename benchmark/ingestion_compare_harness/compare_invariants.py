#!/usr/bin/env python3
"""
Compare tamer-op and multipers end-to-end invariant benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeasureTerm:
    coords: tuple[float, ...]
    weight: Fraction


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_optional_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return _read_csv(path)


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "":
        return float("nan")
    return float(s)


def _geom_mean(vals: list[float]) -> float:
    good = [v for v in vals if math.isfinite(v) and v > 0.0]
    if not good:
        return float("nan")
    return math.exp(sum(math.log(v) for v in good) / len(good))


def _write_rows(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _weight_fraction(raw: str) -> Fraction:
    token = raw.strip()
    if token == "":
        return Fraction(0, 1)
    if "//" in token:
        num, den = token.split("//", 1)
        return Fraction(int(num), int(den))
    return Fraction(Decimal(token))


def _weight_token(weight: Fraction) -> str:
    if weight.denominator == 1:
        return str(weight.numerator)
    return f"{weight.numerator}//{weight.denominator}"


def _parse_measure_terms(raw: str) -> list[MeasureTerm]:
    s = raw.strip()
    if s == "":
        return []
    acc: dict[tuple[float, ...], Fraction] = defaultdict(Fraction)
    for chunk in s.split(";"):
        if chunk == "":
            continue
        coords_raw, weight_raw = chunk.rsplit("=>", 1)
        coords = tuple(float(piece) for piece in coords_raw.split("|") if piece != "")
        acc[coords] += _weight_fraction(weight_raw)
    out: list[MeasureTerm] = []
    for coords in sorted(acc):
        weight = acc[coords]
        if weight == 0:
            continue
        out.append(MeasureTerm(coords=coords, weight=weight))
    return out


def _parse_coord_tuple(raw: str) -> tuple[float, ...]:
    s = raw.strip()
    if s == "":
        return ()
    return tuple(float(piece) for piece in s.split("|") if piece != "")


def _parse_slice_barcode_records(raw: str) -> list[tuple[tuple[float, ...], tuple[float, ...], list[MeasureTerm]]]:
    s = raw.strip()
    if s == "":
        return []
    out: list[tuple[tuple[float, ...], tuple[float, ...], list[MeasureTerm]]] = []
    for idx, chunk in enumerate(s.split("###"), start=1):
        if chunk == "":
            continue
        parts = chunk.split("@@")
        if len(parts) != 3 or not parts[0].startswith("dir=") or not parts[1].startswith("off=") or not parts[2].startswith("bars="):
            raise ValueError(f"Malformed slice barcode record at index {idx}")
        direction = _parse_coord_tuple(parts[0][4:])
        offset = _parse_coord_tuple(parts[1][4:])
        bars = _parse_measure_terms(parts[2][5:])
        out.append((direction, offset, bars))
    return out


def _merge_measure_terms(terms: list[MeasureTerm]) -> list[MeasureTerm]:
    acc: dict[tuple[float, ...], Fraction] = defaultdict(Fraction)
    for term in terms:
        acc[term.coords] += term.weight
    out: list[MeasureTerm] = []
    for coords in sorted(acc):
        weight = acc[coords]
        if weight == 0:
            continue
        out.append(MeasureTerm(coords=coords, weight=weight))
    return out


def _parse_rank_axes(raw: str) -> list[tuple[float, ...]]:
    s = raw.strip()
    if s == "":
        return []
    axes: list[tuple[float, ...]] = []
    for axis_raw in s.split(";;"):
        pieces = [piece for piece in axis_raw.split("|") if piece != ""]
        axes.append(tuple(float(piece) for piece in pieces))
    return axes


def _parse_rank_table(raw: str) -> dict[tuple[tuple[float, ...], tuple[float, ...]], Fraction]:
    s = raw.strip()
    if s == "":
        return {}
    out: dict[tuple[tuple[float, ...], tuple[float, ...]], Fraction] = defaultdict(Fraction)
    for chunk in s.split(";"):
        if chunk == "":
            continue
        coords_raw, weight_raw = chunk.rsplit("=>", 1)
        p_raw, q_raw = coords_raw.split("||", 1)
        p = tuple(float(piece) for piece in p_raw.split("|") if piece != "")
        q = tuple(float(piece) for piece in q_raw.split("|") if piece != "")
        out[(p, q)] += _weight_fraction(weight_raw)
    return {key: weight for key, weight in out.items() if weight != 0}


def _intersect_rank_axis(
    t_axis: tuple[float, ...],
    m_axis: tuple[float, ...],
    tol: float,
) -> list[tuple[float, float]]:
    i = 0
    j = 0
    out: list[tuple[float, float]] = []
    while i < len(t_axis) and j < len(m_axis):
        tv = t_axis[i]
        mv = m_axis[j]
        if abs(tv - mv) <= tol or (math.isinf(tv) and math.isinf(mv) and tv == mv):
            out.append((tv, mv))
            i += 1
            j += 1
        elif tv < mv - tol:
            i += 1
        else:
            j += 1
    return out


def _shared_rank_query_axes(
    t_axes: list[tuple[float, ...]],
    m_axes: list[tuple[float, ...]],
    tol: float,
) -> tuple[list[list[tuple[float, float]]], str | None]:
    if not t_axes or not m_axes:
        return [], "rank_query_axes_missing"
    if len(t_axes) != len(m_axes):
        return [], f"rank_query_dim_mismatch:{len(t_axes)}!={len(m_axes)}"

    shared: list[list[tuple[float, float]]] = []
    for dim, (t_axis, m_axis) in enumerate(zip(t_axes, m_axes, strict=True), start=1):
        axis_shared = _intersect_rank_axis(t_axis, m_axis, tol)
        if not axis_shared:
            return [], f"rank_query_axis_intersection_empty@dim={dim}"
        shared.append(axis_shared)
    return shared, None


def _compare_rank_tables(
    t_axes_raw: str,
    t_table_raw: str,
    m_axes_raw: str,
    m_table_raw: str,
    tol: float = 1e-8,
) -> tuple[str, str]:
    t_axes = _parse_rank_axes(t_axes_raw)
    m_axes = _parse_rank_axes(m_axes_raw)
    shared_axes, err = _shared_rank_query_axes(t_axes, m_axes, tol)
    if err is not None:
        return "mismatched", err

    t_table = _parse_rank_table(t_table_raw)
    m_table = _parse_rank_table(m_table_raw)
    ndim = len(shared_axes)

    def _first_mismatch() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], Fraction, Fraction] | None:
        def _walk(
            depth: int,
            p_t: list[float],
            p_m: list[float],
            q_t: list[float],
            q_m: list[float],
        ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], Fraction, Fraction] | None:
            if depth == ndim:
                pt = tuple(p_t)
                pm = tuple(p_m)
                qt = tuple(q_t)
                qm = tuple(q_m)
                tv = t_table.get((pt, qt), Fraction(0, 1))
                mv = m_table.get((pm, qm), Fraction(0, 1))
                if tv != mv:
                    return pt, qt, pm, qm, tv, mv
                return None

            axis_shared = shared_axes[depth]
            for p_idx, (p_tv, p_mv) in enumerate(axis_shared):
                p_t.append(p_tv)
                p_m.append(p_mv)
                for q_tv, q_mv in axis_shared[p_idx:]:
                    q_t.append(q_tv)
                    q_m.append(q_mv)
                    mismatch = _walk(depth + 1, p_t, p_m, q_t, q_m)
                    q_t.pop()
                    q_m.pop()
                    if mismatch is not None:
                        return mismatch
                p_t.pop()
                p_m.pop()
            return None

        return _walk(0, [], [], [], [])

    mismatch = _first_mismatch()
    if mismatch is None:
        return "matched", "matched_rank_table_shared_grid"

    pt, qt, pm, qm, tv, mv = mismatch
    pt_tok = "|".join(repr(float(x)) for x in pt)
    qt_tok = "|".join(repr(float(x)) for x in qt)
    pm_tok = "|".join(repr(float(x)) for x in pm)
    qm_tok = "|".join(repr(float(x)) for x in qm)
    return (
        "mismatched",
        "rank_table_mismatch@"
        f"tamer_p={pt_tok};tamer_q={qt_tok};multipers_p={pm_tok};multipers_q={qm_tok};"
        f"tamer={_weight_token(tv)};multipers={_weight_token(mv)}",
    )


def _normalize_trailing_neg_inf_dims(
    t_terms: list[MeasureTerm],
    m_terms: list[MeasureTerm],
) -> tuple[list[MeasureTerm], list[MeasureTerm]]:
    if not t_terms or not m_terms:
        return t_terms, m_terms

    t_dims = {len(term.coords) for term in t_terms}
    m_dims = {len(term.coords) for term in m_terms}
    if len(t_dims) != 1 or len(m_dims) != 1:
        return t_terms, m_terms

    t_dim = next(iter(t_dims))
    m_dim = next(iter(m_dims))
    if t_dim == m_dim:
        return t_terms, m_terms

    if t_dim < m_dim:
        small_terms, big_terms = t_terms, m_terms
        strip_big = "multipers"
        target_dim = t_dim
    else:
        small_terms, big_terms = m_terms, t_terms
        strip_big = "tamer"
        target_dim = m_dim

    def _strip_ok(coords: tuple[float, ...]) -> bool:
        return all(math.isinf(x) and x < 0.0 for x in coords[target_dim:])

    if not all(_strip_ok(term.coords) for term in big_terms):
        return t_terms, m_terms

    stripped_big = _merge_measure_terms(
        [
            MeasureTerm(coords=term.coords[:target_dim], weight=term.weight)
            for term in big_terms
        ]
    )
    if strip_big == "multipers":
        return t_terms, stripped_big
    return stripped_big, m_terms


def _compare_measure_terms(tamer_raw: str, multipers_raw: str, tol: float = 1e-8) -> tuple[str, str]:
    t_terms = _parse_measure_terms(tamer_raw)
    m_terms = _parse_measure_terms(multipers_raw)
    t_terms, m_terms = _normalize_trailing_neg_inf_dims(t_terms, m_terms)
    if len(t_terms) != len(m_terms):
        return "mismatched", f"term_count_diff:{len(t_terms)}!={len(m_terms)}"
    for i, (tt, mt) in enumerate(zip(t_terms, m_terms, strict=True), start=1):
        if len(tt.coords) != len(mt.coords):
            return "mismatched", f"ambient_dim_diff@{i}:{len(tt.coords)}!={len(mt.coords)}"
        if any(abs(a - b) > tol for a, b in zip(tt.coords, mt.coords, strict=True)):
            return "mismatched", f"coordinate_mismatch@{i}"
        if tt.weight != mt.weight:
            return "mismatched", f"weight_mismatch@{i}:{_weight_token(tt.weight)}!={_weight_token(mt.weight)}"
    return "matched", "matched"


def _compare_slice_barcodes(tamer_raw: str, multipers_raw: str, tol: float = 1e-8) -> tuple[str, str]:
    t_records = _parse_slice_barcode_records(tamer_raw)
    m_records = _parse_slice_barcode_records(multipers_raw)
    if len(t_records) != len(m_records):
        return "mismatched", f"slice_count_diff:{len(t_records)}!={len(m_records)}"
    for idx, (t_rec, m_rec) in enumerate(zip(t_records, m_records, strict=True), start=1):
        t_dir, t_off, t_bars = t_rec
        m_dir, m_off, m_bars = m_rec
        if len(t_dir) != len(m_dir):
            return "mismatched", f"slice_direction_dim_diff@{idx}:{len(t_dir)}!={len(m_dir)}"
        if len(t_off) != len(m_off):
            return "mismatched", f"slice_offset_dim_diff@{idx}:{len(t_off)}!={len(m_off)}"
        if any(abs(a - b) > tol for a, b in zip(t_dir, m_dir, strict=True)):
            return "mismatched", f"slice_direction_mismatch@{idx}"
        if any(abs(a - b) > tol for a, b in zip(t_off, m_off, strict=True)):
            return "mismatched", f"slice_offset_mismatch@{idx}"
        if len(t_bars) != len(m_bars):
            return "mismatched", f"barcode_term_count_diff@slice={idx}:{len(t_bars)}!={len(m_bars)}"
        for j, (tb, mb) in enumerate(zip(t_bars, m_bars, strict=True), start=1):
            if len(tb.coords) != len(mb.coords):
                return "mismatched", f"barcode_dim_diff@slice={idx};term={j}:{len(tb.coords)}!={len(mb.coords)}"
            if any(abs(a - b) > tol for a, b in zip(tb.coords, mb.coords, strict=True)):
                return "mismatched", f"barcode_endpoint_mismatch@slice={idx};term={j}"
            if tb.weight != mb.weight:
                return "mismatched", f"barcode_mult_mismatch@slice={idx};term={j}:{_weight_token(tb.weight)}!={_weight_token(mb.weight)}"
    return "matched", "matched_slice_barcodes"


def _compare_outputs(
    invariant_kind: str,
    tamer_row: dict[str, Any],
    multipers_row: dict[str, Any],
    tol: float = 1e-8,
) -> tuple[str, str]:
    if invariant_kind == "rank_signed_measure":
        return _compare_rank_tables(
            tamer_row.get("output_rank_query_axes_canonical", ""),
            tamer_row.get("output_rank_table_canonical", ""),
            multipers_row.get("output_rank_query_axes_canonical", ""),
            multipers_row.get("output_rank_table_canonical", ""),
            tol,
        )
    if invariant_kind == "slice_barcodes":
        return _compare_slice_barcodes(
            tamer_row.get("output_measure_canonical", ""),
            multipers_row.get("output_measure_canonical", ""),
            tol,
        )
    return _compare_measure_terms(
        tamer_row.get("output_measure_canonical", ""),
        multipers_row.get("output_measure_canonical", ""),
        tol,
    )


def _failure_row(
    *,
    case_id: str,
    regime: str,
    invariant_kind: str,
    degree: str,
    failure_stage: str,
    failure_reason: str,
    tamer_status: str,
    multipers_status: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "regime": regime,
        "invariant_kind": invariant_kind,
        "degree": degree,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "tamer_status": tamer_status,
        "multipers_status": multipers_status,
    }


def _ingest_runner_failures(path: Path | None, tool: str) -> list[dict[str, Any]]:
    rows = _read_optional_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            _failure_row(
                case_id=row.get("case_id", ""),
                regime=row.get("regime", ""),
                invariant_kind=row.get("invariant_kind", ""),
                degree=row.get("degree", ""),
                failure_stage=f"{tool}_runner",
                failure_reason=row.get("failure_reason", row.get("run_status", "runner_failure")),
                tamer_status=row.get("run_status", "") if tool == "tamer" else "missing",
                multipers_status=row.get("run_status", "") if tool == "multipers" else "missing",
            )
        )
    return out


def main() -> None:
    _configure_csv_field_limit()
    p = argparse.ArgumentParser(description="Compare tamer-op vs multipers end-to-end invariant CSV outputs.")
    p.add_argument("--tamer", type=Path, default=Path(__file__).with_name("results_tamer_invariants.csv"))
    p.add_argument("--multipers", type=Path, default=Path(__file__).with_name("results_multipers_invariants.csv"))
    p.add_argument("--tamer_failures", type=Path, default=None)
    p.add_argument("--multipers_failures", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("comparison_invariants.csv"))
    p.add_argument("--summary_out", type=Path, default=Path(__file__).with_name("comparison_summary_invariants.csv"))
    p.add_argument("--failures_out", type=Path, default=Path(__file__).with_name("comparison_failures_invariants.csv"))
    args = p.parse_args()

    t_rows = _read_csv(args.tamer)
    m_rows = _read_csv(args.multipers)
    keyf = lambda r: (r["case_id"], r["regime"], r["invariant_kind"], r["degree"])
    t_map = {keyf(r): r for r in t_rows}
    m_map = {keyf(r): r for r in m_rows}

    joined: list[dict[str, Any]] = []
    failures = []
    failures.extend(_ingest_runner_failures(args.tamer_failures, "tamer"))
    failures.extend(_ingest_runner_failures(args.multipers_failures, "multipers"))

    matched_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    matched_by_group_cold: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    all_keys = sorted(set(t_map) | set(m_map))
    if not all_keys and not failures:
        raise RuntimeError("No invariant benchmark rows were available for comparison.")

    for key in all_keys:
        tr = t_map.get(key)
        mr = m_map.get(key)
        case_id, regime, invariant_kind, degree = key
        if tr is None:
            failures.append(
                _failure_row(
                    case_id=case_id,
                    regime=regime,
                    invariant_kind=invariant_kind,
                    degree=degree,
                    failure_stage="comparison",
                    failure_reason="missing_tamer_result",
                    tamer_status="missing",
                    multipers_status="ok",
                )
            )
            continue
        if mr is None:
            failures.append(
                _failure_row(
                    case_id=case_id,
                    regime=regime,
                    invariant_kind=invariant_kind,
                    degree=degree,
                    failure_stage="comparison",
                    failure_reason="missing_multipers_result",
                    tamer_status="ok",
                    multipers_status="missing",
                )
            )
            continue

        parity_status, parity_reason = _compare_outputs(invariant_kind, tr, mr)

        t_warm = _f(tr.get("warm_median_ms"))
        m_warm = _f(mr.get("warm_median_ms"))
        t_cold = _f(tr.get("cold_ms"))
        m_cold = _f(mr.get("cold_ms"))
        warm_slowdown = t_warm / m_warm if math.isfinite(t_warm) and math.isfinite(m_warm) and m_warm > 0 else float("nan")
        cold_slowdown = t_cold / m_cold if math.isfinite(t_cold) and math.isfinite(m_cold) and m_cold > 0 else float("nan")

        if parity_status == "matched":
            group = (regime, invariant_kind, degree)
            matched_by_group[group].append(warm_slowdown)
            matched_by_group_cold[group].append(cold_slowdown)
            summary_included = "yes"
        else:
            summary_included = "no"
            failures.append(
                _failure_row(
                    case_id=case_id,
                    regime=regime,
                    invariant_kind=invariant_kind,
                    degree=degree,
                    failure_stage="parity",
                    failure_reason=parity_reason,
                    tamer_status="ok",
                    multipers_status="ok",
                )
            )

        joined.append(
            {
                "case_id": case_id,
                "regime": regime,
                "invariant_kind": invariant_kind,
                "degree": degree,
                "n_points": tr.get("n_points", ""),
                "ambient_dim": tr.get("ambient_dim", ""),
                "max_dim": tr.get("max_dim", ""),
                "tamer_cold_ms": tr.get("cold_ms", ""),
                "multipers_cold_ms": mr.get("cold_ms", ""),
                "cold_slowdown_tamer_over_multipers": f"{cold_slowdown:.6f}" if math.isfinite(cold_slowdown) else "",
                "tamer_warm_median_ms": tr.get("warm_median_ms", ""),
                "multipers_warm_median_ms": mr.get("warm_median_ms", ""),
                "warm_slowdown_tamer_over_multipers": f"{warm_slowdown:.6f}" if math.isfinite(warm_slowdown) else "",
                "tamer_output_term_count": tr.get("output_term_count", ""),
                "multipers_output_term_count": mr.get("output_term_count", ""),
                "tamer_output_abs_mass": tr.get("output_abs_mass", ""),
                "multipers_output_abs_mass": mr.get("output_abs_mass", ""),
                "parity_status": parity_status,
                "parity_reason": parity_reason,
                "summary_included": summary_included,
            }
        )

    summary: list[dict[str, Any]] = []
    for regime, invariant_kind, degree in sorted(matched_by_group):
        warm_g = _geom_mean(matched_by_group[(regime, invariant_kind, degree)])
        cold_g = _geom_mean(matched_by_group_cold[(regime, invariant_kind, degree)])
        summary.append(
            {
                "regime": regime,
                "invariant_kind": invariant_kind,
                "degree": degree,
                "n_cases": len(matched_by_group[(regime, invariant_kind, degree)]),
                "warm_geomean_slowdown_tamer_over_multipers": f"{warm_g:.6f}" if math.isfinite(warm_g) else "",
                "cold_geomean_slowdown_tamer_over_multipers": f"{cold_g:.6f}" if math.isfinite(cold_g) else "",
            }
        )

    _write_rows(
        args.out,
        joined,
        [
            "case_id",
            "regime",
            "invariant_kind",
            "degree",
            "n_points",
            "ambient_dim",
            "max_dim",
            "tamer_cold_ms",
            "multipers_cold_ms",
            "cold_slowdown_tamer_over_multipers",
            "tamer_warm_median_ms",
            "multipers_warm_median_ms",
            "warm_slowdown_tamer_over_multipers",
            "tamer_output_term_count",
            "multipers_output_term_count",
            "tamer_output_abs_mass",
            "multipers_output_abs_mass",
            "parity_status",
            "parity_reason",
            "summary_included",
        ],
    )
    _write_rows(
        args.summary_out,
        summary,
        [
            "regime",
            "invariant_kind",
            "degree",
            "n_cases",
            "warm_geomean_slowdown_tamer_over_multipers",
            "cold_geomean_slowdown_tamer_over_multipers",
        ],
    )
    _write_rows(
        args.failures_out,
        failures,
        [
            "case_id",
            "regime",
            "invariant_kind",
            "degree",
            "failure_stage",
            "failure_reason",
            "tamer_status",
            "multipers_status",
        ],
    )

    print(f"Wrote case-level invariant comparison: {args.out}")
    print(f"Wrote summary invariant comparison: {args.summary_out}")
    print(f"Wrote invariant comparison failures: {args.failures_out}")
    print("")
    for row in summary:
        print(
            f"{row['regime']:20s} inv={row['invariant_kind']:22s} degree={row['degree']} "
            f"warm_geomean={row['warm_geomean_slowdown_tamer_over_multipers']} "
            f"cold_geomean={row['cold_geomean_slowdown_tamer_over_multipers']} n={row['n_cases']}"
        )


if __name__ == "__main__":
    main()
