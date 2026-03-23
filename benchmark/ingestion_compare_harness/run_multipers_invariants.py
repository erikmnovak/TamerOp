#!/usr/bin/env python3
"""
Run end-to-end invariant benchmarks for multipers on deterministic fixtures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import math
import statistics
import time
from pathlib import Path
from typing import Any
import tomllib

import numpy as np


def _profile_defaults(profile: str) -> dict[str, int | bool]:
    profile = profile.lower()
    if profile == "desktop":
        return {"reps": 4, "trim_between_reps": True, "trim_between_cases": True}
    if profile == "balanced":
        return {"reps": 5, "trim_between_reps": False, "trim_between_cases": True}
    if profile == "stress":
        return {"reps": 9, "trim_between_reps": False, "trim_between_cases": False}
    if profile == "probe":
        return {"reps": 3, "trim_between_reps": False, "trim_between_cases": True}
    raise ValueError("--profile must be one of: desktop, balanced, stress, probe")


def _memory_relief() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _try_import_multipers():
    try:
        import gudhi as gd  # type: ignore
        import multipers as mp  # type: ignore
        import multipers.filtrations as mf  # type: ignore
        import multipers.ml.point_clouds as mmp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "multipers is not available. Install it in the active Python environment."
        ) from exc
    return mmp, mp, gd, mf


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    return raw


def _load_case_data(case: dict[str, Any], fixture: Path) -> np.ndarray:
    dataset = str(case["dataset"])
    data = np.loadtxt(fixture, delimiter=",", dtype=np.float64)
    if dataset == "gaussian_shell":
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    if dataset == "image_sine":
        if data.ndim != 2:
            raise RuntimeError(f"image_sine fixture must be 2D matrix, got shape={data.shape}.")
        return data
    raise RuntimeError(f"Unsupported dataset: {dataset}")


def _build_multipers_filtered_complex(mmp, mp, gd, mf, data: np.ndarray, case: dict[str, Any]):
    regime = str(case["regime"])
    max_dim = int(case["max_dim"])
    points = data
    if regime == "degree_rips_parity":
        degree_radius = float(case["degree_radius"])
        try:
            st = mf.DegreeRips(points=points, threshold_radius=degree_radius)
            return st, {"builder": "filtrations.DegreeRips", "threshold_radius": degree_radius, "max_dim": max_dim}
        except NotImplementedError as exc:
            raise RuntimeError(
                "filtrations.DegreeRips is unavailable in the local multipers build; "
                "lower-star emulation is disabled for this parity benchmark."
            ) from exc
    if regime == "rips_parity":
        if "parity_radius" in case:
            parity_radius = float(case["parity_radius"])
            st0 = gd.RipsComplex(points=points, max_edge_length=parity_radius).create_simplex_tree(
                max_dimension=max_dim
            )
            st = mp.simplex_tree_multi.SimplexTreeMulti(st0, num_parameters=1, safe_conversion=False)
            return st, {
                "builder": "gudhi_rips_radius",
                "radius": parity_radius,
                "max_dim": max_dim,
            }
        common = {
            "complex": "rips",
            "output_type": "simplextree",
            "expand_dim": max_dim,
        }
        tries = [
            {
                **common,
                "num_collapses": 0,
            },
            {
                **common,
            },
        ]
        last_err: Exception | None = None
        for kw in tries:
            try:
                est = mmp.PointCloud2FilteredComplex(**kw)
                out = est.fit_transform([points])
                if len(out) == 0 or len(out[0]) == 0:
                    raise RuntimeError("PointCloud2FilteredComplex returned empty output.")
                st = out[0][0]
                return st, kw
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise RuntimeError("All Rips constructor attempts failed.") from last_err
    if regime == "rips_codensity_parity":
        codensity_radius = float(case["codensity_radius"])
        dtm_mass_val = float(case["codensity_dtm_mass"])
        st = mf.RipsCodensity(points, dtm_mass=dtm_mass_val, threshold_radius=codensity_radius)
        return st, {
            "builder": "filtrations.RipsCodensity",
            "threshold_radius": codensity_radius,
            "dtm_mass": dtm_mass_val,
            "max_dim": max_dim,
        }
    if regime == "rips_lowerstar_parity":
        lowerstar_radius = float(case["lowerstar_radius"])
        fvals = np.asarray(points[:, 0], dtype=np.float64)
        st = mf.RipsLowerstar(points=points, function=fvals, threshold_radius=lowerstar_radius)
        return st, {
            "builder": "filtrations.RipsLowerstar",
            "threshold_radius": lowerstar_radius,
            "function": "coord1",
            "max_dim": max_dim,
        }
    if regime == "landmark_parity":
        if "landmarks" not in case:
            raise RuntimeError("landmark_parity case requires landmarks in manifest.")
        if "landmark_radius" not in case:
            raise RuntimeError("landmark_parity case requires landmark_radius in manifest.")
        landmarks_1 = np.asarray(case["landmarks"], dtype=np.int64)
        if landmarks_1.size == 0:
            raise RuntimeError("landmark_parity landmarks cannot be empty.")
        idx0 = landmarks_1 - 1
        if idx0.min() < 0 or idx0.max() >= points.shape[0]:
            raise RuntimeError("landmark_parity landmarks are out of bounds for fixture point cloud.")
        sub = points[idx0, :]
        radius = float(case["landmark_radius"])
        st0 = gd.RipsComplex(points=sub, max_edge_length=radius).create_simplex_tree(max_dimension=max_dim)
        st = mp.simplex_tree_multi.SimplexTreeMulti(st0, num_parameters=1, safe_conversion=False)
        return st, {
            "builder": "gudhi_landmark_rips_radius",
            "radius": radius,
            "max_dim": max_dim,
            "n_landmarks": int(sub.shape[0]),
        }
    if regime == "delaunay_lowerstar_parity":
        fvals = np.asarray(points[:, 0], dtype=np.float64)
        try:
            st = mf.DelaunayLowerstar(points, fvals, verbose=False)
        except (AssertionError, NotImplementedError) as exc:
            msg = str(exc)
            if "function_delaunay" in msg or "DelaunayLowerstar" in msg:
                raise RuntimeError(
                    "filtrations.DelaunayLowerstar is unavailable in the local multipers build; "
                    "Delaunay lower-star invariant parity is disabled."
                ) from exc
            raise
        return st, {
            "builder": "filtrations.DelaunayLowerstar",
            "function": "coord1",
            "max_dim": max_dim,
        }
    if regime == "alpha_parity":
        tries = [
            {
                "complex": "alpha",
                "output_type": "slicer_novine",
                "expand_dim": max_dim,
                "num_collapses": 0,
                "safe_conversion": False,
            },
            {
                "complex": "alpha",
                "output_type": "slicer",
                "expand_dim": max_dim,
                "num_collapses": 0,
                "safe_conversion": False,
            },
        ]
        last_err: Exception | None = None
        for kw in tries:
            try:
                est = mmp.PointCloud2FilteredComplex(**kw)
                out = est.fit_transform([points])
                if len(out) == 0 or len(out[0]) == 0:
                    raise RuntimeError("PointCloud2FilteredComplex returned empty output.")
                st = out[0][0]
                return st, kw
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise RuntimeError("All alpha constructor attempts failed.") from last_err
    if regime == "core_delaunay_parity":
        ks = case.get("core_ks")
        ks_vals = None if ks is None else [int(v) for v in ks]
        st = mf.CoreDelaunay(points, ks=ks_vals, precision="safe", verbose=False)
        st.prune_above_dimension(max_dim)
        return st, {"builder": "filtrations.CoreDelaunay", "max_dim": max_dim, "ks": ks_vals}
    if regime == "cubical_parity":
        image = data
        if image.ndim != 2:
            raise RuntimeError(f"cubical_parity expects 2D image fixture, got shape={image.shape}")
        slicer = mf.Cubical(image[:, :, None])
        return slicer, {"builder": "filtrations.Cubical", "image_shape": tuple(int(v) for v in image.shape)}
    raise RuntimeError(f"Unsupported regime for invariant benchmark: {regime}")


def _parse_invariants(raw: str) -> list[str]:
    s = raw.strip().lower()
    if s in ("", "all"):
        return []
    out: list[str] = []
    for part in s.split(","):
        token = part.strip().lower().replace("-", "_")
        if token in ("rank", "rank_signed_measure"):
            out.append("rank_signed_measure")
        elif token in ("slice", "slice_barcodes"):
            out.append("slice_barcodes")
        elif token in ("euler", "euler_signed_measure"):
            out.append("euler_signed_measure")
        else:
            raise ValueError(f"--invariants contains unsupported token {part}")
    if not out:
        raise ValueError("--invariants selected no invariants.")
    return list(dict.fromkeys(out))


def _approved_invariants_for_regime(manifest: dict[str, Any], regime: str) -> list[str] | None:
    tbl = manifest.get("invariant_eligibility")
    if tbl is None:
        return None
    if not isinstance(tbl, dict):
        raise ValueError("manifest invariant_eligibility must be a table.")
    vals = tbl.get(regime)
    if vals is None:
        return None
    if not isinstance(vals, list):
        raise ValueError(f"manifest invariant_eligibility[{regime!r}] must be an array.")
    out: list[str] = []
    for v in vals:
        token = str(v).strip().lower().replace("-", "_")
        if token in ("rank", "rank_signed_measure"):
            out.append("rank_signed_measure")
        elif token in ("slice", "slice_barcodes"):
            out.append("slice_barcodes")
        elif token in ("euler", "euler_signed_measure"):
            out.append("euler_signed_measure")
        else:
            raise ValueError(f"Unsupported manifest-approved invariant token {v!r} for regime {regime!r}")
    return list(dict.fromkeys(out))


def _selected_invariants_for_case(requested: list[str], approved: list[str] | None) -> list[str]:
    requested_all = len(requested) == 0
    if approved is None:
        return ["euler_signed_measure"] if requested_all else requested
    if requested_all:
        return approved
    approved_set = set(approved)
    return [inv for inv in requested if inv in approved_set]


def _weight_token(weight: Any) -> str:
    if isinstance(weight, (int, np.integer)):
        return str(int(weight))
    if isinstance(weight, (float, np.floating)):
        f = float(weight)
        if math.isfinite(f) and f.is_integer():
            return str(int(round(f)))
        return format(f, ".17g")
    return str(weight)


def _coord_token(x: float) -> str:
    return format(float(x), ".17g")


def _coord_key_tuple(coords: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return "|".join(_coord_token(float(x)) for x in coords)


def _serialize_rank_axes(axes: tuple[np.ndarray, ...] | list[np.ndarray]) -> str:
    return ";;".join(_coord_key_tuple(axis) for axis in axes)


def _serialize_rank_table_from_measure(mp, axes: tuple[np.ndarray, ...] | list[np.ndarray], res) -> str:
    measure = _extract_measure_payload(res)
    ndim = len(axes)
    dims = tuple(len(axis) for axis in axes)
    entries: list[str] = []

    def _walk(depth: int, p_idx: list[int], q_idx: list[int]) -> None:
        if depth == ndim:
            p = np.asarray([float(axes[k][p_idx[k]]) for k in range(ndim)], dtype=np.float64)
            q = np.asarray([float(axes[k][q_idx[k]]) for k in range(ndim)], dtype=np.float64)
            val = mp.point_measure.estimate_rank_from_rank_sm(measure, p, q)
            if int(val) == 0:
                return
            entries.append(
                _coord_key_tuple(tuple(p.tolist()))
                + "||"
                + _coord_key_tuple(tuple(q.tolist()))
                + "=>"
                + _weight_token(val)
            )
            return

        for pi in range(dims[depth]):
            p_idx.append(pi)
            for qi in range(pi, dims[depth]):
                q_idx.append(qi)
                _walk(depth + 1, p_idx, q_idx)
                q_idx.pop()
            p_idx.pop()

    _walk(0, [], [])
    return ";".join(entries)


def _slice_query_from_case(case: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if "slice_directions" not in case or "slice_offsets" not in case:
        raise RuntimeError("slice_barcodes requires slice_directions and slice_offsets in the manifest.")
    directions = [np.asarray(row, dtype=np.float64) for row in case["slice_directions"]]
    offsets = [np.asarray(row, dtype=np.float64) for row in case["slice_offsets"]]
    if not directions or not offsets:
        raise RuntimeError("slice_barcodes requires nonempty slice_directions and slice_offsets.")
    return directions, offsets


def _canonical_barcode_terms(intervals: np.ndarray) -> list[tuple[tuple[float, float], int]]:
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    agg: dict[tuple[float, float], int] = {}
    for birth, death in arr.tolist():
        key = (float(birth), float(death))
        agg[key] = agg.get(key, 0) + 1
    out: list[tuple[tuple[float, float], int]] = []
    for key in sorted(agg, key=lambda t: (_coord_token(t[0]), _coord_token(t[1]))):
        mult = agg[key]
        if mult == 0:
            continue
        out.append((key, mult))
    return out


def _serialize_barcode_terms(terms: list[tuple[tuple[float, float], int]]) -> str:
    return ";".join(
        _coord_token(interval[0]) + "|" + _coord_token(interval[1]) + "=>" + _weight_token(mult)
        for interval, mult in terms
    )


def _slice_output_contract(res: dict[str, Any]) -> tuple[int, float, str]:
    records: list[str] = []
    term_count = 0
    abs_mass = 0.0
    directions = res["directions"]
    offsets = res["offsets"]
    barcodes = res["barcodes"]
    for i, direction in enumerate(directions):
        for j, offset in enumerate(offsets):
            terms = _canonical_barcode_terms(barcodes[i][j])
            term_count += len(terms)
            abs_mass += sum(abs(float(mult)) for _, mult in terms)
            records.append(
                "dir="
                + _coord_key_tuple(tuple(direction.tolist()))
                + "@@off="
                + _coord_key_tuple(tuple(offset.tolist()))
                + "@@bars="
                + _serialize_barcode_terms(terms)
            )
    return term_count, abs_mass, "###".join(records)


def _extract_measure_payload(res):
    if isinstance(res, tuple):
        if len(res) == 0:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)
        if len(res) == 2 and not isinstance(res[0], tuple):
            return res
        return res[0]
    if isinstance(res, list):
        if len(res) == 0:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)
        return res[0]
    raise TypeError(f"Unsupported signed-measure payload type: {type(res)!r}")


def _canonical_measure_terms(res) -> list[tuple[tuple[float, ...], str]]:
    locs, weights = _extract_measure_payload(res)
    weights_arr = np.asarray(weights)
    if weights_arr.size == 0:
        return []
    locs_arr = np.asarray(locs, dtype=np.float64)
    if locs_arr.ndim == 1:
        if weights_arr.size == 1:
            locs_arr = locs_arr.reshape(1, -1)
        else:
            locs_arr = locs_arr.reshape(-1, 1)
    elif locs_arr.ndim != 2:
        locs_arr = np.reshape(locs_arr, (weights_arr.size, -1))
    if locs_arr.shape[0] != weights_arr.size:
        raise RuntimeError(
            f"Signed-measure location/weight shape mismatch: locs={locs_arr.shape}, weights={weights_arr.shape}"
        )
    agg: dict[tuple[float, ...], Any] = {}
    for row, weight in zip(locs_arr.tolist(), weights_arr.tolist(), strict=True):
        key = tuple(float(x) for x in row)
        agg[key] = agg.get(key, 0) + weight
    out: list[tuple[tuple[float, ...], str]] = []
    for key in sorted(agg):
        weight = agg[key]
        if weight == 0:
            continue
        out.append((key, _weight_token(weight)))
    return out


def _serialize_measure_terms(terms: list[tuple[tuple[float, ...], str]]) -> str:
    return ";".join("|".join(_coord_token(x) for x in coords) + "=>" + weight for coords, weight in terms)


def _timed_call(f):
    t0 = time.perf_counter()
    out = f()
    t1 = time.perf_counter()
    return out, 1000.0 * (t1 - t0)


def _p90(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = max(0, int(np.ceil(0.9 * len(s))) - 1)
    return s[idx]


def _native_rank_query_axes(filtered_complex, mp) -> tuple[np.ndarray, ...]:
    if getattr(filtered_complex, "is_squeezed", False):
        grid = getattr(filtered_complex, "filtration_grid", None)
        if grid is None or len(grid) == 0:
            raise RuntimeError("Squeezed multipers complex is missing filtration_grid for rank query contract.")
        return tuple(np.asarray(axis, dtype=np.float64) for axis in grid)
    grid = mp.grids.compute_grid(filtered_complex, strategy="exact")
    return tuple(np.asarray(axis, dtype=np.float64) for axis in grid)


def _run_invariant_uncached(mmp, mp, gd, mf, data: np.ndarray, case: dict[str, Any], invariant_kind: str, degree: int):
    filtered_complex, meta = _build_multipers_filtered_complex(mmp, mp, gd, mf, data, case)
    if (
        invariant_kind == "euler_signed_measure"
        and str(case["regime"]) == "core_delaunay_parity"
        and getattr(filtered_complex, "is_kcritical", False)
    ):
        raise RuntimeError(
            "multipers signed_measure does not support Euler on CoreDelaunay k-critical outputs "
            "in the local build."
        )
    if invariant_kind == "rank_signed_measure":
        rank_query_axes = _native_rank_query_axes(filtered_complex, mp)
        res = mp.signed_measure(
            filtered_complex,
            degree=degree,
            invariant="rank",
            plot=False,
            verbose=False,
            grid=rank_query_axes,
        )
        meta = {**meta, "rank_query_axes": rank_query_axes}
    elif invariant_kind == "slice_barcodes":
        directions, offsets = _slice_query_from_case(case)
        slicer = mp.Slicer(filtered_complex)
        barcodes: list[list[np.ndarray]] = []
        for direction in directions:
            row: list[np.ndarray] = []
            for offset in offsets:
                raw_bars = slicer.persistence_on_line(offset, direction=direction, full=False)
                if degree < len(raw_bars):
                    arr = np.asarray(raw_bars[degree], dtype=np.float64)
                else:
                    arr = np.empty((0, 2), dtype=np.float64)
                if arr.size == 0:
                    arr = np.empty((0, 2), dtype=np.float64)
                else:
                    arr = arr.reshape(-1, 2)
                row.append(arr)
            barcodes.append(row)
        res = {"directions": directions, "offsets": offsets, "barcodes": barcodes}
    elif invariant_kind == "euler_signed_measure":
        res = mp.signed_measure(
            filtered_complex,
            degree=None,
            invariant="euler",
            plot=False,
            verbose=False,
        )
    else:
        raise RuntimeError(f"Unsupported invariant_kind={invariant_kind}")
    return res, meta


def _extract_measure_stats(res) -> tuple[int, float]:
    terms = _canonical_measure_terms(res)
    abs_mass = 0.0
    for _, weight in terms:
        if "//" in weight:
            num, den = weight.split("//", 1)
            abs_mass += abs(int(num)) / abs(int(den))
        else:
            abs_mass += abs(float(weight))
    return len(terms), abs_mass


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
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
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Run end-to-end invariant benchmark for multipers.")
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("fixtures_invariants") / "manifest.toml")
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results_multipers_invariants.csv"))
    p.add_argument("--profile", default="desktop")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--regime", default="all")
    p.add_argument("--case", default="")
    p.add_argument("--invariants", default="all")
    p.add_argument("--degree", type=int, default=0)
    args = p.parse_args()

    defaults = _profile_defaults(args.profile)
    reps = defaults["reps"] if args.reps is None else args.reps
    if reps < 1:
        raise ValueError("--reps must be >= 1.")

    invariants = _parse_invariants(args.invariants)
    print(f"[profile] {args.profile} (reps={reps}, trim_between_reps={defaults['trim_between_reps']}, trim_between_cases={defaults['trim_between_cases']})")
    print(f"Invariants: {'all' if not invariants else invariants}")

    mmp, mp, gd, mf = _try_import_multipers()
    manifest = _load_manifest(args.manifest)
    cases = manifest["cases"]
    rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["id"])
        regime = str(case["regime"])
        if args.regime != "all" and regime != args.regime:
            continue
        if args.case and case_id != args.case:
            continue

        approved_invariants = _approved_invariants_for_regime(manifest, regime)
        selected_invariants = _selected_invariants_for_case(invariants, approved_invariants)
        if not selected_invariants:
            print(f"[skip] {case_id}: no benchmark-approved invariants for regime={regime}")
            continue

        fixture = args.manifest.parent / str(case["path"])
        n_points = int(case["n_points"])
        ambient_dim = int(case["ambient_dim"])
        max_dim = int(case["max_dim"])
        data = _load_case_data(case, fixture)

        for invariant_kind in selected_invariants:
            if invariant_kind == "slice_barcodes":
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_slice_barcodes"
            else:
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_signed_measure"
            try:
                _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
            except Exception as exc:
                print(f"[skip] {case_id} {invariant_kind}: {exc}")
                continue

            _memory_relief()
            (cold_res, meta), cold_ms = _timed_call(
                lambda: _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
            )
            notes = notes + f";builder={meta.get('builder', 'unknown')}"
            if invariant_kind == "slice_barcodes":
                output_term_count, output_abs_mass, output_measure_canonical = _slice_output_contract(cold_res)
                output_rank_query_axes_canonical = ""
                output_rank_table_canonical = ""
            else:
                output_term_count, output_abs_mass = _extract_measure_stats(cold_res)
                output_measure_canonical = _serialize_measure_terms(_canonical_measure_terms(cold_res))
                if invariant_kind == "rank_signed_measure":
                    rank_query_axes = meta["rank_query_axes"]
                    output_rank_query_axes_canonical = _serialize_rank_axes(rank_query_axes)
                    output_rank_table_canonical = _serialize_rank_table_from_measure(mp, rank_query_axes, cold_res)
                else:
                    output_rank_query_axes_canonical = ""
                    output_rank_table_canonical = ""

            warm_times: list[float] = []
            for _ in range(reps):
                _, tms = _timed_call(
                    lambda: _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
                )
                warm_times.append(tms)
                if defaults["trim_between_reps"]:
                    _memory_relief()

            warm_median_ms = statistics.median(warm_times)
            warm_p90_ms = _p90(warm_times)

            print(
                f"{case_id:<32} inv={invariant_kind} cold_ms={cold_ms:.3f} "
                f"warm_med_ms={warm_median_ms:.3f} terms={output_term_count}"
            )

            rows.append(
                {
                    "tool": "multipers",
                    "case_id": case_id,
                    "regime": regime,
                    "invariant_kind": invariant_kind,
                    "degree": args.degree,
                    "n_points": n_points,
                    "ambient_dim": ambient_dim,
                    "max_dim": max_dim,
                    "cold_ms": cold_ms,
                    "cold_alloc_kib": "",
                    "warm_median_ms": warm_median_ms,
                    "warm_p90_ms": warm_p90_ms,
                    "warm_alloc_median_kib": "",
                    "output_term_count": output_term_count,
                    "output_abs_mass": output_abs_mass,
                    "output_measure_canonical": output_measure_canonical,
                    "output_rank_query_axes_canonical": output_rank_query_axes_canonical,
                    "output_rank_table_canonical": output_rank_table_canonical,
                    "notes": notes,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )

            if defaults["trim_between_cases"]:
                _memory_relief()

    if not rows:
        raise RuntimeError("No invariant benchmark rows were produced.")
    _write_rows(args.out, rows)
    print(f"Wrote multipers invariant results: {args.out}")


if __name__ == "__main__":
    main()
