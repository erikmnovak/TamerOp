#!/usr/bin/env python3
"""
Focused smoke tests for the end-to-end invariant benchmark harness.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "benchmark" / "ingestion_compare_harness"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class MultipersExtractionSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module(HARNESS_DIR / "run_multipers_invariants.py", "run_multipers_invariants_test")

    def test_tuple_return_shape(self) -> None:
        locs = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        weights = np.array([2, -3], dtype=np.int64)
        terms = self.mod._canonical_measure_terms((locs, weights))
        self.assertEqual(terms, [((0.0, 1.0), "2"), ((1.0, 2.0), "-3")])

    def test_list_return_shape(self) -> None:
        locs = np.array([[0.0], [1.0]], dtype=np.float64)
        weights = np.array([1.5, -1.5], dtype=np.float64)
        terms = self.mod._canonical_measure_terms([(locs, weights)])
        self.assertEqual(terms, [((0.0,), "1.5"), ((1.0,), "-1.5")])

    def test_empty_return_shape(self) -> None:
        terms = self.mod._canonical_measure_terms([])
        self.assertEqual(terms, [])


@unittest.skipUnless(importlib.util.find_spec("multipers") is not None, "multipers not installed")
class MultipersInvariantRouteSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module(HARNESS_DIR / "run_multipers_invariants.py", "run_multipers_invariants_route_test")
        cls.mmp, cls.mp, cls.gd, cls.mf = cls.mod._try_import_multipers()

    def test_alpha_route_uses_slicer_and_computes_euler(self) -> None:
        case = {"regime": "alpha_parity", "max_dim": 2}
        points = np.random.default_rng(0).normal(size=(12, 2))
        filtered_complex, meta = self.mod._build_multipers_filtered_complex(
            self.mmp, self.mp, self.gd, self.mf, points, case
        )
        self.assertIn("slicer", type(filtered_complex).__name__.lower())
        self.assertEqual(meta["output_type"], "slicer_novine")

        res, _ = self.mod._run_invariant_uncached(
            self.mmp, self.mp, self.gd, self.mf, points, case, "euler_signed_measure", 0
        )
        self.assertGreater(len(self.mod._canonical_measure_terms(res)), 0)

    def test_rips_parity_route_uses_shared_radius(self) -> None:
        points = np.random.default_rng(4).normal(size=(12, 2))
        case = {
            "regime": "rips_parity",
            "max_dim": 1,
            "parity_radius": 1.25,
        }
        filtered_complex, meta = self.mod._build_multipers_filtered_complex(
            self.mmp, self.mp, self.gd, self.mf, points, case
        )
        self.assertEqual(meta["builder"], "gudhi_rips_radius")
        self.assertEqual(meta["radius"], 1.25)

        res, _ = self.mod._run_invariant_uncached(
            self.mmp, self.mp, self.gd, self.mf, points, case, "euler_signed_measure", 0
        )
        self.assertGreater(len(self.mod._canonical_measure_terms(res)), 0)

    def test_rips_parity_slice_route_returns_barcodes(self) -> None:
        points = np.random.default_rng(5).normal(size=(12, 2))
        case = {
            "regime": "rips_parity",
            "max_dim": 1,
            "parity_radius": 1.25,
            "slice_directions": [[1.0]],
            "slice_offsets": [[0.0], [0.2]],
        }
        res, _ = self.mod._run_invariant_uncached(
            self.mmp, self.mp, self.gd, self.mf, points, case, "slice_barcodes", 0
        )
        self.assertEqual(len(res["directions"]), 1)
        self.assertEqual(len(res["offsets"]), 2)
        term_count, abs_mass, canonical = self.mod._slice_output_contract(res)
        self.assertGreater(term_count, 0)
        self.assertGreater(abs_mass, 0.0)
        self.assertIn("dir=", canonical)

    def test_landmark_route_uses_manifest_subset(self) -> None:
        points = np.random.default_rng(2).normal(size=(12, 2))
        case = {
            "regime": "landmark_parity",
            "max_dim": 1,
            "landmarks": [1, 4, 7, 11],
            "landmark_radius": 1.25,
        }
        filtered_complex, meta = self.mod._build_multipers_filtered_complex(
            self.mmp, self.mp, self.gd, self.mf, points, case
        )
        self.assertEqual(meta["builder"], "gudhi_landmark_rips_radius")
        self.assertEqual(meta["n_landmarks"], 4)

        res, _ = self.mod._run_invariant_uncached(
            self.mmp, self.mp, self.gd, self.mf, points, case, "euler_signed_measure", 0
        )
        self.assertGreater(len(self.mod._canonical_measure_terms(res)), 0)

    def test_delaunay_lowerstar_route_uses_filtration_and_computes_euler(self) -> None:
        case = {"regime": "delaunay_lowerstar_parity", "max_dim": 2}
        points = np.random.default_rng(3).normal(size=(18, 2))
        try:
            filtered_complex, meta = self.mod._build_multipers_filtered_complex(
                self.mmp, self.mp, self.gd, self.mf, points, case
            )
        except RuntimeError as exc:
            self.assertIn("DelaunayLowerstar", str(exc))
            return

        self.assertEqual(meta["builder"], "filtrations.DelaunayLowerstar")
        self.assertEqual(meta["function"], "coord1")

        res, _ = self.mod._run_invariant_uncached(
            self.mmp, self.mp, self.gd, self.mf, points, case, "euler_signed_measure", 0
        )
        self.assertGreater(len(self.mod._canonical_measure_terms(res)), 0)

    def test_core_delaunay_reports_kcritical_unsupported(self) -> None:
        case = {"regime": "core_delaunay_parity", "max_dim": 2}
        points = np.random.default_rng(1).normal(size=(12, 2))
        with self.assertRaisesRegex(RuntimeError, "k-critical"):
            self.mod._run_invariant_uncached(
                self.mmp, self.mp, self.gd, self.mf, points, case, "euler_signed_measure", 0
            )


class CompareInvariantSmoke(unittest.TestCase):
    def test_only_matched_rows_contribute_to_summary(self) -> None:
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
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tamer_csv = td_path / "tamer.csv"
            multipers_csv = td_path / "multipers.csv"
            out_csv = td_path / "comparison.csv"
            summary_csv = td_path / "summary.csv"
            failures_csv = td_path / "failures.csv"

            matched_tamer = "0=>1;1=>-1"
            matched_multipers = "0|-inf=>1;1|-inf=>-1"
            mismatched_tamer = "0|0=>2"
            mismatched_multipers = "0|0=>3"
            base = {
                "degree": "0",
                "n_points": "10",
                "ambient_dim": "2",
                "max_dim": "1",
                "cold_alloc_kib": "",
                "warm_p90_ms": "2.0",
                "warm_alloc_median_kib": "",
                "notes": "",
                "timestamp_utc": "2026-03-19T00:00:00+00:00",
            }
            _write_csv(
                tamer_csv,
                [
                    {
                        **base,
                        "tool": "tamer_op",
                        "case_id": "matched_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "4.0",
                        "warm_median_ms": "5.0",
                        "output_term_count": "2",
                        "output_abs_mass": "2.0",
                        "output_measure_canonical": matched_tamer,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    },
                    {
                        **base,
                        "tool": "tamer_op",
                        "case_id": "mismatched_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "6.0",
                        "warm_median_ms": "7.0",
                        "output_term_count": "1",
                        "output_abs_mass": "2.0",
                        "output_measure_canonical": mismatched_tamer,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    },
                ],
                cols,
            )
            _write_csv(
                multipers_csv,
                [
                    {
                        **base,
                        "tool": "multipers",
                        "case_id": "matched_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "2.0",
                        "warm_median_ms": "2.5",
                        "output_term_count": "2",
                        "output_abs_mass": "2.0",
                        "output_measure_canonical": matched_multipers,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    },
                    {
                        **base,
                        "tool": "multipers",
                        "case_id": "mismatched_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "3.0",
                        "warm_median_ms": "3.5",
                        "output_term_count": "1",
                        "output_abs_mass": "3.0",
                        "output_measure_canonical": mismatched_multipers,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    },
                ],
                cols,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "compare_invariants.py"),
                    "--tamer",
                    str(tamer_csv),
                    "--multipers",
                    str(multipers_csv),
                    "--out",
                    str(out_csv),
                    "--summary_out",
                    str(summary_csv),
                    "--failures_out",
                    str(failures_csv),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                comparison_rows = list(csv.DictReader(f))
            with summary_csv.open("r", encoding="utf-8", newline="") as f:
                summary_rows = list(csv.DictReader(f))
            with failures_csv.open("r", encoding="utf-8", newline="") as f:
                failure_rows = list(csv.DictReader(f))

            self.assertEqual(len(comparison_rows), 2)
            self.assertEqual(
                {row["case_id"]: row["parity_status"] for row in comparison_rows},
                {"matched_case": "matched", "mismatched_case": "mismatched"},
            )
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["n_cases"], "1")
            self.assertEqual(summary_rows[0]["regime"], "probe")
            self.assertEqual(summary_rows[0]["invariant_kind"], "euler_signed_measure")
            self.assertTrue(any(row["failure_stage"] == "parity" for row in failure_rows))

    def test_compare_handles_large_measure_fields(self) -> None:
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
        base = {
            "degree": "0",
            "n_points": "10",
            "ambient_dim": "2",
            "max_dim": "1",
            "cold_alloc_kib": "",
            "warm_p90_ms": "2.0",
            "warm_alloc_median_kib": "",
            "notes": "",
            "timestamp_utc": "2026-03-19T00:00:00+00:00",
        }
        large_measure = ";".join(f"{i}=>1" for i in range(30000))

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tamer_csv = td_path / "tamer.csv"
            multipers_csv = td_path / "multipers.csv"
            out_csv = td_path / "comparison.csv"
            summary_csv = td_path / "summary.csv"
            failures_csv = td_path / "failures.csv"

            _write_csv(
                tamer_csv,
                [
                    {
                        **base,
                        "tool": "tamer_op",
                        "case_id": "large_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "4.0",
                        "warm_median_ms": "5.0",
                        "output_term_count": "30000",
                        "output_abs_mass": "30000.0",
                        "output_measure_canonical": large_measure,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    }
                ],
                cols,
            )
            _write_csv(
                multipers_csv,
                [
                    {
                        **base,
                        "tool": "multipers",
                        "case_id": "large_case",
                        "regime": "probe",
                        "invariant_kind": "euler_signed_measure",
                        "cold_ms": "2.0",
                        "warm_median_ms": "2.5",
                        "output_term_count": "30000",
                        "output_abs_mass": "30000.0",
                        "output_measure_canonical": large_measure,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    }
                ],
                cols,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "compare_invariants.py"),
                    "--tamer",
                    str(tamer_csv),
                    "--multipers",
                    str(multipers_csv),
                    "--out",
                    str(out_csv),
                    "--summary_out",
                    str(summary_csv),
                    "--failures_out",
                    str(failures_csv),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                comparison_rows = list(csv.DictReader(f))
            self.assertEqual(len(comparison_rows), 1)
            self.assertEqual(comparison_rows[0]["parity_status"], "matched")

    def test_rank_compare_uses_common_query_grid(self) -> None:
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
        base = {
            "degree": "0",
            "n_points": "8",
            "ambient_dim": "2",
            "max_dim": "1",
            "cold_alloc_kib": "",
            "warm_p90_ms": "2.0",
            "warm_alloc_median_kib": "",
            "notes": "",
            "timestamp_utc": "2026-03-23T00:00:00+00:00",
        }
        tamer_rank = "0|0|0|1=>1"
        multipers_rank = "0|0|inf|inf=>7"
        rank_axes = "0|1;;0|1"
        rank_table = "0|0||0|0=>1;0|0||1|1=>1;1|1||1|1=>2"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tamer_csv = td_path / "tamer.csv"
            multipers_csv = td_path / "multipers.csv"
            out_csv = td_path / "comparison.csv"
            summary_csv = td_path / "summary.csv"
            failures_csv = td_path / "failures.csv"

            _write_csv(
                tamer_csv,
                [
                    {
                        **base,
                        "tool": "tamer_op",
                        "case_id": "rank_case",
                        "regime": "probe",
                        "invariant_kind": "rank_signed_measure",
                        "cold_ms": "4.0",
                        "warm_median_ms": "5.0",
                        "output_term_count": "12",
                        "output_abs_mass": "12.0",
                        "output_measure_canonical": tamer_rank,
                        "output_rank_query_axes_canonical": rank_axes,
                        "output_rank_table_canonical": rank_table,
                    }
                ],
                cols,
            )
            _write_csv(
                multipers_csv,
                [
                    {
                        **base,
                        "tool": "multipers",
                        "case_id": "rank_case",
                        "regime": "probe",
                        "invariant_kind": "rank_signed_measure",
                        "cold_ms": "2.0",
                        "warm_median_ms": "2.5",
                        "output_term_count": "8",
                        "output_abs_mass": "8.0",
                        "output_measure_canonical": multipers_rank,
                        "output_rank_query_axes_canonical": rank_axes,
                        "output_rank_table_canonical": rank_table,
                    }
                ],
                cols,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "compare_invariants.py"),
                    "--tamer",
                    str(tamer_csv),
                    "--multipers",
                    str(multipers_csv),
                    "--out",
                    str(out_csv),
                    "--summary_out",
                    str(summary_csv),
                    "--failures_out",
                    str(failures_csv),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                comparison_rows = list(csv.DictReader(f))

            self.assertEqual(len(comparison_rows), 1)
            self.assertEqual(comparison_rows[0]["parity_status"], "matched")
            self.assertEqual(comparison_rows[0]["parity_reason"], "matched_rank_table_shared_grid")

    def test_slice_barcodes_compare_matches_identical_records(self) -> None:
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
        base = {
            "degree": "0",
            "n_points": "12",
            "ambient_dim": "2",
            "max_dim": "1",
            "cold_alloc_kib": "",
            "warm_p90_ms": "2.0",
            "warm_alloc_median_kib": "",
            "notes": "",
            "timestamp_utc": "2026-03-23T00:00:00+00:00",
        }
        canonical = "dir=1@@off=0@@bars=0|1=>1;0|inf=>1###dir=1@@off=0.2@@bars=0|0.8=>1"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            tamer_csv = td_path / "tamer.csv"
            multipers_csv = td_path / "multipers.csv"
            out_csv = td_path / "comparison.csv"
            summary_csv = td_path / "summary.csv"
            failures_csv = td_path / "failures.csv"

            _write_csv(
                tamer_csv,
                [
                    {
                        **base,
                        "tool": "tamer_op",
                        "case_id": "slice_case",
                        "regime": "probe",
                        "invariant_kind": "slice_barcodes",
                        "cold_ms": "4.0",
                        "warm_median_ms": "5.0",
                        "output_term_count": "3",
                        "output_abs_mass": "3.0",
                        "output_measure_canonical": canonical,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    }
                ],
                cols,
            )
            _write_csv(
                multipers_csv,
                [
                    {
                        **base,
                        "tool": "multipers",
                        "case_id": "slice_case",
                        "regime": "probe",
                        "invariant_kind": "slice_barcodes",
                        "cold_ms": "2.0",
                        "warm_median_ms": "2.5",
                        "output_term_count": "3",
                        "output_abs_mass": "3.0",
                        "output_measure_canonical": canonical,
                        "output_rank_query_axes_canonical": "",
                        "output_rank_table_canonical": "",
                    }
                ],
                cols,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "compare_invariants.py"),
                    "--tamer",
                    str(tamer_csv),
                    "--multipers",
                    str(multipers_csv),
                    "--out",
                    str(out_csv),
                    "--summary_out",
                    str(summary_csv),
                    "--failures_out",
                    str(failures_csv),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                comparison_rows = list(csv.DictReader(f))

            self.assertEqual(len(comparison_rows), 1)
            self.assertEqual(comparison_rows[0]["parity_status"], "matched")
            self.assertEqual(comparison_rows[0]["parity_reason"], "matched_slice_barcodes")


@unittest.skipUnless(importlib.util.find_spec("multipers") is not None, "multipers not installed")
class SupervisorSmoke(unittest.TestCase):
    def test_supervisor_probe_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cases_path = td_path / "cases_probe.toml"
            fixtures_dir = td_path / "fixtures"
            work_dir = td_path / "work"
            tamer_out = td_path / "results_tamer.csv"
            multipers_out = td_path / "results_multipers.csv"
            comparison_out = td_path / "comparison.csv"
            summary_out = td_path / "summary.csv"
            failures_out = td_path / "comparison_failures.csv"

            cases_path.write_text(
                textwrap.dedent(
                    """
                    [meta]
                    version = 1
                    description = "Probe invariant cases for supervisor smoke."

                    [invariant_eligibility]
                    cubical_parity = ["euler_signed_measure"]
                    unsupported_probe = []

                    [[cases]]
                    id = "probe_cubical"
                    regime = "cubical_parity"
                    dataset = "image_sine"
                    n_points = 144
                    ambient_dim = 2
                    seed = 11
                    max_dim = 2
                    image_side = 12

                    [[cases]]
                    id = "probe_unsupported"
                    regime = "unsupported_probe"
                    dataset = "gaussian_shell"
                    n_points = 64
                    ambient_dim = 2
                    seed = 12
                    max_dim = 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "generate_fixtures.py"),
                    "--cases",
                    str(cases_path),
                    "--out_dir",
                    str(fixtures_dir),
                    "--scale",
                    "0.05",
                    "--force",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "run_invariant_supervisor.py"),
                    "--manifest",
                    str(fixtures_dir / "manifest.toml"),
                    "--tools",
                    "both",
                    "--profile",
                    "probe",
                    "--invariants",
                    "all",
                    "--degree",
                    "0",
                    "--work_dir",
                    str(work_dir),
                    "--tamer_out",
                    str(tamer_out),
                    "--multipers_out",
                    str(multipers_out),
                    "--comparison_out",
                    str(comparison_out),
                    "--summary_out",
                    str(summary_out),
                    "--comparison_failures_out",
                    str(failures_out),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            with tamer_out.open("r", encoding="utf-8", newline="") as f:
                tamer_rows = list(csv.DictReader(f))
            with multipers_out.open("r", encoding="utf-8", newline="") as f:
                multipers_rows = list(csv.DictReader(f))
            with failures_out.open("r", encoding="utf-8", newline="") as f:
                comparison_failures = list(csv.DictReader(f))

            self.assertTrue(any(row["case_id"] == "probe_cubical" for row in tamer_rows))
            self.assertTrue(any(row["case_id"] == "probe_cubical" for row in multipers_rows))
            self.assertFalse(any(row["case_id"] == "probe_unsupported" for row in tamer_rows))
            self.assertFalse(any(row["case_id"] == "probe_unsupported" for row in multipers_rows))
            self.assertFalse(any(row["case_id"] == "probe_unsupported" for row in comparison_failures))
            self.assertTrue(summary_out.exists())
            self.assertTrue(comparison_out.exists())


if __name__ == "__main__":
    unittest.main()
