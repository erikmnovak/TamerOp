#!/usr/bin/env python3
"""
Generate reproducible point-cloud fixtures for ingestion comparison.

Outputs:
- CSV point-cloud files (one case per file)
- fixtures manifest TOML with resolved paths and effective sizes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
import tomllib
import numpy as np


def _load_contract(path: Path) -> dict[str, Any]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    return raw


def _gaussian_shell(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.2 * rng.standard_normal((n, d), dtype=np.float64)
    if d >= 2:
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
        radius = 1.0 + 0.06 * rng.standard_normal(size=n)
        x[:, 0] = radius * np.cos(theta) + 0.02 * rng.standard_normal(size=n)
        x[:, 1] = radius * np.sin(theta) + 0.02 * rng.standard_normal(size=n)
    return x


def _image_sine(side: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 2.0 * math.pi, side, dtype=np.float64)
    ys = np.linspace(0.0, 2.0 * math.pi, side, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    img = 0.6 * np.sin(1.7 * xx) + 0.45 * np.cos(1.3 * yy) + 0.08 * rng.standard_normal((side, side))
    return img.astype(np.float64)


def _build_points(dataset: str, n: int, d: int, seed: int) -> np.ndarray:
    if dataset == "gaussian_shell":
        return _gaussian_shell(n, d, seed)
    raise ValueError(f"Unsupported dataset kind: {dataset}")


def _estimate_claim_radius(points: np.ndarray, k: int, seed: int) -> float:
    """Estimate a shared radius so both tools run the same radius-threshold policy."""
    n = points.shape[0]
    if n <= 1:
        return 0.0
    k_eff = max(1, min(k, n - 1))
    rng = np.random.default_rng(seed + 7919)
    sample_n = min(n, 256)
    idx = rng.choice(n, size=sample_n, replace=False)
    kth = np.empty(sample_n, dtype=np.float64)
    for t, i in enumerate(idx):
        d2 = np.sum((points - points[i]) ** 2, axis=1)
        d2[i] = np.inf
        kth[t] = float(np.sqrt(np.partition(d2, k_eff - 1)[k_eff - 1]))
    return float(np.median(kth))


def _deterministic_landmarks(n: int, m: int, seed: int) -> list[int]:
    if n <= 0:
        return []
    m_eff = max(1, min(m, n))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=m_eff, replace=False))
    # Store 1-based indices for direct Julia consumption.
    return [int(i) + 1 for i in idx]


def _format_float(x: float) -> str:
    return format(float(x), ".17g")


def _format_float_vec(xs: list[float]) -> str:
    return "[" + ", ".join(_format_float(x) for x in xs) + "]"


def _format_float_mat(rows: list[list[float]]) -> str:
    return "[" + ", ".join(_format_float_vec(row) for row in rows) + "]"


def _slice_query_for_case(case: dict[str, Any], points: np.ndarray) -> tuple[list[list[float]], list[list[float]]] | None:
    regime = str(case["regime"])
    if regime == "rips_parity":
        radius = float(case["parity_radius"])
        directions = [[1.0]]
        offsets = [[0.0], [0.25 * radius], [0.5 * radius]]
        return directions, offsets
    if regime == "rips_lowerstar_parity":
        radius = float(case["lowerstar_radius"])
        coord1 = np.asarray(points[:, 0], dtype=np.float64)
        ylo = float(np.min(coord1))
        directions = [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
        offsets = [[0.0, ylo], [0.25 * radius, ylo], [0.5 * radius, ylo]]
        return directions, offsets
    return None


def _manifest_toml(
    meta: dict[str, Any],
    cases: list[dict[str, Any]],
    invariant_eligibility: dict[str, list[str]] | None = None,
) -> str:
    out: list[str] = []
    out.append("[meta]")
    out.append(f'version = {int(meta["version"])}')
    out.append(f'cases_file = "{meta["cases_file"]}"')
    out.append(f'scale = {meta["scale"]}')
    out.append("")
    if invariant_eligibility:
        out.append("[invariant_eligibility]")
        for regime in sorted(invariant_eligibility):
            vals = ", ".join(f'"{v}"' for v in invariant_eligibility[regime])
            out.append(f"{regime} = [{vals}]")
        out.append("")
    for c in cases:
        out.append("[[cases]]")
        out.append(f'id = "{c["id"]}"')
        out.append(f'regime = "{c["regime"]}"')
        out.append(f'dataset = "{c["dataset"]}"')
        out.append(f"path = \"{c['path']}\"")
        out.append(f'n_points = {int(c["n_points"])}')
        out.append(f'ambient_dim = {int(c["ambient_dim"])}')
        out.append(f'seed = {int(c["seed"])}')
        out.append(f'max_dim = {int(c["max_dim"])}')
        if "claim_policy" in c:
            out.append(f'claim_policy = "{c["claim_policy"]}"')
        if "claim_k" in c:
            out.append(f'claim_k = {int(c["claim_k"])}')
        if "claim_radius" in c:
            out.append(f'claim_radius = {float(c["claim_radius"]):.12g}')
        if "parity_policy" in c:
            out.append(f'parity_policy = "{c["parity_policy"]}"')
        if "parity_k" in c:
            out.append(f'parity_k = {int(c["parity_k"])}')
        if "parity_radius" in c:
            out.append(f'parity_radius = {float(c["parity_radius"]):.12g}')
        if "degree_policy" in c:
            out.append(f'degree_policy = "{c["degree_policy"]}"')
        if "degree_k" in c:
            out.append(f'degree_k = {int(c["degree_k"])}')
        if "degree_radius" in c:
            out.append(f'degree_radius = {float(c["degree_radius"]):.12g}')
        if "codensity_policy" in c:
            out.append(f'codensity_policy = "{c["codensity_policy"]}"')
        if "codensity_k" in c:
            out.append(f'codensity_k = {int(c["codensity_k"])}')
        if "codensity_radius" in c:
            out.append(f'codensity_radius = {float(c["codensity_radius"]):.12g}')
        if "codensity_dtm_mass" in c:
            out.append(f'codensity_dtm_mass = {float(c["codensity_dtm_mass"]):.12g}')
        if "lowerstar_policy" in c:
            out.append(f'lowerstar_policy = "{c["lowerstar_policy"]}"')
        if "lowerstar_k" in c:
            out.append(f'lowerstar_k = {int(c["lowerstar_k"])}')
        if "lowerstar_radius" in c:
            out.append(f'lowerstar_radius = {float(c["lowerstar_radius"]):.12g}')
        if "landmark_policy" in c:
            out.append(f'landmark_policy = "{c["landmark_policy"]}"')
        if "landmark_k" in c:
            out.append(f'landmark_k = {int(c["landmark_k"])}')
        if "landmark_radius" in c:
            out.append(f'landmark_radius = {float(c["landmark_radius"]):.12g}')
        if "landmarks" in c:
            lms = ", ".join(str(int(x)) for x in c["landmarks"])
            out.append(f"landmarks = [{lms}]")
        if "slice_directions" in c:
            out.append(f'slice_directions = {_format_float_mat(c["slice_directions"])}')
        if "slice_offsets" in c:
            out.append(f'slice_offsets = {_format_float_mat(c["slice_offsets"])}')
        if "image_side" in c:
            out.append(f'image_side = {int(c["image_side"])}')
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ingestion comparison fixtures.")
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.toml"))
    p.add_argument("--out_dir", type=Path, default=Path(__file__).with_name("fixtures"))
    p.add_argument("--scale", type=float, default=1.0, help="Scale case n_points by this factor.")
    p.add_argument("--force", action="store_true", help="Overwrite existing CSV fixtures.")
    args = p.parse_args()

    if args.scale <= 0.0:
        raise ValueError("--scale must be > 0.")

    contract = _load_contract(args.cases)
    cases = contract["cases"]
    raw_eligibility = contract.get("invariant_eligibility", {})
    if raw_eligibility is None:
        raw_eligibility = {}
    if not isinstance(raw_eligibility, dict):
        raise ValueError("invariant_eligibility must be a table mapping regime names to string arrays.")
    invariant_eligibility: dict[str, list[str]] = {}
    for regime, vals in raw_eligibility.items():
        if not isinstance(regime, str):
            raise ValueError("invariant_eligibility keys must be strings.")
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            raise ValueError(f"invariant_eligibility[{regime!r}] must be an array of strings.")
        invariant_eligibility[regime] = vals
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []
    for c in cases:
        cid = str(c["id"])
        dataset = str(c["dataset"])
        base_n = int(c["n_points"])
        d = int(c["ambient_dim"])
        seed = int(c["seed"])
        max_dim = int(c["max_dim"])
        regime = str(c["regime"])

        if dataset == "image_sine":
            base_side = int(c.get("image_side", max(8, int(round(math.sqrt(base_n))))))
            side = max(8, int(round(base_side * math.sqrt(args.scale))))
            x = _image_sine(side, seed)
            n_points = int(side * side)
            d = 2
            image_side = side
        else:
            n_points = max(2, int(round(float(base_n) * args.scale)))
            x = _build_points(dataset, n_points, d, seed)
            image_side = None

        rel = Path(f"{cid}.csv")
        dst = out_dir / rel
        if dst.exists() and not args.force:
            raise FileExistsError(f"{dst} already exists (use --force).")
        np.savetxt(dst, x, delimiter=",", fmt="%.8f")
        manifest_cases.append(
            {
                "id": cid,
                "regime": regime,
                "dataset": dataset,
                "path": rel.as_posix(),
                "n_points": n_points,
                "ambient_dim": d,
                "seed": seed,
                "max_dim": max_dim,
            }
        )
        if image_side is not None:
            manifest_cases[-1]["image_side"] = image_side
        if regime == "claim_matching":
            claim_k = 16
            claim_radius = _estimate_claim_radius(x, claim_k, seed)
            manifest_cases[-1]["claim_policy"] = "radius_k"
            manifest_cases[-1]["claim_k"] = claim_k
            manifest_cases[-1]["claim_radius"] = claim_radius
        elif regime == "rips_parity":
            # Force an explicit shared threshold policy for both tools so this
            # regime remains a cleaner algorithmic comparison.
            parity_k = 16
            parity_radius = _estimate_claim_radius(x, parity_k, seed + 104729)
            manifest_cases[-1]["parity_policy"] = "radius_k"
            manifest_cases[-1]["parity_k"] = parity_k
            manifest_cases[-1]["parity_radius"] = parity_radius
        elif regime == "degree_rips_parity":
            degree_k = 16
            degree_radius = _estimate_claim_radius(x, degree_k, seed + 1009)
            manifest_cases[-1]["degree_policy"] = "radius_k"
            manifest_cases[-1]["degree_k"] = degree_k
            manifest_cases[-1]["degree_radius"] = degree_radius
        elif regime == "rips_codensity_parity":
            codensity_k = 16
            codensity_radius = _estimate_claim_radius(x, codensity_k, seed + 2039)
            # multipers.RipsCodensity supports DTM mass; map k/N to keep parity knobs explicit.
            codensity_dtm_mass = min(0.5, max(1.0 / float(n_points), float(codensity_k) / float(n_points)))
            manifest_cases[-1]["codensity_policy"] = "radius_k+dtm_mass"
            manifest_cases[-1]["codensity_k"] = codensity_k
            manifest_cases[-1]["codensity_radius"] = codensity_radius
            manifest_cases[-1]["codensity_dtm_mass"] = codensity_dtm_mass
        elif regime == "rips_lowerstar_parity":
            lowerstar_k = 16
            lowerstar_radius = _estimate_claim_radius(x, lowerstar_k, seed + 3253)
            manifest_cases[-1]["lowerstar_policy"] = "radius_k+coord1_function"
            manifest_cases[-1]["lowerstar_k"] = lowerstar_k
            manifest_cases[-1]["lowerstar_radius"] = lowerstar_radius
        elif regime == "landmark_parity":
            # Shared landmark subset + radius-threshold policy on the subset.
            landmark_count = max(16, int(round(math.sqrt(n_points))))
            landmarks = _deterministic_landmarks(n_points, landmark_count, seed + 13007)
            x_land = x[np.array(landmarks, dtype=np.int64) - 1]
            landmark_k = 16
            landmark_radius = _estimate_claim_radius(x_land, landmark_k, seed + 15485863)
            manifest_cases[-1]["landmark_policy"] = "radius_k"
            manifest_cases[-1]["landmark_k"] = landmark_k
            manifest_cases[-1]["landmark_radius"] = landmark_radius
            manifest_cases[-1]["landmarks"] = landmarks

        slice_query = _slice_query_for_case(manifest_cases[-1], x)
        if slice_query is not None:
            directions, offsets = slice_query
            manifest_cases[-1]["slice_directions"] = directions
            manifest_cases[-1]["slice_offsets"] = offsets

    manifest = _manifest_toml(
        {
            "version": 1,
            "cases_file": args.cases.as_posix(),
            "scale": float(args.scale),
        },
        manifest_cases,
        invariant_eligibility,
    )
    manifest_path = out_dir / "manifest.toml"
    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Wrote fixtures: {out_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
