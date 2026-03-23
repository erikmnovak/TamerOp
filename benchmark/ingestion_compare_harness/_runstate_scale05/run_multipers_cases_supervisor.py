from pathlib import Path
import subprocess, csv

cases = [
    "inv_degree_rips_n1000_d2_s11",
    "inv_degree_rips_n1000_d8_s11",
    "inv_rips_codensity_n750_d2_s11",
    "inv_rips_codensity_n750_d8_s11",
    "inv_rips_lowerstar_n1000_d2_s11",
    "inv_rips_lowerstar_n1000_d8_s11",
    "inv_alpha_n150_d2_md2_s11",
    "inv_core_delaunay_n150_d2_md2_s11",
    "inv_cubical_side24_s11",
]
base = Path("benchmark/ingestion_compare_harness")
outdir = base / "results_multipers_invariants_scale05_cases"
log = base / "_runstate_scale05" / "multipers_supervisor.log"
status = base / "_runstate_scale05" / "multipers_status.txt"
status.write_text("starting\n")
header = None
rows = []
with log.open('w') as lf:
    for i, case in enumerate(cases, 1):
        lf.write(f"=== {case} ({i}/{len(cases)}) ===\n")
        lf.flush()
        status.write_text(f"running {case} {i}/{len(cases)}\n")
        out = outdir / f"{case}.csv"
        cmd = [
            'python', 'benchmark/ingestion_compare_harness/run_multipers_invariants.py',
            '--manifest', 'benchmark/ingestion_compare_harness/fixtures_invariants_scale05/manifest.toml',
            '--out', str(out), '--profile', 'desktop', '--invariants', 'all', '--degree', '0', '--case', case,
        ]
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.')
        lf.write(f"[rc] {case} -> {proc.returncode}\n")
        lf.flush()
        if proc.returncode != 0:
            status.write_text(f"failed {case} rc={proc.returncode}\n")
            raise SystemExit(proc.returncode)
        with out.open() as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = reader.fieldnames
            rows.extend(reader)
    combo = base / 'results_multipers_invariants_scale05.csv'
    with combo.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    status.write_text('done\n')
