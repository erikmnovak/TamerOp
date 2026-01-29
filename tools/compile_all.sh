#!/usr/bin/env bash
# tools/compile_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA_BIN="${JULIA:-julia}"
SCRIPT="$ROOT/tools/compile_one.jl"

FILES=(
  CoreModules.jl
  FiniteFringe.jl
  IndicatorTypes.jl
  ExactQQ.jl
  Encoding.jl
  HomExt.jl
  IndicatorResolutions.jl
  FlangeZn.jl
  CrossValidateFlangePL.jl
  Serialization.jl
  PLPolyhedra.jl
  M2SingularBridge.jl
  Viz2D.jl
  PLBackend.jl
  PosetModules.jl
)

ok=0
fail=0
for f in "${FILES[@]}"; do
  if "$JULIA_BIN" --project="$ROOT" -q "$SCRIPT" "$f"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
  fi
done

echo "Done. OK=$ok FAIL=$fail"
if [ "$fail" -gt 0 ]; then exit 1; fi
