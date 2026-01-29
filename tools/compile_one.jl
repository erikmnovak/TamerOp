#!/usr/bin/env julia
# tools/compile_one.jl
#
# Load a single project file (and only the minimal prerequisites it needs)
# into a fresh sandbox module to verify it parses/evaluates cleanly.

using Printf

const ROOT = abspath(joinpath(@__DIR__, ".."))

# Minimal per-file prerequisite mapping inferred from your sources.
# You can edit this list if you add new files or change dependencies.
const DEPS = Dict(
    "CoreModules.jl"            => String[],                         # base definitions (QQ, flags)
    "FiniteFringe.jl"           => String[],                         # poset + fringe types
    "IndicatorTypes.jl"         => ["FiniteFringe.jl"],              # uses ..FiniteFringe (Upset/Downset)  [IndicatorTypes]
    "ExactQQ.jl"                => ["CoreModules.jl"],               # uses ..CoreModules: QQ               [ExactQQ]
    "Encoding.jl"               => ["FiniteFringe.jl"],              # uses ..FiniteFringe                  [Encoding]
    "HomExt.jl"                 => ["CoreModules.jl","FiniteFringe.jl","IndicatorTypes.jl","ExactQQ.jl"],
    "IndicatorResolutions.jl"   => ["CoreModules.jl","FiniteFringe.jl","IndicatorTypes.jl","ExactQQ.jl","HomExt.jl"],
    "FlangeZn.jl"               => ["CoreModules.jl","FiniteFringe.jl"],   # conservative (Face/Flange live here)
    "CrossValidateFlangePL.jl"  => ["CoreModules.jl","ExactQQ.jl","FlangeZn.jl"],
    "Serialization.jl"          => ["CoreModules.jl","FiniteFringe.jl","FlangeZn.jl"],
    "PLPolyhedra.jl"            => ["CoreModules.jl","FiniteFringe.jl"],   # optional Polyhedra/CDDLib guarded   [PLPolyhedra]
    "M2SingularBridge.jl"       => ["CoreModules.jl","FlangeZn.jl"],
    "Viz2D.jl"                  => ["CoreModules.jl","ExactQQ.jl","FlangeZn.jl"],
    "PLBackend.jl"              => ["CoreModules.jl","FiniteFringe.jl","ExactQQ.jl"],  # conservative guess
    "PosetModules.jl"           => String[]                          # aggregator (if present)
)

# Targets: either everything known, or the list provided on the CLI.
targets = isempty(ARGS) ? sort(collect(keys(DEPS))) : collect(ARGS)

# Verify all targets are recognized
unknown = filter(t -> !haskey(DEPS, t), targets)
if !isempty(unknown)
    @printf("Unknown target(s): %s\n", join(unknown, ", "))
    exit(2)
end

exitcode = 0

for fname in targets
    fpath = joinpath(ROOT, fname)
    if !isfile(fpath)
        @printf("[SKIP] %s (file not found at %s)\n", fname, fpath)
        continue
    end
    try
        # Create a fresh parent so relative imports like `using ..CoreModules` resolve
        # to CompileSandbox.CoreModules, etc., without polluting Main.
        sandbox = Module(:CompileSandbox)

        # Load prerequisites first (in this same parent).
        for dep in DEPS[fname]
            include(sandbox, joinpath(ROOT, dep))
        end

        # Now load the file under test.
        include(sandbox, fpath)
        @printf("[OK]   %s\n", fname)
    catch e
        @printf("[FAIL] %s\n", fname)
        showerror(stdout, e, catch_backtrace())
        println()
        exitcode = 1
    end
end

exit(exitcode)
