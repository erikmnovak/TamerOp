# =============================================================================
# shared_types.jl
#
# Shared invariant-side record types and tiny cross-family adapters.
#
# Owns:
# - `SliceSpec`
# - compiled-encoding unwrapping helper
#
# Does not own:
# - invariant algorithms
# - slice barcode / distance kernels
# - invariant API wrappers
# =============================================================================

function chain end
function weight end

"""
    SliceSpec{W,V}
    SliceSpec(chain; values=nothing, weight=1.0)

Typed explicit slice descriptor used in high-throughput slice pipelines.

Mathematically, a `SliceSpec` records one weighted slice through a finite
encoding:
- `chain` is the ordered list of region/vertex labels visited by the slice,
- `values` records the scalar parameter values attached to those chain entries,
- `weight` is the nonnegative weight attached to this slice when it is used in
  aggregate slice statistics.

Conventions:
- `values === nothing` means "index mode": downstream code uses endpoint indices
  rather than explicit scalar slice values.
- integer `values` are appropriate when the slice is recorded against a
  discrete index axis,
- real `values` are appropriate when the slice should carry explicit geometric
  parameter values such as `t` coordinates.

Expected invariants:
- `chain` stores positive integer labels,
- if `values !== nothing`, it should have the same length as `chain`,
- `weight` should be finite and nonnegative.

Use [`check_slice_spec`](@ref) when hand-building slice specs in notebooks or
tests and you want an explicit validation report rather than a downstream shape
error.

# Examples

```jldoctest
julia> using TamerOp

julia> spec = TamerOp.InvariantCore.SliceSpec([1, 3, 4];
                                                  values=[0.0, 0.5, 1.0],
                                                  weight=0.25);

julia> TamerOp.describe(spec).values_mode
:real

julia> TamerOp.InvariantCore.check_slice_spec(spec).valid
true
```
"""
struct SliceSpec{W<:Real,V}
    chain::Vector{Int}
    values::V
    weight::W
end

function SliceSpec(chain::AbstractVector{<:Integer}; values=nothing, weight::Real=1.0)
    c = chain isa Vector{Int} ? copy(chain) : Int[chain...]
    w = float(weight)
    if values === nothing
        return SliceSpec{Float64,Nothing}(c, nothing, w)
    elseif values isa AbstractVector{<:Integer}
        return SliceSpec{Float64,Vector{Int}}(c, Int[values...], w)
    elseif values isa AbstractVector{<:Real}
        return SliceSpec{Float64,Vector{Float64}}(c, Float64[float(v) for v in values], w)
    end
    throw(ArgumentError("SliceSpec: values must be nothing, integer vector, or real vector"))
end

@inline chain(spec::SliceSpec) = spec.chain
@inline values(spec::SliceSpec) = spec.values
@inline weight(spec::SliceSpec) = spec.weight

@inline function _slice_values_mode(values)
    values === nothing && return :index
    values isa AbstractVector{<:Integer} && return :integer
    values isa AbstractVector{<:Real} && return :real
    return :unsupported
end

@inline function _slice_spec_describe(spec::SliceSpec)
    vals = values(spec)
    ch = chain(spec)
    return (;
        kind=:slice_spec,
        chain_length=length(ch),
        values_mode=_slice_values_mode(vals),
        values_length=vals === nothing ? nothing : length(vals),
        weight=float(weight(spec)),
        first_label=isempty(ch) ? nothing : first(ch),
        last_label=isempty(ch) ? nothing : last(ch),
    )
end

"""
    describe(spec::SliceSpec) -> NamedTuple

Return a compact semantic summary of a typed slice specification.

This is the preferred lightweight inspection surface for `SliceSpec` in the
REPL, notebooks, and targeted invariant tests.
"""
describe(spec::SliceSpec) = _slice_spec_describe(spec)

"""
    check_slice_spec(spec; throw=false) -> NamedTuple

Validate the local structural contract of a [`SliceSpec`](@ref).

This helper checks only the shape and scalar-value conventions visible from the
spec itself; it does not try to validate that `chain` is a chain in some
ambient poset.
"""
function check_slice_spec(spec::SliceSpec; throw::Bool=false)
    issues = String[]
    ch = chain(spec)
    vals = values(spec)
    w = float(weight(spec))

    any(x -> x < 1, ch) && push!(issues, "chain contains labels < 1.")
    if vals !== nothing && length(vals) != length(ch)
        push!(issues, "values length $(length(vals)) does not match chain length $(length(ch)).")
    end
    if vals !== nothing
        mode = _slice_values_mode(vals)
        mode === :unsupported && push!(issues, "values must be an integer vector, a real vector, or nothing.")
        if vals isa AbstractVector{<:Real}
            any(v -> !isfinite(float(v)), vals) && push!(issues, "values contain non-finite entries.")
        end
    end
    isfinite(w) || push!(issues, "weight must be finite.")
    w >= 0 || push!(issues, "weight must be nonnegative.")

    report = (;
        kind=:slice_spec,
        valid=isempty(issues),
        chain_length=length(ch),
        values_mode=_slice_values_mode(vals),
        values_length=vals === nothing ? nothing : length(vals),
        weight=w,
        issues=issues,
    )
    if throw && !report.valid
        Base.throw(ArgumentError("check_slice_spec: " * join(report.issues, " ")))
    end
    return report
end

function show(io::IO, spec::SliceSpec)
    d = describe(spec)
    print(io, "SliceSpec(length=", d.chain_length,
          ", values=", d.values_mode,
          ", weight=", d.weight, ")")
end

function show(io::IO, ::MIME"text/plain", spec::SliceSpec)
    d = describe(spec)
    println(io, "SliceSpec")
    println(io, "  chain_length: ", d.chain_length)
    println(io, "  values_mode: ", d.values_mode)
    println(io, "  values_length: ", d.values_length)
    println(io, "  weight: ", d.weight)
    println(io, "  first_label: ", d.first_label)
    println(io, "  last_label: ", d.last_label)
end

@inline _unwrap_compiled(pi) = (pi isa CompiledEncoding ? pi.pi : pi)
