# Owned MPPI cache JSON formats.

# =============================================================================
# C) Invariant caches (MPPI)
# =============================================================================

# MPPI types live in `TamerOp.Invariants`. We intentionally do NOT import
# them here to avoid include-order constraints. Instead, we fetch the module
# lazily when the MPPI JSON functions are called.

@inline function _invariants_module()
    TO = parentmodule(@__MODULE__)
    isdefined(TO, :Invariants) || error("MPPI JSON: TamerOp.Invariants is not loaded.")
    return getfield(TO, :Invariants)
end

function _mpp_floatvec2(x)::Vector{Float64}
    length(x) == 2 || error("MPPI JSON: expected a length-2 vector")
    return Float64[Float64(x[1]), Float64(x[2])]
end

function _mpp_decomposition_to_dict(decomp)
    lines = Vector{Any}(undef, length(decomp.lines))
    for (i, ls) in enumerate(decomp.lines)
        lines[i] = Dict(
            "dir" => ls.dir,
            "off" => ls.off,
            "x0" => ls.x0,
            "omega" => ls.omega,
        )
    end

    summands = Vector{Any}(undef, length(decomp.summands))
    for k in 1:length(decomp.summands)
        segs = decomp.summands[k]
        arr = Vector{Any}(undef, length(segs))
        for j in 1:length(segs)
            (p, q, om) = segs[j]
            arr[j] = Dict("p" => p, "q" => q, "omega" => om)
        end
        summands[k] = arr
    end

    lo, hi = decomp.box

    return Dict(
        "kind" => "MPPDecomposition",
        "version" => 1,
        "lines" => lines,
        "summands" => summands,
        "weights" => decomp.weights,
        "box" => Dict("lo" => lo, "hi" => hi),
    )
end

function _mpp_decomposition_from_dict(obj; validation::Bool=true)
    if validation
        if !haskey(obj, "kind") || String(obj["kind"]) != "MPPDecomposition"
            error("MPPI JSON: expected kind == 'MPPDecomposition'")
        end
        version = haskey(obj, "version") ? Int(obj["version"]) : 0
        version == 1 || error("Unsupported MPPDecomposition JSON version: $(version). Expected 1.")
    end

    Inv = _invariants_module()
    LineSpec = getfield(Inv, :MPPLineSpec)
    Decomp = getfield(Inv, :MPPDecomposition)

    lines_obj = obj["lines"]
    lines = Vector{LineSpec}(undef, length(lines_obj))
    for (i, l) in enumerate(lines_obj)
        dir = _mpp_floatvec2(l["dir"])
        off = Float64(l["off"])
        x0 = _mpp_floatvec2(l["x0"])
        omega = Float64(l["omega"])
        lines[i] = LineSpec(dir, off, x0, omega)
    end

    summands_obj = obj["summands"]
    summands = Vector{Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}}(undef, length(summands_obj))
    for (k, s) in enumerate(summands_obj)
        segs = Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}(undef, length(s))
        for (j, seg) in enumerate(s)
            p = _mpp_floatvec2(seg["p"])
            q = _mpp_floatvec2(seg["q"])
            om = Float64(seg["omega"])
            segs[j] = (p, q, om)
        end
        summands[k] = segs
    end

    weights_obj = obj["weights"]
    weights = Float64[Float64(w) for w in weights_obj]

    box_obj = obj["box"]
    lo = _mpp_floatvec2(box_obj["lo"])
    hi = _mpp_floatvec2(box_obj["hi"])

    return Decomp(lines, summands, weights, (lo, hi))
end

function _mpp_image_to_dict(img; include_decomp::Bool=true)
    ny, nx = size(img.img)
    mat = Vector{Any}(undef, ny)
    for i in 1:ny
        mat[i] = [img.img[i, j] for j in 1:nx]
    end

    d = Dict(
        "kind" => "MPPImage",
        "version" => 1,
        "sigma" => img.sigma,
        "xgrid" => img.xgrid,
        "ygrid" => img.ygrid,
        "img" => mat,
    )

    if include_decomp
        d["decomp"] = _mpp_decomposition_to_dict(img.decomp)
    end

    return d
end

function _mpp_image_from_dict(obj; validation::Bool=true)
    if validation
        if !haskey(obj, "kind") || String(obj["kind"]) != "MPPImage"
            error("MPPI JSON: expected kind == 'MPPImage'")
        end
        version = haskey(obj, "version") ? Int(obj["version"]) : 0
        version == 1 || error("Unsupported MPPImage JSON version: $(version). Expected 1.")
    end

    Inv = _invariants_module()
    Image = getfield(Inv, :MPPImage)

    sig = Float64(obj["sigma"])
    xgrid = Float64[Float64(x) for x in obj["xgrid"]]
    ygrid = Float64[Float64(y) for y in obj["ygrid"]]

    rows = obj["img"]
    length(rows) == length(ygrid) || error("MPPI JSON: img row count does not match ygrid")
    imgmat = zeros(Float64, length(ygrid), length(xgrid))
    for i in 1:length(ygrid)
        row = rows[i]
        length(row) == length(xgrid) || error("MPPI JSON: img column count does not match xgrid")
        for j in 1:length(xgrid)
            imgmat[i, j] = Float64(row[j])
        end
    end

    haskey(obj, "decomp") || error("MPPI JSON: missing field 'decomp' (cannot reconstruct MPPImage without it)")
    decomp = _mpp_decomposition_from_dict(obj["decomp"]; validation=validation)

    return Image(xgrid, ygrid, imgmat, sig, decomp)
end

"""
    save_mpp_decomposition_json(path, decomp; profile=:compact, pretty=nothing)

Save an `MPPDecomposition` to the stable TamerOp-owned MPPI
decomposition-cache schema.

This is a good cache point: the decomposition contains the slice tracks and weights,
but not the full image grid. After loading, evaluate images via `mpp_image(decomp; ...)`.

Use [`mpp_decomposition_json_summary`](@ref) or [`inspect_json`](@ref) to
inspect an existing cache artifact cheaply, and use
[`check_mpp_decomposition_json`](@ref) when you need strict validation before
calling [`load_mpp_decomposition_json`](@ref).

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.

Returns `path`.
"""
function save_mpp_decomposition_json(path::AbstractString, decomp;
                                     profile::Symbol=:compact,
                                     pretty::Union{Nothing,Bool}=nothing)
    obj = _mpp_decomposition_to_dict(decomp)
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_mpp_decomposition_json(path; validation=:strict)

Load an `MPPDecomposition` written by [`save_mpp_decomposition_json`](@ref).

This is a strict owned-schema loader for the MPPI decomposition cache format.
Prefer [`mpp_decomposition_json_summary`](@ref) for cheap-first inspection and
[`check_mpp_decomposition_json`](@ref) when you need explicit schema validation
before loading.

Use `validation=:trusted` only for TamerOp-produced cache artifacts when
you want to skip the extra owned-schema version checks on load.
"""
function load_mpp_decomposition_json(path::AbstractString; validation::Symbol=:strict)
    obj = _json_read(path)
    return _mpp_decomposition_from_dict(obj; validation=_resolve_validation_mode(validation))
end

"""
    save_mpp_image_json(path, img; include_decomp=true, profile=:compact, pretty=nothing)

Save an `MPPImage` to the stable TamerOp-owned MPPI image-cache schema.

By default this includes the underlying decomposition so the artifact remains
self-contained on load. Use [`mpp_image_json_summary`](@ref) or
[`inspect_json`](@ref) to inspect an existing cache artifact cheaply, and use
[`check_mpp_image_json`](@ref) when you need strict validation before calling
[`load_mpp_image_json`](@ref).

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.

Returns `path`.
"""
function save_mpp_image_json(path::AbstractString, img;
                             include_decomp::Bool=true,
                             profile::Symbol=:compact,
                             pretty::Union{Nothing,Bool}=nothing)
    obj = _mpp_image_to_dict(img; include_decomp=include_decomp)
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_mpp_image_json(path; validation=:strict)

Load an `MPPImage` written by [`save_mpp_image_json`](@ref).

Note: `load_mpp_image_json` requires that the JSON contains a `"decomp"` field.

This is a strict owned-schema loader for the MPPI image-cache format. Prefer
[`mpp_image_json_summary`](@ref) for cheap-first inspection and
[`check_mpp_image_json`](@ref) when you need explicit schema validation before
loading.

Use `validation=:trusted` only for TamerOp-produced cache artifacts when
you want to skip the extra owned-schema version checks on load.
"""
function load_mpp_image_json(path::AbstractString; validation::Symbol=:strict)
    obj = _json_read(path)
    return _mpp_image_from_dict(obj; validation=_resolve_validation_mode(validation))
end

