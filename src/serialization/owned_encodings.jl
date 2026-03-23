# Owned encoding-family JSON formats: flange, PL fringe, and finite encodings.

# -----------------------------------------------------------------------------
# A1) Flange (Z^n)  (FlangeZn.Flange)
# -----------------------------------------------------------------------------

"""
    save_flange_json(path, FG::FlangeZn.Flange; profile=:compact, pretty=nothing)

Stable TamerOp-owned schema:

{
  "kind": "FlangeZn",
  "n": n,
  "flats":      [ {"b":[...], "tau":[i1,i2,...]}, ... ],
  "injectives": [ {"b":[...], "tau":[...]} , ... ],
  "coeff_field": { ... },
  "phi": [[ "num/den", ...], ...]   # rows = #injectives, cols = #flats
}

Notes
* `tau` is stored as a list of 1-based coordinate indices where the face is true.
* Scalars are encoded according to the coefficient field descriptor.

This is the canonical owned write path for flange artifacts. Use
[`flange_json_summary`](@ref) or [`inspect_json`](@ref) to inspect an existing
artifact cheaply, and use [`check_flange_json`](@ref) when you need strict
schema validation before calling [`load_flange_json`](@ref).

The shared owned-artifact save profiles are:
- `:compact` (default): compact JSON output
- `:debug`: pretty-printed JSON for inspection/debugging
"""
function save_flange_json(path::AbstractString, FG::Flange;
                          profile::Symbol=:compact,
                          pretty::Union{Nothing,Bool}=nothing)
    n = FG.n
    flats = [Dict("b" => collect(F.b), "tau" => findall(identity, F.tau.coords)) for F in FG.flats]
    injectives = [Dict("b" => collect(E.b), "tau" => findall(identity, E.tau.coords)) for E in FG.injectives]
    phi = [[_scalar_to_json(FG.field, FG.phi[i, j]) for j in 1:length(FG.flats)]
           for i in 1:length(FG.injectives)]
    obj = Dict("kind" => "FlangeZn",
               "n" => n,
               "flats" => flats,
               "injectives" => injectives,
               "coeff_field" => _field_to_obj(FG.field),
               "phi" => phi)
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end


# -----------------------------------------------------------------------------
# A1b) PL fringe (R^n)  (PLPolyhedra.PLFringe)
# -----------------------------------------------------------------------------

@inline _qq_json(x::QQ) = _scalar_to_json(QQField(), x)
@inline _qq_parse(x) = _scalar_from_json(QQField(), x)

function _qq_vector_to_obj(v::AbstractVector{QQ})
    out = Vector{Any}(undef, length(v))
    @inbounds for i in eachindex(v)
        out[i] = _qq_json(v[i])
    end
    return out
end

function _qq_matrix_to_obj(A::AbstractMatrix{QQ})
    m, n = size(A)
    rows = Vector{Any}(undef, m)
    @inbounds for i in 1:m
        row = Vector{Any}(undef, n)
        for j in 1:n
            row[j] = _qq_json(A[i, j])
        end
        rows[i] = row
    end
    return rows
end

function _parse_qq_vector(v_any, label::String)
    v_any isa AbstractVector || error("$(label) must be a vector.")
    out = Vector{QQ}(undef, length(v_any))
    @inbounds for i in eachindex(v_any)
        out[i] = _qq_parse(v_any[i])
    end
    return out
end

function _parse_qq_matrix(A_any, label::String)
    A_any isa AbstractVector || error("$(label) must be a matrix encoded as row vectors.")
    m = length(A_any)
    if m == 0
        return Matrix{QQ}(undef, 0, 0)
    end
    row1 = A_any[1]
    row1 isa AbstractVector || error("$(label)[1] must be a vector.")
    n = length(row1)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m
        row = A_any[i]
        row isa AbstractVector || error("$(label)[$i] must be a vector.")
        length(row) == n || error("$(label) row-length mismatch at row $(i).")
        for j in 1:n
            A[i, j] = _qq_parse(row[j])
        end
    end
    return A
end

function _strict_mask_from_json(mask_any, nrows::Int, label::String)
    if mask_any === nothing
        return falses(nrows)
    elseif mask_any isa AbstractVector{Bool}
        length(mask_any) == nrows || error("$(label) length mismatch.")
        return BitVector(mask_any)
    elseif mask_any isa AbstractVector
        if length(mask_any) == nrows && all(x -> x isa Bool, mask_any)
            return BitVector(Bool[x for x in mask_any])
        end
        bits = falses(nrows)
        for t in mask_any
            idx = Int(t)
            1 <= idx <= nrows || error("$(label) index $(idx) out of range 1:$(nrows).")
            bits[idx] = true
        end
        return bits
    end
    error("$(label) must be Bool vector or index list.")
end

function _pl_hpoly_to_obj(h::PLPolyhedra.HPoly)
    return Dict(
        "A" => _qq_matrix_to_obj(h.A),
        "b" => _qq_vector_to_obj(h.b),
        "strict_mask" => collect(h.strict_mask),
        "strict_eps" => _qq_json(h.strict_eps),
    )
end

function _pl_union_to_obj(U::PLPolyhedra.PolyUnion)
    return Dict(
        "n" => U.n,
        "parts" => [_pl_hpoly_to_obj(p) for p in U.parts],
    )
end

function _parse_pl_hpoly_obj(part_obj, n::Int, label::String)
    haskey(part_obj, "A") || error("$(label) missing A.")
    haskey(part_obj, "b") || error("$(label) missing b.")
    A = _parse_qq_matrix(part_obj["A"], "$(label).A")
    b = _parse_qq_vector(part_obj["b"], "$(label).b")
    size(A, 1) == length(b) || error("$(label): size(A,1) must match length(b).")
    size(A, 2) == n || error("$(label): A has wrong ambient dimension $(size(A,2)); expected $(n).")
    strict_mask = _strict_mask_from_json(get(part_obj, "strict_mask", nothing), size(A, 1), "$(label).strict_mask")
    strict_eps = haskey(part_obj, "strict_eps") ? _qq_parse(part_obj["strict_eps"]) : PLPolyhedra.STRICT_EPS_QQ
    return PLPolyhedra.HPoly(n, A, b, nothing, strict_mask, strict_eps)
end

function _parse_pl_union_obj(union_obj, n::Int, label::String)
    haskey(union_obj, "n") || error("$(label) missing canonical `n`.")
    Int(union_obj["n"]) == n || error("$(label).n must equal ambient dimension $(n).")
    haskey(union_obj, "parts") || error("$(label) missing canonical `parts`.")
    parts_any = union_obj["parts"]
    parts_any isa AbstractVector || error("$(label).parts must be a vector.")
    parts = Vector{PLPolyhedra.HPoly}(undef, length(parts_any))
    @inbounds for i in eachindex(parts_any)
        parts[i] = _parse_pl_hpoly_obj(parts_any[i], n, "$(label).parts[$(i)]")
    end
    return PLPolyhedra.PolyUnion(n, parts)
end

function _parse_pl_generators(obj, key::String, n::Int, which::Symbol)
    haskey(obj, key) || error("PLFringe JSON missing '$(key)'.")
    gens_any = obj[key]
    gens_any isa AbstractVector || error("PLFringe $(key) must be a vector.")
    if which === :up
        out = Vector{PLPolyhedra.PLUpset}(undef, length(gens_any))
        @inbounds for i in eachindex(gens_any)
            U = _parse_pl_union_obj(gens_any[i], n, "$(key)[$(i)]")
            out[i] = PLPolyhedra.PLUpset(U)
        end
        return out
    end
    out = Vector{PLPolyhedra.PLDownset}(undef, length(gens_any))
    @inbounds for i in eachindex(gens_any)
        D = _parse_pl_union_obj(gens_any[i], n, "$(key)[$(i)]")
        out[i] = PLPolyhedra.PLDownset(D)
    end
    return out
end

function _parse_pl_phi(obj, n_down::Int, n_up::Int)
    if !haskey(obj, "phi")
        n_down == 0 || n_up == 0 || error("PLFringe JSON missing 'phi'.")
        return zeros(QQ, n_down, n_up)
    end
    Phi = _parse_qq_matrix(obj["phi"], "phi")
    size(Phi, 1) == n_down || error("PLFringe phi has wrong number of rows $(size(Phi,1)); expected $(n_down).")
    size(Phi, 2) == n_up || error("PLFringe phi has wrong number of cols $(size(Phi,2)); expected $(n_up).")
    return Phi
end

function _parse_pl_fringe_obj(obj; validation::Bool=true)
    if validation
        haskey(obj, "kind") && String(obj["kind"]) == "PLFringe" ||
            error("PLFringe JSON must set kind=\"PLFringe\".")
        version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 0
        version == PLFRINGE_SCHEMA_VERSION ||
            error("Unsupported PLFringe JSON schema_version: $(version). Expected $(PLFRINGE_SCHEMA_VERSION).")
    end
    n = Int(obj["n"])
    n >= 0 || error("PLFringe n must be >= 0.")
    Ups = _parse_pl_generators(obj, "ups", n, :up)
    Downs = _parse_pl_generators(obj, "downs", n, :down)
    Phi = _parse_pl_phi(obj, length(Downs), length(Ups))
    return PLPolyhedra.PLFringe(n, Ups, Downs, Phi)
end

@inline function _parse_pl_hpoly_typed(part_obj::_CanonicalPLHPolyJSON, n::Int, label::String)
    part_obj.A === nothing && error("$(label) missing A.")
    part_obj.b === nothing && error("$(label) missing b.")
    A = _parse_qq_matrix(part_obj.A, "$(label).A")
    b = _parse_qq_vector(part_obj.b, "$(label).b")
    size(A, 1) == length(b) || error("$(label): size(A,1) must match length(b).")
    size(A, 2) == n || error("$(label): A has wrong ambient dimension $(size(A,2)); expected $(n).")
    strict_mask = _strict_mask_from_json(part_obj.strict_mask, size(A, 1), "$(label).strict_mask")
    strict_eps = part_obj.strict_eps === nothing ? PLPolyhedra.STRICT_EPS_QQ : _qq_parse(part_obj.strict_eps)
    return PLPolyhedra.HPoly(n, A, b, nothing, strict_mask, strict_eps)
end

function _parse_pl_union_typed(union_obj::_CanonicalPLUnionJSON, n::Int, label::String)
    union_obj.n == n || error("$(label).n must equal ambient dimension $(n).")
    parts = Vector{PLPolyhedra.HPoly}(undef, length(union_obj.parts))
    @inbounds for i in eachindex(union_obj.parts)
        parts[i] = _parse_pl_hpoly_typed(union_obj.parts[i], n, "$(label).parts[$(i)]")
    end
    return PLPolyhedra.PolyUnion(n, parts)
end

function _parse_pl_fringe_typed(obj::_CanonicalPLFringeJSON; validation::Bool=true)
    if validation
        obj.kind == "PLFringe" || error("PLFringe JSON must set kind=\"PLFringe\".")
        obj.schema_version == PLFRINGE_SCHEMA_VERSION ||
            error("Unsupported PLFringe JSON schema_version: $(obj.schema_version). Expected $(PLFRINGE_SCHEMA_VERSION).")
    end
    n = obj.n
    n >= 0 || error("PLFringe n must be >= 0.")
    Ups = Vector{PLPolyhedra.PLUpset}(undef, length(obj.ups))
    Downs = Vector{PLPolyhedra.PLDownset}(undef, length(obj.downs))
    @inbounds for i in eachindex(obj.ups)
        Ups[i] = PLPolyhedra.PLUpset(_parse_pl_union_typed(obj.ups[i], n, "ups[$(i)]"))
    end
    @inbounds for i in eachindex(obj.downs)
        Downs[i] = PLPolyhedra.PLDownset(_parse_pl_union_typed(obj.downs[i], n, "downs[$(i)]"))
    end
    Phi = obj.phi === nothing ? zeros(QQ, length(Downs), length(Ups)) : _parse_qq_matrix(obj.phi, "phi")
    size(Phi, 1) == length(Downs) || error("PLFringe phi has wrong number of rows $(size(Phi,1)); expected $(length(Downs)).")
    size(Phi, 2) == length(Ups) || error("PLFringe phi has wrong number of cols $(size(Phi,2)); expected $(length(Ups)).")
    return PLPolyhedra.PLFringe(n, Ups, Downs, Phi)
end

"""
    save_pl_fringe_json(path, F::PLPolyhedra.PLFringe; profile=:compact, pretty=nothing)

Stable TamerOp-owned PL fringe schema for `PLPolyhedra.PLFringe`.

This is the canonical owned write path for PL fringe artifacts. Use
[`pl_fringe_json_summary`](@ref) or [`inspect_json`](@ref) to inspect an
existing file cheaply, and use [`check_pl_fringe_json`](@ref) when you need
strict schema validation before calling [`load_pl_fringe_json`](@ref).

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.
"""
function save_pl_fringe_json(path::AbstractString, F::PLPolyhedra.PLFringe;
                             profile::Symbol=:compact,
                             pretty::Union{Nothing,Bool}=nothing)
    ups = [_pl_union_to_obj(U.U) for U in F.Ups]
    downs = [_pl_union_to_obj(D.D) for D in F.Downs]
    phi = _qq_matrix_to_obj(F.Phi)
    obj = Dict(
        "kind" => "PLFringe",
        "schema_version" => PLFRINGE_SCHEMA_VERSION,
        "n" => F.n,
        "coeff_field" => _field_to_obj(QQField()),
        "ups" => ups,
        "downs" => downs,
        "phi" => phi,
    )
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_pl_fringe_json(path; validation=:strict) -> PLPolyhedra.PLFringe

Load a PL fringe artifact written by [`save_pl_fringe_json`](@ref).

This is a strict owned-schema loader. Prefer [`pl_fringe_json_summary`](@ref)
for a cheap-first family check and [`check_pl_fringe_json`](@ref) when you need
explicit schema validation before loading.

Use `validation=:trusted` only for TamerOp-produced files when you want
the lighter typed parse path without the extra owned-schema checks.
"""
function load_pl_fringe_json(path::AbstractString; validation::Symbol=:strict)
    validate = _resolve_validation_mode(validation)
    raw = read(path)
    if validate
        return _parse_pl_fringe_obj(JSON3.read(raw); validation=true)
    end
    return _parse_pl_fringe_typed(JSON3.read(raw, _CanonicalPLFringeJSON); validation=false)
end

# -----------------------------------------------------------------------------
# A2) Finite encodings (FiniteFringe + typed v1 schema)
# -----------------------------------------------------------------------------

@inline function _pi_to_obj(pi)
    if pi isa CompiledEncoding
        pi = pi.pi
    end
    if pi isa GridEncodingMap
        return _GridEncodingMapJSON(kind="GridEncodingMap",
                                    coords=[collect(ax) for ax in pi.coords],
                                    orientation=collect(pi.orientation))
    elseif pi isa ZnEncoding.ZnEncodingMap
        return _ZnEncodingMapJSON(
            kind="ZnEncodingMap",
            n=pi.n,
            coords=[collect(ax) for ax in pi.coords],
            sig_y=_pack_signature_rows_obj(pi.sig_y),
            sig_z=_pack_signature_rows_obj(pi.sig_z),
            reps=[collect(r) for r in pi.reps],
            flats=[_FaceGeneratorJSON(collect(f.b), findall(identity, f.tau.coords)) for f in pi.flats],
            injectives=[_FaceGeneratorJSON(collect(e.b), findall(identity, e.tau.coords)) for e in pi.injectives],
            cell_shape=collect(pi.cell_shape),
            cell_strides=collect(pi.cell_strides),
            cell_to_region=pi.cell_to_region === nothing ? nothing : collect(pi.cell_to_region),
        )
    elseif pi isa PLBackend.PLEncodingMapBoxes
        return _PLEncodingMapBoxesJSON(
            kind="PLEncodingMapBoxes",
            n=pi.n,
            coords=[collect(ax) for ax in pi.coords],
            sig_y=_pack_masks_obj(pi.sig_y),
            sig_z=_pack_masks_obj(pi.sig_z),
            reps=[collect(r) for r in pi.reps],
            Ups=[collect(u.ell) for u in pi.Ups],
            Downs=[collect(d.u) for d in pi.Downs],
            cell_shape=collect(pi.cell_shape),
            cell_strides=collect(pi.cell_strides),
            cell_to_region=collect(pi.cell_to_region),
            coord_flags=[collect(f) for f in pi.coord_flags],
            axis_is_uniform=collect(pi.axis_is_uniform),
            axis_step=collect(pi.axis_step),
            axis_min=collect(pi.axis_min),
        )
    end
    error("Unsupported encoding map type for JSON serialization.")
end

@inline function _zn_sigkey(sig_y::PackedSignatureRows{MY},
                            sig_z::PackedSignatureRows{MZ},
                            t::Int) where {MY,MZ}
    ywords = ntuple(i -> sig_y.words[i, t], Val(MY))
    zwords = ntuple(i -> sig_z.words[i, t], Val(MZ))
    return ZnEncoding.SigKey{MY,MZ}(ywords, zwords)
end

function _pi_from_typed(P::AbstractPoset, obj::_GridEncodingMapJSON)
    n = length(obj.coords)
    coords = ntuple(i -> Vector{Float64}(obj.coords[i]), n)
    length(obj.orientation) == n || error("GridEncodingMap.orientation length mismatch.")
    orientation = ntuple(i -> Int(obj.orientation[i]), n)
    return GridEncodingMap(P, coords; orientation=orientation)
end

function _pi_from_typed(::AbstractPoset, obj::_ZnEncodingMapJSON)
    n = obj.n
    length(obj.coords) == n || error("ZnEncodingMap.coords length mismatch.")
    coords = ntuple(i -> Vector{Int}(obj.coords[i]), n)
    reps = [ntuple(i -> Int(r[i]), n) for r in obj.reps]
    mkface(idxs) = begin
        m = falses(n)
        for t in idxs
            m[Int(t)] = true
        end
        Face(n, m)
    end
    flats = [IndFlat(mkface(f.tau), Vector{Int}(f.b); id=:F) for f in obj.flats]
    injectives = [IndInj(mkface(e.tau), Vector{Int}(e.b); id=:E) for e in obj.injectives]
    sig_y = _parse_signature_rows_packed(obj.sig_y, "ZnEncodingMap.sig_y")
    sig_z = _parse_signature_rows_packed(obj.sig_z, "ZnEncodingMap.sig_z")
    length(sig_y) == length(reps) || error("ZnEncodingMap.sig_y region count mismatch.")
    length(sig_z) == length(reps) || error("ZnEncodingMap.sig_z region count mismatch.")
    sig_y.bitlen == length(flats) || error("ZnEncodingMap.sig_y bit length mismatch with flats.")
    sig_z.bitlen == length(injectives) || error("ZnEncodingMap.sig_z bit length mismatch with injectives.")
    MY = size(sig_y.words, 1)
    MZ = size(sig_z.words, 1)
    sig_to_region = Dict{ZnEncoding.SigKey{MY,MZ},Int}()
    @inbounds for t in 1:length(sig_y)
        sig_to_region[_zn_sigkey(sig_y, sig_z, t)] = t
    end
    if obj.cell_shape === nothing || obj.cell_strides === nothing
        return ZnEncoding.ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats, injectives, sig_to_region)
    end
    cell_shape = Vector{Int}(obj.cell_shape)
    cell_strides = Vector{Int}(obj.cell_strides)
    length(cell_shape) == n || error("ZnEncodingMap.cell_shape length mismatch.")
    length(cell_strides) == n || error("ZnEncodingMap.cell_strides length mismatch.")
    cell_to_region = obj.cell_to_region === nothing ? nothing : Vector{Int}(obj.cell_to_region)
    return ZnEncoding.ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats, injectives, sig_to_region,
                                    ntuple(i -> cell_shape[i], n),
                                    ntuple(i -> cell_strides[i], n),
                                    cell_to_region)
end

function _pi_from_typed(::AbstractPoset, obj::_PLEncodingMapBoxesJSON)
    n = obj.n
    length(obj.coords) == n || error("PLEncodingMapBoxes.coords length mismatch.")
    coords = ntuple(i -> Vector{Float64}(obj.coords[i]), n)
    reps = [ntuple(i -> Float64(r[i]), n) for r in obj.reps]
    Ups = [PLBackend.BoxUpset(Vector{Float64}(u)) for u in obj.Ups]
    Downs = [PLBackend.BoxDownset(Vector{Float64}(d)) for d in obj.Downs]
    sig_y = _decode_masks(obj.sig_y, "PLEncodingMapBoxes.sig_y", length(Ups))
    sig_z = _decode_masks(obj.sig_z, "PLEncodingMapBoxes.sig_z", length(Downs))
    length(sig_y) == length(reps) || error("PLEncodingMapBoxes.sig_y region count mismatch.")
    length(sig_z) == length(reps) || error("PLEncodingMapBoxes.sig_z region count mismatch.")
    MY = max(1, cld(max(length(Ups), 1), 64))
    MZ = max(1, cld(max(length(Downs), 1), 64))
    sig_to_region = Dict{PLBackend.SigKey{MY,MZ},Int}()
    @inbounds for t in eachindex(sig_y)
        ywords = PLBackend._pack_bitvector_words(sig_y[t], Val(MY))
        zwords = PLBackend._pack_bitvector_words(sig_z[t], Val(MZ))
        sig_to_region[PLBackend.SigKey{MY,MZ}(ywords, zwords)] = t
    end
    return PLBackend.PLEncodingMapBoxes{n,MY,MZ}(
        n, coords, sig_y, sig_z, reps, Ups, Downs, sig_to_region,
        Vector{Int}(obj.cell_shape),
        Vector{Int}(obj.cell_strides),
        Vector{Int}(obj.cell_to_region),
        [Vector{UInt8}(f) for f in obj.coord_flags],
        BitVector(obj.axis_is_uniform),
        Vector{Float64}(obj.axis_step),
        Vector{Float64}(obj.axis_min))
end

function _pi_from_typed(::AbstractPoset, obj::_PiJSON)
    error("Unsupported encoding map kind: $(typeof(obj))")
end

function _encoding_obj(H::FringeModule{K};
                       pi=nothing,
                       include_leq::Union{Bool,Symbol}=:auto) where {K}
    P = H.P

    U_masks = _pack_masks_obj([U.mask for U in H.U])
    D_masks = _pack_masks_obj([D.mask for D in H.D])
    phi = _phi_obj(H)

    obj = Dict(
        "kind" => "FiniteEncodingFringe",
        "schema_version" => ENCODING_SCHEMA_VERSION,
        "poset" => _poset_obj(P; include_leq=include_leq),
        "U" => U_masks,
        "D" => D_masks,
        "coeff_field" => _field_to_obj(H.field),
        "phi" => phi,
    )
    if pi !== nothing
        obj["pi"] = _pi_to_obj(pi)
    end
    return obj
end

"""
    save_encoding_json(path, H::FringeModule; profile=:compact, include_leq=nothing, pretty=nothing)
    save_encoding_json(path, P, H::FringeModule, pi; profile=:compact, include_leq=nothing, pretty=nothing)

Serialize a finite encoding artifact in the stable TamerOp-owned encoding
schema.

This is the canonical owned write path for fringe/encoding artifacts. The
default `profile=:compact` keeps the file cheap by omitting dense `leq`
materialization and writing compact JSON. Use [`encoding_json_summary`](@ref)
or [`inspect_json`](@ref) to inspect an existing artifact cheaply, and use
[`check_encoding_json`](@ref) when you need strict schema validation before
calling [`load_encoding_json`](@ref).
"""
function save_encoding_json(path::AbstractString, H::FringeModule{K};
                            profile::Symbol=:compact,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing) where {K}
    defaults = _resolve_encoding_save_profile(profile)
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    return _json_write(path, _encoding_obj(H; include_leq=include_leq_resolved); pretty=pretty_resolved)
end

function save_encoding_json(path::AbstractString, P::AbstractPoset, H::FringeModule{K}, pi;
                            profile::Symbol=:compact,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing) where {K}
    defaults = _resolve_encoding_save_profile(profile)
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    P === H.P || error("save_encoding_json: P does not match H.P.")
    return _json_write(path, _encoding_obj(H; pi=pi, include_leq=include_leq_resolved); pretty=pretty_resolved)
end

"""
    save_encoding_json(path, enc::EncodingResult; profile=:compact, include_pi=nothing, include_leq=nothing, pretty=nothing)

Convenience serialization entrypoint for workflow users.

This uses the same owned encoding-schema contract as the lower-level
`save_encoding_json` methods above while accepting a workflow-level
`EncodingResult`.
"""
function save_encoding_json(path::AbstractString, enc::EncodingResult;
                            profile::Symbol=:compact,
                            include_pi::Union{Nothing,Bool}=nothing,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing)
    defaults = _resolve_encoding_save_profile(profile)
    include_pi_resolved = include_pi === nothing ? defaults.include_pi : include_pi
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    H = enc.H
    H === nothing && (H = fringe_presentation(materialize_module(enc.M)))
    H isa FringeModule || error("save_encoding_json: EncodingResult.H must be a FringeModule.")
    if include_pi_resolved
        return save_encoding_json(path, enc.P, H, enc.pi; include_leq=include_leq_resolved, pretty=pretty_resolved)
    end
    return save_encoding_json(path, H; include_leq=include_leq_resolved, pretty=pretty_resolved)
end

# Load the schema emitted by save_encoding_json.
#
# This loader is intentionally strict: it expects the schema emitted by
# save_encoding_json (missing required keys => error).
function _load_encoding_json_v1(raw;
                                output::Symbol=:encoding_result,
                                field::Union{Nothing,AbstractCoeffField}=nothing,
                                validation::Symbol=:strict)
    outmode = _resolve_encoding_output_mode(output)
    validate_masks = _resolve_validation_mode(validation)
    obj = JSON3.read(raw, _FiniteEncodingFringeJSONV1)
    obj.kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(obj.kind)")
    obj.schema_version == ENCODING_SCHEMA_VERSION ||
        error("Unsupported encoding JSON schema_version: $(obj.schema_version)")

    P = _parse_poset_from_typed(obj.poset)
    n = nvertices(P)
    if validate_masks && obj.U isa _MaskPackedWordsJSON && obj.D isa _MaskPackedWordsJSON
        plan = _packed_cover_validation_plan(P)
        U = _validated_upsets(obj.U, P, plan, "U", n)
        D = _validated_downsets(obj.D, P, plan, "D", n)
    elseif !validate_masks && obj.U isa _MaskPackedWordsJSON && obj.D isa _MaskPackedWordsJSON
        U = _decode_upsets(obj.U, P, "U", n)
        D = _decode_downsets(obj.D, P, "D", n)
    elseif validate_masks
        Umasks = _decode_masks(obj.U, "U", n)
        Dmasks = _decode_masks(obj.D, "D", n)
        U = _build_upsets(P, Umasks, true)
        D = _build_downsets(P, Dmasks, true)
    else
        Umasks = _decode_masks(obj.U, "U", n)
        Dmasks = _decode_masks(obj.D, "D", n)
        U = _build_upsets(P, Umasks, false)
        D = _build_downsets(P, Dmasks, false)
    end

    saved_field = _field_from_typed(obj.coeff_field)
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    m = length(D)
    k = length(U)
    Phi = _decode_phi(obj.phi, saved_field, target_field, m, k)

    H = FiniteFringe.FringeModule{K}(P, U, D, Phi; field=target_field)
    if outmode === :fringe
        return H
    elseif outmode === :fringe_with_pi
        obj.pi === nothing && error("load_encoding_json: output=:fringe_with_pi requires a stored pi payload. Re-save with save_encoding_json(...; include_pi=true) or load with output=:fringe.")
        return H, _pi_from_typed(P, obj.pi)
    elseif outmode === :encoding_result
        obj.pi === nothing && error("load_encoding_json: output=:encoding_result requires a stored pi payload. Re-save with save_encoding_json(...; include_pi=true) or load with output=:fringe.")
        pi = _pi_from_typed(P, obj.pi)
        M = pmodule_from_fringe(H)
        return EncodingResult(P, M, pi;
                              H=H,
                              presentation=nothing,
                              opts=EncodingOptions(),
                              backend=:serialization,
                              meta=(; source=:load_encoding_json, schema_version=obj.schema_version))
    end
    error("unreachable output mode: $(outmode)")
end

"""
    load_encoding_json(path; output=:encoding_result, field=nothing, validation=:strict)

Load an encoding artifact written by [`save_encoding_json`](@ref).

Keywords
--------
- `output=:fringe | :fringe_with_pi | :encoding_result`
- `field`: optional coefficient-field override
- `validation=:strict | :trusted`

This is a strict owned-schema loader by default. The cheap-first exploration
path is [`encoding_json_summary`](@ref) or [`inspect_json`](@ref); use those to
confirm the artifact family and inspect whether a stored encoding map or dense
`leq` payload is present before choosing a heavier load mode.

Use `validation=:trusted` only for artifacts produced by TamerOp itself
when you explicitly want to skip mask validation on the hot load path.
"""
function load_encoding_json(path::AbstractString;
                            output::Symbol=:encoding_result,
                            field::Union{Nothing,AbstractCoeffField}=nothing,
                            validation::Symbol=:strict)
    raw = read(path)
    return _load_encoding_json_v1(raw; output=output, field=field, validation=validation)
end

