# Canonical external-adapter JSON parsers for finite fringe / PL fringe
# and related CAS-driven entrypoints.

# =============================================================================
# B) External adapters (CAS ingestion)
# =============================================================================

function _external_parse_mask_rows(mask_any, name::String, ncols::Int)
    if mask_any isa AbstractDict
        if _is_packed_words_obj(mask_any)
            packed = _MaskPackedWordsJSON(kind=String(mask_any["kind"]),
                                          nrows=Int(mask_any["nrows"]),
                                          ncols=Int(mask_any["ncols"]),
                                          words_per_row=Int(mask_any["words_per_row"]),
                                          words=Vector{UInt64}(mask_any["words"]))
            return _decode_masks(packed, name, ncols)
        end
        error("$(name): unsupported mask object; expected packed_words_v1 or an explicit vector of mask rows.")
    end

    mask_any isa AbstractVector || error("$(name) must be a vector of masks.")
    rows = Vector{BitVector}(undef, length(mask_any))
    @inbounds for i in eachindex(mask_any)
        row_any = mask_any[i]
        row_any isa AbstractVector || error("$(name)[$(i)] must be a vector.")
        if length(row_any) == ncols && all(x -> x isa Bool, row_any)
            rows[i] = BitVector(Bool[x for x in row_any])
        else
            mask = falses(ncols)
            for t in row_any
                idx = Int(t)
                1 <= idx <= ncols || error("$(name)[$(i)] contains out-of-range index $(idx) for ncols=$(ncols).")
                mask[idx] = true
            end
            rows[i] = mask
        end
    end
    return rows
end

function _external_packed_mask_obj(mask_any)
    if mask_any isa AbstractDict && _is_packed_words_obj(mask_any)
        return _MaskPackedWordsJSON(kind=String(mask_any["kind"]),
                                    nrows=Int(mask_any["nrows"]),
                                    ncols=Int(mask_any["ncols"]),
                                    words_per_row=Int(mask_any["words_per_row"]),
                                    words=Vector{UInt64}(mask_any["words"]))
    end
    return nothing
end

function _external_parse_upsets(mask_any,
                                P::AbstractPoset,
                                name::String,
                                validate_masks::Bool,
                                plan::Union{Nothing,_PackedCoverValidationPlan}=nothing)
    packed = _external_packed_mask_obj(mask_any)
    if packed !== nothing
        if validate_masks
            plan === nothing && (plan = _packed_cover_validation_plan(P))
            return _validated_upsets(packed, P, plan, name, nvertices(P))
        end
        return _decode_upsets(packed, P, name, nvertices(P))
    end
    masks = _external_parse_mask_rows(mask_any, name, nvertices(P))
    return _build_upsets(P, masks, validate_masks)
end

function _external_parse_downsets(mask_any,
                                  P::AbstractPoset,
                                  name::String,
                                  validate_masks::Bool,
                                  plan::Union{Nothing,_PackedCoverValidationPlan}=nothing)
    packed = _external_packed_mask_obj(mask_any)
    if packed !== nothing
        if validate_masks
            plan === nothing && (plan = _packed_cover_validation_plan(P))
            return _validated_downsets(packed, P, plan, name, nvertices(P))
        end
        return _decode_downsets(packed, P, name, nvertices(P))
    end
    masks = _external_parse_mask_rows(mask_any, name, nvertices(P))
    return _build_downsets(P, masks, validate_masks)
end

function _external_parse_field(obj, field_override::Union{Nothing,AbstractCoeffField})
    saved_field = if haskey(obj, "coeff_field")
        _field_from_obj(obj["coeff_field"])
    elseif haskey(obj, "field")
        error("finite fringe JSON must use canonical `coeff_field`, not legacy `field`.")
    else
        QQField()
    end
    target_field = field_override === nothing ? saved_field : field_override
    return saved_field, target_field
end

function _external_parse_field(coeff_field_any::Union{Nothing,_CoeffFieldJSON},
                               field_override::Union{Nothing,AbstractCoeffField})
    saved_field = coeff_field_any === nothing ? QQField() : _field_from_typed(coeff_field_any)
    target_field = field_override === nothing ? saved_field : field_override
    return saved_field, target_field
end

function _external_parse_phi(phi_any,
                             saved_field::AbstractCoeffField,
                             target_field::AbstractCoeffField,
                             m_expected::Int,
                             k_expected::Int)
    if phi_any isa AbstractDict && haskey(phi_any, "kind")
        kind = String(phi_any["kind"])
        if kind == "qq_chunks_v1"
            phi_obj = _PhiQQChunksJSON(kind=kind,
                                       m=Int(phi_any["m"]),
                                       k=Int(phi_any["k"]),
                                       base=Int(phi_any["base"]),
                                       num_sign=Vector{Int8}(Int8.(phi_any["num_sign"])),
                                       num_ptr=Vector{Int}(phi_any["num_ptr"]),
                                       num_chunks=Vector{UInt32}(UInt32.(phi_any["num_chunks"])),
                                       den_ptr=Vector{Int}(phi_any["den_ptr"]),
                                       den_chunks=Vector{UInt32}(UInt32.(phi_any["den_chunks"])))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        elseif kind == "fp_flat_v1"
            phi_obj = _PhiFpFlatJSON(kind=kind,
                                     m=Int(phi_any["m"]),
                                     k=Int(phi_any["k"]),
                                     p=Int(phi_any["p"]),
                                     data=Vector{Int}(phi_any["data"]))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        elseif kind == "real_flat_v1"
            phi_obj = _PhiRealFlatJSON(kind=kind,
                                       m=Int(phi_any["m"]),
                                       k=Int(phi_any["k"]),
                                       T=String(phi_any["T"]),
                                       data=Vector{Float64}(phi_any["data"]))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        end
        error("Unsupported external phi kind: $(kind)")
    end

    phi_any isa AbstractVector || error("phi must be a matrix encoded as row vectors.")
    length(phi_any) == m_expected || error("phi row count mismatch (expected $(m_expected), got $(length(phi_any))).")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m_expected, k_expected)
    @inbounds for i in 1:m_expected
        row = phi_any[i]
        row isa AbstractVector || error("phi[$(i)] must be a row vector.")
        length(row) == k_expected || error("phi column count mismatch at row $(i) (expected $(k_expected), got $(length(row))).")
        for j in 1:k_expected
            s = _scalar_from_json(saved_field, row[j])
            Phi[i, j] = coerce(target_field, s)
        end
    end
    return Phi
end

"""
    parse_finite_fringe_json(json_src; field=nothing, validation=:strict) -> FringeModule

Parse canonical finite-fringe JSON with top-level keys
`poset`, `U`, `D`, and `phi`.
"""
function parse_finite_fringe_json(json_src;
                                  field::Union{Nothing,AbstractCoeffField}=nothing,
                                  validation::Symbol=:strict)
    obj = JSON3.read(json_src)
    validate_masks = _resolve_validation_mode(validation)

    haskey(obj, "poset") || error("parse_finite_fringe_json: missing canonical `poset`.")
    P = _parse_poset_from_obj(obj["poset"])
    n = nvertices(P)

    haskey(obj, "U") || error("parse_finite_fringe_json: missing canonical `U`.")
    haskey(obj, "D") || error("parse_finite_fringe_json: missing canonical `D`.")
    haskey(obj, "phi") || error("parse_finite_fringe_json: missing `phi` matrix.")
    U_any = obj["U"]
    D_any = obj["D"]
    maybe_plan = validate_masks && _external_packed_mask_obj(U_any) !== nothing && _external_packed_mask_obj(D_any) !== nothing ?
        _packed_cover_validation_plan(P) : nothing
    U = _external_parse_upsets(U_any, P, "U", validate_masks, maybe_plan)
    D = _external_parse_downsets(D_any, P, "D", validate_masks, maybe_plan)
    saved_field, target_field = _external_parse_field(obj, field)
    Phi = _external_parse_phi(obj["phi"], saved_field, target_field, length(D), length(U))
    K = coeff_type(target_field)
    return FiniteFringe.FringeModule{K}(P, U, D, Phi; field=target_field)
end

"""
    parse_pl_fringe_json(json_src) -> PLPolyhedra.PLFringe

Parse canonical TamerOp PL fringe JSON. The payload must match the
schema emitted by `save_pl_fringe_json`.
"""
function parse_pl_fringe_json(json_src)
    obj = JSON3.read(json_src, _CanonicalPLFringeJSON)
    return _parse_pl_fringe_typed(obj)
end

"""
    finite_fringe_from_m2(cmd::Cmd; jsonpath=nothing, field=nothing, validation=:strict)
        -> FringeModule

Run a CAS command that prints (or writes) finite-fringe JSON accepted by
`parse_finite_fringe_json`, then parse it.
"""
function finite_fringe_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing,
                               field::Union{Nothing,AbstractCoeffField}=nothing,
                               validation::Symbol=:strict)
    if jsonpath === nothing
        io = read(cmd, String)
        return parse_finite_fringe_json(io; field=field, validation=validation)
    end
    run(cmd)
    open(jsonpath, "r") do io
        return parse_finite_fringe_json(io; field=field, validation=validation)
    end
end

"""
JSON schema expected from an external CAS (Macaulay2, Singular, ...):

{
  "n": 3,                                   // ambient dimension
  "coeff_field": { "kind": "qq" },          // optional; defaults to QQ
  "flats": [
     {"b":[0,0,0], "tau":[true,false,false], "id":"F1"},
     {"b":[2,1,0], "tau":[false,false,true], "id":"F2"}
  ],
  "injectives": [
     {"b":[1,3,5], "tau":[true,false,false], "id":"E1"},
     {"b":[4,4,0], "tau":[false,true,false], "id":"E2"}
  ],
  // Optional: monomial matrix rows=#injectives, cols=#flats
  "phi": [[1,0],
          [0,1]]
}

Notes:
* `tau` denotes a face of N^n. We accept either a Bool vector or a list of indices.
* Scalars in `phi` are interpreted in QQ (exact rationals).
* If `phi` is omitted, we fall back to `canonical_matrix(flats, injectives)`.
"""
function parse_flange_json(json_src; field::Union{Nothing,AbstractCoeffField}=nothing)
    obj = JSON3.read(json_src)
    n = Int(obj["n"])
    saved_field = if haskey(obj, "coeff_field")
        _field_from_obj(obj["coeff_field"])
    elseif haskey(obj, "field")
        String(obj["field"]) == "QQ" ? QQField() : QQField()
    else
        QQField()
    end
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)

    function _mkface(n::Int, tau_any)
        if tau_any isa AbstractVector{Bool}
            return Face(n, BitVector(tau_any))
        end
        bits = falses(n)
        for t in tau_any
            bits[Int(t)] = true
        end
        return Face(n, bits)
    end

    flats = IndFlat{n}[]
    for f in obj["flats"]
        b = Vector{Int}(f["b"])
        tau = _mkface(n, f["tau"])
        id = Symbol(String(get(f, "id", "F")))
        push!(flats, IndFlat(tau, b; id=id))
    end

    injectives = IndInj{n}[]
    for e in obj["injectives"]
        b = Vector{Int}(e["b"])
        tau = _mkface(n, e["tau"])
        id = Symbol(String(get(e, "id", "E")))
        push!(injectives, IndInj(tau, b; id=id))
    end

    Phi = if haskey(obj, "phi")
        A = obj["phi"]
        m = length(injectives)
        ncol = length(flats)
        M = zeros(K, m, ncol)
        nonintegral_numeric_entries = 0
        @assert length(A) == m "phi: wrong number of rows"
        for i in 1:m
            row = A[i]
            @assert length(row) == ncol "phi: wrong number of cols"
            for j in 1:ncol
                val = row[j]
                if val isa String
                    M[i, j] = _scalar_from_json(saved_field, val)
                elseif val isa Integer
                    M[i, j] = _scalar_from_json(saved_field, val)
                else
                    nonintegral_numeric_entries += 1
                    M[i, j] = _scalar_from_json(saved_field, val)
                end
                if target_field !== saved_field
                    M[i, j] = coerce(target_field, M[i, j])
                end
            end
        end
        if nonintegral_numeric_entries > 0
            @warn "phi has $(nonintegral_numeric_entries) non-integer numeric entries; prefer exact strings \"num/den\" for exactness"
        end
        M
    else
        canonical_matrix(flats, injectives; field=target_field)
    end

    return Flange{K}(n, flats, injectives, Phi; field=target_field)
end

"""
    flange_from_m2(cmd::Cmd; jsonpath=nothing) -> Flange{QQ}

Run a CAS command that prints (or writes) the JSON described in `parse_flange_json`,
then parse it into a `Flange{QQ}`.
"""
function flange_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing,
                        field::Union{Nothing,AbstractCoeffField}=nothing)
    if jsonpath === nothing
        io = read(cmd, String)
        return parse_flange_json(io; field=field)
    end
    run(cmd)
    open(jsonpath, "r") do io
        return parse_flange_json(io; field=field)
    end
end

