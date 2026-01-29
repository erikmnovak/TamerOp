# File: Serialization.jl

"""
PosetModules.Serialization

All JSON-facing I/O lives here.

Separation of concerns
----------------------
A) Internal formats (owned/stable):
   - `save_*_json` / `load_*_json`
   - Schemas are controlled by PosetModules. Loaders are intentionally strict.

B) External adapters (CAS ingestion):
   - `parse_*_json` / `*_from_*`
   - Best-effort parsers for JSON emitted by external CAS tools (Macaulay2, Singular, ...).
     These schemas are not owned by PosetModules and may change upstream.

C) Invariant caches (MPPI):
   - `save_mpp_*_json` / `load_mpp_*_json`
   - Convenience cache formats for expensive derived objects defined in `PosetModules.Invariants`.

File structure (keep in this order)
-----------------------------------
1) Shared helpers
2) A. Internal formats
3) B. External adapters
4) C. Invariant caches
5) D. Compatibility shims (temporary)

If you add new JSON formats, put them in the appropriate section and keep the
public API functions (`save_*`, `load_*`, `parse_*`) at the top of that section.
"""
module Serialization

using JSON3

using ..CoreModules: QQ, rational_to_string, string_to_rational
import ..FlangeZn: Face, IndFlat, IndInj, Flange, canonical_matrix
import ..FiniteFringe: FinitePoset, FringeModule
using ..FiniteFringe

export save_flange_json, load_flange_json,
       save_encoding_json, load_encoding_json,
       parse_flange_json, flange_from_m2,
       save_mpp_decomposition_json, load_mpp_decomposition_json,
       save_mpp_image_json, load_mpp_image_json

# =============================================================================
# 1) Shared helpers
# =============================================================================

@inline function _json_write(path::AbstractString, obj)
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

@inline _json_read(path::AbstractString) = open(JSON3.read, path)

# =============================================================================
# A) Internal formats (owned/stable)
# =============================================================================

# -----------------------------------------------------------------------------
# A1) Flange (Z^n)  (FlangeZn.Flange)
# -----------------------------------------------------------------------------

"""
    save_flange_json(path, FG::FlangeZn.Flange)

Stable PosetModules-owned schema:

{
  "kind": "FlangeZn",
  "n": n,
  "flats":      [ {"b":[...], "tau":[i1,i2,...]}, ... ],
  "injectives": [ {"b":[...], "tau":[...]} , ... ],
  "phi": [[ "num/den", ...], ...]   # rows = #injectives, cols = #flats
}

Notes
* `tau` is stored as a list of 1-based coordinate indices where the face is true.
* Scalars are exact QQ encoded as "num/den" strings.
"""
function save_flange_json(path::AbstractString, FG::Flange)
    n = FG.n
    flats = [Dict("b" => F.b, "tau" => findall(identity, F.tau.coords)) for F in FG.flats]
    injectives = [Dict("b" => E.b, "tau" => findall(identity, E.tau.coords)) for E in FG.injectives]
    phi = [[rational_to_string(FG.phi[i, j]) for j in 1:length(FG.flats)]
           for i in 1:length(FG.injectives)]
    obj = Dict("kind" => "FlangeZn",
               "n" => n,
               "flats" => flats,
               "injectives" => injectives,
               "phi" => phi)
    return _json_write(path, obj)
end

"Inverse of `save_flange_json`."
function load_flange_json(path::AbstractString)
    obj = _json_read(path)
    @assert haskey(obj, "kind") && String(obj["kind"]) == "FlangeZn"
    n = Int(obj["n"])

    mkface(idxs) = Face(n, begin
        m = falses(n)
        for t in idxs
            m[Int(t)] = true
        end
        m
    end)

    flats = [IndFlat(mkface(Vector{Int}(f["tau"])), Vector{Int}(f["b"]); id=:F)
             for f in obj["flats"]]
    injectives = [IndInj(mkface(Vector{Int}(e["tau"])), Vector{Int}(e["b"]); id=:E)
                  for e in obj["injectives"]]

    m = length(injectives)
    k = length(flats)
    Phi = Matrix{QQ}(undef, m, k)
    for i in 1:m, j in 1:k
        Phi[i, j] = string_to_rational(String(obj["phi"][i][j]))
    end
    return Flange{QQ}(n, flats, injectives, Phi)
end

# -----------------------------------------------------------------------------
# A2) Finite encodings (FiniteFringe.FinitePoset + FringeModule)
# -----------------------------------------------------------------------------

# Save a finite poset and its fringe module (U, D, phi) to JSON.
# We store:
#   - leq: a dense Bool matrix
#   - each Upset/Downset as a Bool mask (BitVector serialized as Bool list)
#   - phi: exact rationals encoded as "num/den" strings
function save_encoding_json(path::AbstractString, H::FringeModule{QQ})
    P = H.P

    leq = [[P.leq[i, j] for j in 1:P.n] for i in 1:P.n]
    U_masks = [collect(Bool, U.mask) for U in H.U]
    D_masks = [collect(Bool, D.mask) for D in H.D]

    m, n = size(H.phi)
    phi = [[rational_to_string(H.phi[i, j]) for j in 1:n] for i in 1:m]

    obj = Dict(
        "kind" => "FiniteEncodingFringe",
        "poset" => Dict("n" => P.n, "leq" => leq),
        "U" => U_masks,
        "D" => D_masks,
        "phi" => phi,
    )
    return _json_write(path, obj)
end

# Load the schema emitted by save_encoding_json.
#
# This loader is intentionally strict: it expects the schema emitted by
# save_encoding_json (missing required keys => error).
function load_encoding_json(path::AbstractString)
    obj = _json_read(path)

    haskey(obj, "kind") || error("Encoding JSON missing required key 'kind'.")
    kind = String(obj["kind"])
    kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(kind)")

    haskey(obj, "poset") || error("Encoding JSON missing required key 'poset'.")
    poset_obj = obj["poset"]

    haskey(poset_obj, "n") || error("poset missing required key 'n'.")
    n = Int(poset_obj["n"])

    haskey(poset_obj, "leq") || error("poset missing required key 'leq'.")
    leq_any = poset_obj["leq"]
    leq_any isa AbstractVector || error("poset.leq must be a list-of-lists (length n=$(n)).")
    length(leq_any) == n || error("poset.leq must have n=$(n) rows")

    leq = falses(n, n)
    for i in 1:n
        row_any = leq_any[i]
        row_any isa AbstractVector || error("poset.leq row $(i) must be a list of Bool (length n=$(n)).")
        length(row_any) == n || error("poset.leq row length mismatch (row $(i); expected n=$(n)).")
        for j in 1:n
            x = row_any[j]
            x isa Bool || error("poset.leq entries must be Bool (row $(i), col $(j)).")
            leq[i, j] = x
        end
    end
    P = FiniteFringe.FinitePoset(leq)

    function parse_mask(entry, name::String)
        entry isa AbstractVector || error("$(name) entries must be Bool masks (length n=$(P.n)).")
        length(entry) == P.n || error("$(name) mask must have length n=$(P.n)")
        mask = BitVector(undef, P.n)
        for i in 1:P.n
            x = entry[i]
            x isa Bool || error("$(name) mask entries must be Bool (at index $(i)).")
            mask[i] = x
        end
        return mask
    end

    haskey(obj, "U") || error("Encoding JSON missing required key 'U'.")
    haskey(obj, "D") || error("Encoding JSON missing required key 'D'.")

    U_any = obj["U"]
    D_any = obj["D"]
    U_any isa AbstractVector || error("'U' must be a list of Bool masks.")
    D_any isa AbstractVector || error("'D' must be a list of Bool masks.")

    U = Vector{FiniteFringe.Upset}(undef, length(U_any))
    for t in 1:length(U_any)
        mask = parse_mask(U_any[t], "U")
        Uc = FiniteFringe.upset_closure(P, mask)
        Uc.mask == mask || error("U entry $(t) is not an upset mask")
        U[t] = Uc
    end

    D = Vector{FiniteFringe.Downset}(undef, length(D_any))
    for t in 1:length(D_any)
        mask = parse_mask(D_any[t], "D")
        Dc = FiniteFringe.downset_closure(P, mask)
        Dc.mask == mask || error("D entry $(t) is not a downset mask")
        D[t] = Dc
    end

    # Parse QQ entries: prefer exact rationals "a/b", but allow integer JSON numbers too.
    parse_QQ(x) = begin
        if x isa Integer
            return QQ(BigInt(x))
        end
        s = String(x)
        if occursin("/", s)
            return string_to_rational(s)
        end
        return QQ(parse(BigInt, strip(s)))
    end

    m = length(D)
    k = length(U)

    haskey(obj, "phi") || error("Encoding JSON missing required key 'phi'.")
    phi_any = obj["phi"]
    phi_any isa AbstractVector || error("'phi' must be a list-of-lists (size m x k).")
    length(phi_any) == m || error("phi must have m=$(m) rows")

    Phi = zeros(QQ, m, k)
    for i in 1:m
        row = phi_any[i]
        row isa AbstractVector || error("phi row $(i) must be a list (length k=$(k)).")
        length(row) == k || error("phi row length mismatch (row $(i); expected k=$(k))")
        for j in 1:k
            Phi[i, j] = parse_QQ(row[j])
        end
    end

    return FiniteFringe.FringeModule{QQ}(P, U, D, Phi)
end

# =============================================================================
# B) External adapters (CAS ingestion)
# =============================================================================

"""
JSON schema expected from an external CAS (Macaulay2, Singular, ...):

{
  "n": 3,                                   // ambient dimension
  "field": "QQ",                            // base field (informational)
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
function parse_flange_json(json_src)::Flange{QQ}
    obj = JSON3.read(json_src)
    n = Int(obj["n"])

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

    flats = IndFlat[]
    for f in obj["flats"]
        b = Vector{Int}(f["b"])
        tau = _mkface(n, f["tau"])
        id = Symbol(String(get(f, "id", "F")))
        push!(flats, IndFlat(tau, b; id=id))
    end

    injectives = IndInj[]
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
        M = zeros(QQ, m, ncol)
        @assert length(A) == m "phi: wrong number of rows"
        for i in 1:m
            row = A[i]
            @assert length(row) == ncol "phi: wrong number of cols"
            for j in 1:ncol
                val = row[j]
                if val isa String
                    M[i, j] = string_to_rational(val)
                elseif val isa Integer
                    M[i, j] = QQ(val)
                else
                    @warn "phi entry is a non-integer numeric; prefer exact strings \"num/den\" for exactness"
                    M[i, j] = QQ(val)
                end
            end
        end
        M
    else
        canonical_matrix(flats, injectives)
    end

    return Flange{QQ}(n, flats, injectives, Phi)
end

"""
    flange_from_m2(cmd::Cmd; jsonpath=nothing) -> Flange{QQ}

Run a CAS command that prints (or writes) the JSON described in `parse_flange_json`,
then parse it into a `Flange{QQ}`.
"""
function flange_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing)
    if jsonpath === nothing
        io = read(cmd, String)
        return parse_flange_json(io)
    end
    run(cmd)
    open(jsonpath, "r") do io
        return parse_flange_json(io)
    end
end

# =============================================================================
# C) Invariant caches (MPPI)
# =============================================================================

# MPPI types live in `PosetModules.Invariants`. We intentionally do NOT import
# them here to avoid include-order constraints. Instead, we fetch the module
# lazily when the MPPI JSON functions are called.

@inline function _invariants_module()
    PM = parentmodule(@__MODULE__)
    isdefined(PM, :Invariants) || error("MPPI JSON: PosetModules.Invariants is not loaded.")
    return getfield(PM, :Invariants)
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

function _mpp_decomposition_from_dict(obj)
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPDecomposition"
        error("MPPI JSON: expected kind == 'MPPDecomposition'")
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

function _mpp_image_from_dict(obj)
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPImage"
        error("MPPI JSON: expected kind == 'MPPImage'")
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
    decomp = _mpp_decomposition_from_dict(obj["decomp"])

    return Image(xgrid, ygrid, imgmat, sig, decomp)
end

"""
    save_mpp_decomposition_json(path, decomp)

Save an `MPPDecomposition` to a JSON file.

This is a good cache point: the decomposition contains the slice tracks and weights,
but not the full image grid. After loading, evaluate images via `mpp_image(decomp; ...)`.

Returns `path`.
"""
function save_mpp_decomposition_json(path::AbstractString, decomp)
    obj = _mpp_decomposition_to_dict(decomp)
    return _json_write(path, obj)
end

"""
    load_mpp_decomposition_json(path)

Load an `MPPDecomposition` written by `save_mpp_decomposition_json`.
"""
function load_mpp_decomposition_json(path::AbstractString)
    obj = _json_read(path)
    return _mpp_decomposition_from_dict(obj)
end

"""
    save_mpp_image_json(path, img; include_decomp=true)

Save an `MPPImage` (including its decomposition by default) to a JSON file.

Returns `path`.
"""
function save_mpp_image_json(path::AbstractString, img; include_decomp::Bool=true)
    obj = _mpp_image_to_dict(img; include_decomp=include_decomp)
    return _json_write(path, obj)
end

"""
    load_mpp_image_json(path)

Load an `MPPImage` written by `save_mpp_image_json`.

Note: `load_mpp_image_json` requires that the JSON contains a `"decomp"` field.
"""
function load_mpp_image_json(path::AbstractString)
    obj = _json_read(path)
    return _mpp_image_from_dict(obj)
end

end # module Serialization