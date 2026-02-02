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
using SparseArrays
using Random

using ..CoreModules: QQ, rational_to_string, string_to_rational
import ..FlangeZn: Face, IndFlat, IndInj, Flange, canonical_matrix
import ..FiniteFringe: AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
                       FringeModule, nvertices, leq_matrix
import ..ZnEncoding: SignaturePoset
using ..FiniteFringe
using ..Modules: PModule
using ..DataPipeline: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex, FiltrationSpec, GridEncodingMap
import ..ZnEncoding
import ..PLBackend

export save_flange_json, load_flange_json,
       save_encoding_json, load_encoding_json,
       parse_flange_json, flange_from_m2,
       save_mpp_decomposition_json, load_mpp_decomposition_json,
       save_mpp_image_json, load_mpp_image_json,
       save_dataset_json, load_dataset_json,
       save_pipeline_json, load_pipeline_json,
       load_gudhi_json, load_ripserer_json, load_eirene_json,
       load_gudhi_txt, load_ripserer_txt, load_eirene_txt,
       load_ripser_point_cloud, load_ripser_distance,
       load_ripser_lower_distance, load_ripser_upper_distance,
       load_ripser_sparse_triplet, load_ripser_binary_lower_distance,
       load_dipha_distance_matrix,
       load_boundary_complex_json, load_reduced_complex_json,
       load_pmodule_json,
       load_ripser_lower_distance_streaming

# Schema versions for JSON formats
const PIPELINE_SCHEMA_VERSION = 1
const ENCODING_SCHEMA_VERSION = 2

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

# -----------------------------------------------------------------------------
# A0) Datasets + pipeline specs (DataPipeline)
# -----------------------------------------------------------------------------

function _obj_from_dataset(data)
    if data isa PointCloud
        return Dict("kind" => "PointCloud",
                    "points" => [collect(p) for p in data.points])
    elseif data isa ImageNd
        return Dict("kind" => "ImageNd",
                    "size" => collect(size(data.data)),
                    "data" => collect(vec(data.data)))
    elseif data isa GraphData
        return Dict("kind" => "GraphData",
                    "n" => data.n,
                    "edges" => [collect(e) for e in data.edges],
                    "coords" => data.coords === nothing ? nothing : [collect(c) for c in data.coords],
                    "weights" => data.weights === nothing ? nothing : collect(data.weights))
    elseif data isa EmbeddedPlanarGraph2D
        return Dict("kind" => "EmbeddedPlanarGraph2D",
                    "vertices" => [collect(v) for v in data.vertices],
                    "edges" => [collect(e) for e in data.edges],
                    "polylines" => data.polylines === nothing ? nothing : [[collect(p) for p in poly] for poly in data.polylines],
                    "bbox" => data.bbox === nothing ? nothing : collect(data.bbox))
    elseif data isa GradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        return Dict("kind" => "GradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [collect(g) for g in data.grades],
                    "cell_dims" => collect(data.cell_dims))
    else
        error("Unsupported dataset type for serialization.")
    end
end

function _dataset_from_obj(obj)
    kind = String(obj["kind"])
    if kind == "PointCloud"
        pts = [Vector{Float64}(p) for p in obj["points"]]
        return PointCloud(pts)
    elseif kind == "ImageNd"
        sz = Vector{Int}(obj["size"])
        flat = Vector{Float64}(obj["data"])
        data = reshape(flat, Tuple(sz))
        return ImageNd(data)
    elseif kind == "GraphData"
        n = Int(obj["n"])
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        coords = obj["coords"] === nothing ? nothing : [Vector{Float64}(c) for c in obj["coords"]]
        weights = obj["weights"] === nothing ? nothing : Vector{Float64}(obj["weights"])
        return GraphData(n, edges; coords=coords, weights=weights, T=Float64)
    elseif kind == "EmbeddedPlanarGraph2D"
        verts = [Vector{Float64}(v) for v in obj["vertices"]]
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        polylines = obj["polylines"] === nothing ? nothing :
            [[Vector{Float64}(p) for p in poly] for poly in obj["polylines"]]
        bbox = obj["bbox"] === nothing ? nothing : (Float64(obj["bbox"][1]),
                                                   Float64(obj["bbox"][2]),
                                                   Float64(obj["bbox"][3]),
                                                   Float64(obj["bbox"][4]))
        return EmbeddedPlanarGraph2D(verts, edges; polylines=polylines, bbox=bbox)
    elseif kind == "GradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = [Vector{Float64}(g) for g in obj["grades"]]
        cell_dims = Vector{Int}(obj["cell_dims"])
        return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    else
        error("Unknown dataset kind: $kind")
    end
end

function _spec_obj(spec::FiltrationSpec)
    return Dict("kind" => String(spec.kind), "params" => Dict(pairs(spec.params)))
end

function _spec_from_obj(obj)
    kind = Symbol(String(obj["kind"]))
    params_obj = obj["params"]
    params = (; (Symbol(k) => params_obj[k] for k in keys(params_obj))...)
    return FiltrationSpec(; kind=kind, params...)
end

"""
    save_dataset_json(path, data)

Serialize a dataset (PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex).
"""
function save_dataset_json(path::AbstractString, data)
    return _json_write(path, _obj_from_dataset(data))
end

"""
    load_dataset_json(path)

Load a dataset serialized by `save_dataset_json`.
"""
function load_dataset_json(path::AbstractString)
    obj = _json_read(path)
    return _dataset_from_obj(obj)
end

"""
    save_pipeline_json(path, data, spec; degree=nothing, opts=nothing)

Serialize a dataset + filtration spec (and optional options) in a single JSON.
"""
function save_pipeline_json(path::AbstractString, data, spec::FiltrationSpec; degree=nothing, opts=nothing)
    obj = Dict(
        "schema_version" => PIPELINE_SCHEMA_VERSION,
        "dataset" => _obj_from_dataset(data),
        "spec" => _spec_obj(spec),
        "degree" => degree,
        "opts" => opts,
    )
    return _json_write(path, obj)
end

"""
    load_pipeline_json(path) -> (data, spec, degree, opts)

Inverse of `save_pipeline_json`.
"""
function load_pipeline_json(path::AbstractString)
    obj = _json_read(path)
    version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 1
    version <= PIPELINE_SCHEMA_VERSION || error("Unsupported pipeline JSON schema_version: $(version)")
    data = _dataset_from_obj(obj["dataset"])
    spec = _spec_from_obj(obj["spec"])
    degree = haskey(obj, "degree") ? obj["degree"] : nothing
    opts = haskey(obj, "opts") ? obj["opts"] : nothing
    return data, spec, degree, opts
end

# =============================================================================
# B0) Interop adapters: GUDHI / Ripserer / Eirene (JSON)
# =============================================================================

function _simplicial_boundary_from_lists(simplices::Vector{Vector{Int}},
                                         faces::Vector{Vector{Int}})
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    for (j, s) in enumerate(simplices)
        k = length(s)
        for i in 1:k
            f = [s[t] for t in 1:k if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

function _graded_complex_from_simplex_list(simplices::Vector{Vector{Int}}, grades_any::AbstractVector)
    length(simplices) == length(grades_any) ||
        error("simplices and grades length mismatch.")
    max_dim = maximum(length.(simplices)) - 1
    by_dim = [Vector{Vector{Int}}() for _ in 0:max_dim]
    grades = Vector{Vector{Float64}}()
    for (s, g) in zip(simplices, grades_any)
        d = length(s) - 1
        push!(by_dim[d+1], s)
        if g isa AbstractVector
            push!(grades, Vector{Float64}(g))
        else
            push!(grades, [Float64(g)])
        end
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:length(by_dim)
        push!(boundaries, _simplicial_boundary_from_lists(by_dim[d], by_dim[d-1]))
    end
    cells = [collect(1:length(by_dim[d])) for d in 1:length(by_dim)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_gudhi_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]   # or list-of-lists for multiparameter
}
"""
function load_gudhi_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_ripserer_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_ripserer_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_eirene_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_eirene_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_gudhi_txt(path) -> GradedComplex
    load_ripserer_txt(path) -> GradedComplex
    load_eirene_txt(path) -> GradedComplex

Parse a simplex filtration from a text file with one simplex per line.

Supported line formats (whitespace-separated):
1) "dim v1 v2 ... vk filtration"
2) "v1 v2 ... vk filtration"   (dimension inferred from count)

Blank lines and lines starting with '#' are ignored.
"""
function _load_simplex_filtration_txt(path::AbstractString)
    lines = String[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            push!(lines, line)
        end
    end

    # Heuristic: if any line has exactly two tokens, treat the file as
    # "vertices + filtration" (no leading dimension token). Otherwise, use
    # the "dim v1 v2 ... filtration" format.
    has_dim_prefix = true
    for line in lines
        parts = split(line)
        length(parts) >= 2 || error("Invalid simplex line: '$line'")
        if length(parts) == 2
            has_dim_prefix = false
            break
        end
    end

    simplices = Vector{Vector{Int}}()
    grades = Vector{Float64}()
    for line in lines
        parts = split(line)
        if has_dim_prefix
            dim = parse(Int, parts[1])
            verts = [parse(Int, parts[i]) for i in 2:(dim+2)]
            filt = parse(Float64, parts[dim+3])
            push!(simplices, verts)
            push!(grades, filt)
        else
            verts = [parse(Int, parts[i]) for i in 1:(length(parts)-1)]
            filt = parse(Float64, parts[end])
            push!(simplices, verts)
            push!(grades, filt)
        end
    end

    return _graded_complex_from_simplex_list(simplices, grades)
end

load_gudhi_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_ripserer_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_eirene_txt(path::AbstractString) = _load_simplex_filtration_txt(path)

# -----------------------------------------------------------------------------
# B2) Interop adapters: boundary/reduced complexes and direct PModules
# -----------------------------------------------------------------------------

function _parse_poset_from_obj(poset_obj)
    kind = haskey(poset_obj, "kind") ? String(poset_obj["kind"]) : "FinitePoset"
    if kind == "FinitePoset"
        haskey(poset_obj, "n") || error("poset missing required key 'n'.")
        haskey(poset_obj, "leq") || error("poset missing required key 'leq'.")
        n = Int(poset_obj["n"])
        leq_any = poset_obj["leq"]
        leq_any isa AbstractVector || error("poset.leq must be a list-of-lists.")
        length(leq_any) == n || error("poset.leq must have n=$(n) rows")
        leq = falses(n, n)
        for i in 1:n
            row = leq_any[i]
            row isa AbstractVector || error("poset.leq row $(i) must be a list.")
            length(row) == n || error("poset.leq row length mismatch (row $(i); expected n=$(n)).")
            for j in 1:n
                x = row[j]
                x isa Bool || error("poset.leq entries must be Bool (row $(i), col $(j)).")
                leq[i, j] = x
            end
        end
        return FinitePoset(leq)
    elseif kind == "ProductOfChainsPoset"
        haskey(poset_obj, "sizes") || error("ProductOfChainsPoset missing required key 'sizes'.")
        sizes = Vector{Int}(poset_obj["sizes"])
        return ProductOfChainsPoset(sizes)
    elseif kind == "GridPoset"
        haskey(poset_obj, "coords") || error("GridPoset missing required key 'coords'.")
        coords_any = poset_obj["coords"]
        coords_any isa AbstractVector || error("GridPoset.coords must be a list-of-lists.")
        coords = ntuple(i -> Vector{Float64}(coords_any[i]), length(coords_any))
        return GridPoset(coords)
    elseif kind == "ProductPoset"
        haskey(poset_obj, "left") || error("ProductPoset missing required key 'left'.")
        haskey(poset_obj, "right") || error("ProductPoset missing required key 'right'.")
        P1 = _parse_poset_from_obj(poset_obj["left"])
        P2 = _parse_poset_from_obj(poset_obj["right"])
        return ProductPoset(P1, P2)
    elseif kind == "SignaturePoset"
        haskey(poset_obj, "sig_y") || error("SignaturePoset missing required key 'sig_y'.")
        haskey(poset_obj, "sig_z") || error("SignaturePoset missing required key 'sig_z'.")
        sig_y_any = poset_obj["sig_y"]
        sig_z_any = poset_obj["sig_z"]
        sig_y_any isa AbstractVector || error("SignaturePoset.sig_y must be a list-of-lists.")
        sig_z_any isa AbstractVector || error("SignaturePoset.sig_z must be a list-of-lists.")
        sig_y = Vector{BitVector}(undef, length(sig_y_any))
        sig_z = Vector{BitVector}(undef, length(sig_z_any))
        for i in 1:length(sig_y_any)
            row = sig_y_any[i]
            row isa AbstractVector || error("SignaturePoset.sig_y row $(i) must be a list.")
            sig_y[i] = BitVector(Bool[x for x in row])
        end
        for i in 1:length(sig_z_any)
            row = sig_z_any[i]
            row isa AbstractVector || error("SignaturePoset.sig_z row $(i) must be a list.")
            sig_z[i] = BitVector(Bool[x for x in row])
        end
        return SignaturePoset(sig_y, sig_z)
    else
        error("Unsupported poset kind: $(kind)")
    end
end

function _poset_obj(P::AbstractPoset; include_leq::Bool=true)
    if P isa FinitePoset
        include_leq || error("Cannot omit leq for FinitePoset serialization.")
        L = leq_matrix(P)
        leq = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => leq)
    elseif P isa ProductOfChainsPoset
        obj = Dict("kind" => "ProductOfChainsPoset",
                   "n" => nvertices(P),
                   "sizes" => collect(P.sizes))
        if include_leq
            L = leq_matrix(P)
            obj["leq"] = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        end
        return obj
    elseif P isa GridPoset
        obj = Dict("kind" => "GridPoset",
                   "n" => nvertices(P),
                   "coords" => [collect(c) for c in P.coords])
        if include_leq
            L = leq_matrix(P)
            obj["leq"] = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        end
        return obj
    elseif P isa ProductPoset
        obj = Dict("kind" => "ProductPoset",
                   "n" => nvertices(P),
                   "left" => _poset_obj(P.P1; include_leq=include_leq),
                   "right" => _poset_obj(P.P2; include_leq=include_leq))
        if include_leq
            L = leq_matrix(P)
            obj["leq"] = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        end
        return obj
    elseif P isa SignaturePoset
        obj = Dict("kind" => "SignaturePoset",
                   "n" => nvertices(P),
                   "sig_y" => [collect(row) for row in P.sig_y],
                   "sig_z" => [collect(row) for row in P.sig_z])
        if include_leq
            L = leq_matrix(P)
            obj["leq"] = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        end
        return obj
    else
        L = leq_matrix(P)
        leq = [[L[i, j] for j in 1:size(L, 2)] for i in 1:size(L, 1)]
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => leq)
    end
end

function _parse_QQ(x)
    if x isa Integer
        return QQ(BigInt(x))
    end
    if x isa AbstractFloat
        return QQ(x)
    end
    s = String(x)
    if occursin("/", s)
        return string_to_rational(s)
    end
    return QQ(parse(BigInt, strip(s)))
end

"""
    load_boundary_complex_json(path) -> GradedComplex

Expected schema (external adapter):
{
  "cells_by_dim": [[1,2,...], [1,2,...], ...]  // or "counts_by_dim": [n0, n1, ...]
  "boundaries": [ {"m":..,"n":..,"I":[..],"J":[..],"V":[..]}, ... ],
  "grades": [ [..], [..], ... ],
  "cell_dims": [..]   // optional
}
"""
function load_boundary_complex_json(path::AbstractString)
    obj = _json_read(path)
    cells = if haskey(obj, "cells_by_dim")
        [Vector{Int}(c) for c in obj["cells_by_dim"]]
    elseif haskey(obj, "counts_by_dim")
        counts = Vector{Int}(obj["counts_by_dim"])
        out = Vector{Vector{Int}}(undef, length(counts))
        for d in 1:length(counts)
            out[d] = collect(1:counts[d])
        end
        out
    else
        error("boundary complex JSON missing 'cells_by_dim' or 'counts_by_dim'.")
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for b in obj["boundaries"]
        m = Int(b["m"]); n = Int(b["n"])
        I = Vector{Int}(b["I"])
        J = Vector{Int}(b["J"])
        V = Vector{Int}(b["V"])
        push!(boundaries, sparse(I, J, V, m, n))
    end
    grades = [Vector{Float64}(g) for g in obj["grades"]]
    cell_dims = haskey(obj, "cell_dims") ? Vector{Int}(obj["cell_dims"]) : nothing
    return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
end

"""
    load_reduced_complex_json(path) -> GradedComplex

Alias for `load_boundary_complex_json`, intended for reduced boundary matrices.
"""
load_reduced_complex_json(path::AbstractString) = load_boundary_complex_json(path)

"""
    load_pmodule_json(path) -> PModule{QQ}

Expected schema:
{
  "poset": { "kind": "FinitePoset", "n": n, "leq": [[...]] },
  "dims": [d1, d2, ...],
  "edges": [ {"src": i, "dst": j, "mat": [[...]]}, ... ]
}
"""
function load_pmodule_json(path::AbstractString)
    obj = _json_read(path)
    haskey(obj, "poset") || error("pmodule JSON missing 'poset'.")
    P = _parse_poset_from_obj(obj["poset"])
    haskey(obj, "dims") || error("pmodule JSON missing 'dims'.")
    dims = Vector{Int}(obj["dims"])
    length(dims) == nvertices(P) || error("pmodule dims length mismatch with poset size.")
    haskey(obj, "edges") || error("pmodule JSON missing 'edges'.")
    edge_maps = Dict{Tuple{Int,Int},Matrix{QQ}}()
    for e in obj["edges"]
        src = Int(e["src"])
        dst = Int(e["dst"])
        mat_any = e["mat"]
        mat_any isa AbstractVector || error("pmodule edge mat must be a list-of-lists.")
        m = length(mat_any)
        n = m == 0 ? 0 : length(mat_any[1])
        M = zeros(QQ, m, n)
        for i in 1:m
            row = mat_any[i]
            length(row) == n || error("pmodule edge mat row length mismatch.")
            for j in 1:n
                M[i, j] = _parse_QQ(row[j])
            end
        end
        edge_maps[(src, dst)] = M
    end
    return PModule{QQ}(P, dims, edge_maps)
end

# -----------------------------------------------------------------------------
# B1) Interop adapters: Ripser/DIPHA distance matrix formats
# -----------------------------------------------------------------------------

function _read_numeric_rows(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            line = replace(line, ';' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            push!(rows, row)
        end
    end
    return rows
end

function _matrix_from_rows(rows::Vector{Vector{Float64}})
    isempty(rows) && error("distance matrix: empty file.")

    if length(rows) >= 2 && length(rows[1]) == 1
        n_header = Int(round(rows[1][1]))
        if n_header >= 1 && length(rows) - 1 == n_header &&
           all(length(rows[i]) == n_header for i in 2:length(rows))
            dist = zeros(Float64, n_header, n_header)
            for i in 1:n_header
                dist[i, :] = rows[i + 1]
            end
            return dist
        end
    end

    same_len = all(length(r) == length(rows[1]) for r in rows)
    if same_len
        m = length(rows)
        k = length(rows[1])
        if m == k
            dist = zeros(Float64, m, m)
            for i in 1:m
                dist[i, :] = rows[i]
            end
            return dist
        elseif k == 1
            vals = [r[1] for r in rows]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        elseif m == 1
            vals = rows[1]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        end
    end

    vals = reduce(vcat, rows)
    n = round(Int, sqrt(length(vals)))
    n * n == length(vals) || error("distance matrix: could not infer square size.")
    dist = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = vals[(i - 1) * n + j]
    end
    return dist
end

function _infer_n_from_triangular_len(len::Int)
    n1 = Int(floor((sqrt(1 + 8 * len) - 1) / 2))
    if div(n1 * (n1 + 1), 2) == len
        return n1, true
    end
    n2 = Int(floor((1 + sqrt(1 + 8 * len)) / 2))
    if div(n2 * (n2 - 1), 2) == len
        return n2, false
    end
    return 0, false
end

function _triangular_from_rows(rows::Vector{Vector{Float64}}; upper::Bool)
    n = length(rows)
    if upper
        include_diag = if all(length(rows[i]) == n - i + 1 for i in 1:n)
            true
        elseif all(length(rows[i]) == n - i for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for k in 1:(n - i + 1)
                    j = i + k - 1
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            else
                for k in 1:(n - i)
                    j = i + k
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    else
        include_diag = if all(length(rows[i]) == i for i in 1:n)
            true
        elseif all(length(rows[i]) == i - 1 for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for j in 1:i
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            else
                for j in 1:(i - 1)
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    end
end

function _triangular_from_vals(vals::Vector{Float64}; upper::Bool)
    n, include_diag = _infer_n_from_triangular_len(length(vals))
    n > 0 || error("triangular distance list length is not valid.")
    dist = fill(Inf, n, n)
    idx = 1
    if upper
        for i in 1:n
            if include_diag
                for j in i:n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in (i + 1):n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    else
        for i in 1:n
            if include_diag
                for j in 1:i
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in 1:(i - 1)
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    end
    return dist
end

function _combinations(n::Int, k::Int)
    if k == 0
        return [Int[]]
    end
    out = Vector{Vector{Int}}()
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i in start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

function _graded_complex_from_distance_matrix(dist::AbstractMatrix{<:Real};
                                              max_dim::Int=1,
                                              radius=nothing,
                                              max_simplices=nothing,
                                              sparse_rips::Bool=false,
                                              approx_rips::Bool=false,
                                              max_edges=nothing,
                                              max_degree=nothing,
                                              sample_frac=nothing)
    size(dist, 1) == size(dist, 2) || error("distance matrix must be square.")
    n = size(dist, 1)
    n > 0 || error("distance matrix has size 0.")

    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [ [i] for i in 1:n ]
    total = length(simplices[1])

    if sparse_rips
        radius === nothing && error("sparse_rips requires radius.")
        sims = Vector{Vector{Int}}()
        for i in 1:n, j in i+1:n
            if dist[i, j] <= radius
                push!(sims, [i, j])
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("distance matrix Rips: exceeded max_simplices=$(max_simplices).")
        end
    elseif approx_rips
        radius === nothing && error("approx_rips requires radius.")
        sims = Vector{Vector{Int}}()
        deg = zeros(Int, n)
        for i in 1:n
            for j in i+1:n
                if dist[i, j] <= radius
                    if max_degree !== nothing && (deg[i] >= max_degree || deg[j] >= max_degree)
                        continue
                    end
                    if sample_frac !== nothing && rand() > sample_frac
                        continue
                    end
                    push!(sims, [i, j])
                    deg[i] += 1
                    deg[j] += 1
                    if max_edges !== nothing && length(sims) >= max_edges
                        break
                    end
                end
            end
            if max_edges !== nothing && length(sims) >= max_edges
                break
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("distance matrix Rips: exceeded max_simplices=$(max_simplices).")
        end
    else
        for k in 2:max_dim+1
            sims = Vector{Vector{Int}}()
            for comb in _combinations(n, k)
                push!(sims, comb)
            end
            simplices[k] = sims
            total += length(sims)
            if max_simplices !== nothing && total > max_simplices
                error("distance matrix Rips: exceeded max_simplices=$(max_simplices).")
            end
        end
    end

    grades = Vector{Vector{Float64}}()
    for _ in simplices[1]
        push!(grades, [0.0])
    end
    for k in 2:max_dim+1
        for s in simplices[k]
            maxd = 0.0
            for i in 1:length(s)
                for j in (i+1):length(s)
                    d = Float64(dist[s[i], s[j]])
                    if d > maxd
                        maxd = d
                    end
                end
            end
            push!(grades, [maxd])
        end
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:max_dim+1
        Bk = _simplicial_boundary_from_lists(simplices[k], simplices[k-1])
        push!(boundaries, Bk)
    end
    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_ripser_point_cloud(path) -> PointCloud

Parse a Ripser-style point cloud (whitespace-separated coordinates per line).
"""
function load_ripser_point_cloud(path::AbstractString)
    rows = _read_numeric_rows(path)
    isempty(rows) && error("point cloud file has no points.")
    return PointCloud([Vector{Float64}(r) for r in rows])
end

"""
    load_ripser_distance(path; max_dim=1, kwargs...) -> GradedComplex

Parse a full distance matrix (square) and build a 1-parameter Rips graded complex.
"""
function load_ripser_distance(path::AbstractString; max_dim::Int=1, kwargs...)
    rows = _read_numeric_rows(path)
    dist = _matrix_from_rows(rows)
    return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
end

"""
    load_ripser_lower_distance(path; max_dim=1, kwargs...) -> GradedComplex

Parse a lower-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_lower_distance(path::AbstractString; max_dim::Int=1, kwargs...)
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=false)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=false)
    end
    return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
end

"""
    load_ripser_upper_distance(path; max_dim=1, kwargs...) -> GradedComplex

Parse an upper-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_upper_distance(path::AbstractString; max_dim::Int=1, kwargs...)
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=true)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=true)
    end
    return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
end

"""
    load_ripser_sparse_triplet(path; max_dim=1, kwargs...) -> GradedComplex

Parse a sparse triplet format: each nonzero entry as "i j d".
Indices can be 0-based or 1-based.
"""
function load_ripser_sparse_triplet(path::AbstractString; max_dim::Int=1, kwargs...)
    rows = _read_numeric_rows(path)
    isempty(rows) && error("sparse triplet file is empty.")
    for row in rows
        length(row) == 3 || error("sparse triplet rows must have 3 entries.")
    end
    idxs = [Int(round(r[1])) for r in rows]
    jdxs = [Int(round(r[2])) for r in rows]
    base0 = any(i == 0 for i in idxs) || any(j == 0 for j in jdxs)
    if base0
        idxs .= idxs .+ 1
        jdxs .= jdxs .+ 1
    end
    n = max(maximum(idxs), maximum(jdxs))
    n > 0 || error("sparse triplet: could not infer matrix size.")
    dist = fill(Inf, n, n)
    for i in 1:n
        dist[i, i] = 0.0
    end
    for t in 1:length(rows)
        i = idxs[t]
        j = jdxs[t]
        d = Float64(rows[t][3])
        if d < dist[i, j]
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
end

"""
    load_ripser_binary_lower_distance(path; max_dim=1, kwargs...) -> GradedComplex

Parse Ripser's binary lower-triangular distance matrix (Float64 values).
"""
function load_ripser_binary_lower_distance(path::AbstractString; max_dim::Int=1, kwargs...)
    vals = Float64[]
    open(path, "r") do io
        while !eof(io)
            push!(vals, read(io, Float64))
        end
    end
    dist = _triangular_from_vals(vals; upper=false)
    return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
end

"""
    load_dipha_distance_matrix(path; max_dim=1, kwargs...) -> GradedComplex

Parse a DIPHA binary distance matrix and build a Rips graded complex.
"""
function load_dipha_distance_matrix(path::AbstractString; max_dim::Int=1, kwargs...)
    open(path, "r") do io
        eof(io) && error("DIPHA distance matrix: empty file.")
        magic = read(io, Int64)
        magic == 8067171840 || error("DIPHA: invalid magic value.")
        _ = read(io, Int64) # file type id
        n = Int(read(io, Int64))
        n > 0 || error("DIPHA: invalid matrix size.")
        vals = Vector{Float64}(undef, n * n)
        read!(io, vals)
        dist = zeros(Float64, n, n)
        idx = 1
        for i in 1:n, j in 1:n
            dist[i, j] = vals[idx]
            idx += 1
        end
        return _graded_complex_from_distance_matrix(dist; max_dim=max_dim, kwargs...)
    end
end

"""
    load_ripser_lower_distance_streaming(path; radius, max_dim=1, kwargs...) -> GradedComplex

Streaming reader for lower-triangular distance matrices (text). Builds a 1-skeleton
Rips complex without loading the full matrix.
"""
function load_ripser_lower_distance_streaming(path::AbstractString; radius, max_dim::Int=1, kwargs...)
    radius === nothing && error("streaming lower distance requires radius.")
    edges = Vector{Vector{Int}}()
    grades = Vector{Vector{Float64}}()
    n = 0
    include_diag = nothing
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            n += 1
            if include_diag === nothing
                if length(row) == n
                    include_diag = true
                elseif length(row) == n - 1
                    include_diag = false
                else
                    error("streaming lower distance: row length mismatch at row $(n).")
                end
            else
                expected = include_diag ? n : n - 1
                length(row) == expected || error("streaming lower distance: row length mismatch at row $(n).")
            end
            for j in 1:length(row)
                if include_diag && j == n
                    continue
                end
                d = row[j]
                if d <= radius
                    push!(edges, [j, n])
                    push!(grades, [d])
                end
            end
        end
    end
    n > 0 || error("streaming lower distance: no rows found.")
    vertices = [ [i] for i in 1:n ]
    cells = [collect(1:length(vertices)), collect(1:length(edges))]
    all_grades = Vector{Vector{Float64}}(undef, length(vertices) + length(edges))
    for i in 1:length(vertices)
        all_grades[i] = [0.0]
    end
    for i in 1:length(edges)
        all_grades[length(vertices) + i] = grades[i]
    end
    B1 = _simplicial_boundary_from_lists(edges, vertices)
    return GradedComplex(cells, [B1], all_grades)
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
@inline function _pi_to_obj(pi)
    if pi isa GridEncodingMap
        return Dict(
            "kind" => "GridEncodingMap",
            "coords" => [collect(ax) for ax in pi.coords],
            "orientation" => collect(pi.orientation),
        )
    elseif pi isa ZnEncoding.ZnEncodingMap
        return Dict(
            "kind" => "ZnEncodingMap",
            "n" => pi.n,
            "coords" => [collect(ax) for ax in pi.coords],
            "sig_y" => [collect(Bool, s) for s in pi.sig_y],
            "sig_z" => [collect(Bool, s) for s in pi.sig_z],
            "reps" => [collect(r) for r in pi.reps],
            "flats" => [Dict("b" => f.b, "tau" => findall(identity, f.tau.coords)) for f in pi.flats],
            "injectives" => [Dict("b" => e.b, "tau" => findall(identity, e.tau.coords)) for e in pi.injectives],
        )
    elseif pi isa PLBackend.PLEncodingMapBoxes
        return Dict(
            "kind" => "PLEncodingMapBoxes",
            "n" => pi.n,
            "coords" => [collect(ax) for ax in pi.coords],
            "sig_y" => [collect(Bool, s) for s in pi.sig_y],
            "sig_z" => [collect(Bool, s) for s in pi.sig_z],
            "reps" => [collect(r) for r in pi.reps],
            "Ups" => [collect(u.ell) for u in pi.Ups],
            "Downs" => [collect(d.u) for d in pi.Downs],
            "cell_shape" => collect(pi.cell_shape),
            "cell_strides" => collect(pi.cell_strides),
            "cell_to_region" => collect(pi.cell_to_region),
            "coord_flags" => [collect(f) for f in pi.coord_flags],
            "axis_is_uniform" => collect(pi.axis_is_uniform),
            "axis_step" => collect(pi.axis_step),
            "axis_min" => collect(pi.axis_min),
        )
    end
    error("Unsupported encoding map type for JSON serialization.")
end

function _pi_from_obj(P::AbstractPoset, obj)
    kind = String(obj["kind"])
    if kind == "GridEncodingMap"
        coords = tuple((Vector{Float64}(ax) for ax in obj["coords"])...)
        orientation = tuple((Int(o) for o in obj["orientation"])...)
        return GridEncodingMap(P, coords; orientation=orientation)
    elseif kind == "ZnEncodingMap"
        n = Int(obj["n"])
        coords = [Vector{Int}(ax) for ax in obj["coords"]]
        sig_y = [BitVector(s) for s in obj["sig_y"]]
        sig_z = [BitVector(s) for s in obj["sig_z"]]
        reps = [Vector{Int}(r) for r in obj["reps"]]
        mkface(idxs) = begin
            m = falses(n)
            for t in idxs
                m[Int(t)] = true
            end
            Face(n, m)
        end
        flats = [IndFlat(mkface(Vector{Int}(f["tau"])), Vector{Int}(f["b"]); id=:F)
                 for f in obj["flats"]]
        injectives = [IndInj(mkface(Vector{Int}(e["tau"])), Vector{Int}(e["b"]); id=:E)
                      for e in obj["injectives"]]
        sig_to_region = Dict{Tuple{Tuple,Tuple},Int}()
        for t in 1:length(sig_y)
            sig_to_region[(Tuple(sig_y[t]), Tuple(sig_z[t]))] = t
        end
        return ZnEncoding.ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats, injectives, sig_to_region)
    elseif kind == "PLEncodingMapBoxes"
        n = Int(obj["n"])
        coords = [Vector{Float64}(ax) for ax in obj["coords"]]
        sig_y = [BitVector(s) for s in obj["sig_y"]]
        sig_z = [BitVector(s) for s in obj["sig_z"]]
        reps = [Vector{Float64}(r) for r in obj["reps"]]
        Ups = [PLBackend.BoxUpset(Vector{Float64}(u)) for u in obj["Ups"]]
        Downs = [PLBackend.BoxDownset(Vector{Float64}(d)) for d in obj["Downs"]]
        cell_shape = Vector{Int}(obj["cell_shape"])
        cell_strides = Vector{Int}(obj["cell_strides"])
        cell_to_region = Vector{Int}(obj["cell_to_region"])
        coord_flags = [Vector{UInt8}(f) for f in obj["coord_flags"]]
        axis_is_uniform = BitVector(obj["axis_is_uniform"])
        axis_step = Vector{Float64}(obj["axis_step"])
        axis_min = Vector{Float64}(obj["axis_min"])
        MY = cld(length(Ups), 64)
        MZ = cld(length(Downs), 64)
        sig_to_region = Dict{PLBackend.SigKey{MY,MZ},Int}()
        for t in 1:length(sig_y)
            ywords = PLBackend._pack_bitvector_words(sig_y[t], Val(MY))
            zwords = PLBackend._pack_bitvector_words(sig_z[t], Val(MZ))
            sig_to_region[PLBackend.SigKey{MY,MZ}(ywords, zwords)] = t
        end
        return PLBackend.PLEncodingMapBoxes{MY,MZ}(n, coords, sig_y, sig_z, reps, Ups, Downs,
                                                  sig_to_region, cell_shape, cell_strides, cell_to_region,
                                                  coord_flags, axis_is_uniform, axis_step, axis_min)
    end
    error("Unsupported encoding map kind: $(kind)")
end

function _encoding_obj(H::FringeModule{QQ}; pi=nothing, include_leq::Bool=true)
    P = H.P

    U_masks = [collect(Bool, U.mask) for U in H.U]
    D_masks = [collect(Bool, D.mask) for D in H.D]

    m, n = size(H.phi)
    phi = [[rational_to_string(H.phi[i, j]) for j in 1:n] for i in 1:m]

    obj = Dict(
        "kind" => "FiniteEncodingFringe",
        "schema_version" => ENCODING_SCHEMA_VERSION,
        "poset" => _poset_obj(P; include_leq=include_leq),
        "U" => U_masks,
        "D" => D_masks,
        "phi" => phi,
    )
    if pi !== nothing
        obj["pi"] = _pi_to_obj(pi)
    end
    return obj
end

function save_encoding_json(path::AbstractString, H::FringeModule{QQ}; include_leq::Bool=true)
    return _json_write(path, _encoding_obj(H; include_leq=include_leq))
end

function save_encoding_json(path::AbstractString, P::AbstractPoset, H::FringeModule{QQ}, pi; include_leq::Bool=true)
    P === H.P || error("save_encoding_json: P does not match H.P.")
    return _json_write(path, _encoding_obj(H; pi=pi, include_leq=include_leq))
end

# Load the schema emitted by save_encoding_json.
#
# This loader is intentionally strict: it expects the schema emitted by
# save_encoding_json (missing required keys => error).
function load_encoding_json(path::AbstractString; return_pi::Bool=false)
    obj = _json_read(path)

    haskey(obj, "kind") || error("Encoding JSON missing required key 'kind'.")
    kind = String(obj["kind"])
    kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(kind)")
    version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 1
    version <= ENCODING_SCHEMA_VERSION || error("Unsupported encoding JSON schema_version: $(version)")

    haskey(obj, "poset") || error("Encoding JSON missing required key 'poset'.")
    poset_obj = obj["poset"]
    P = _parse_poset_from_obj(poset_obj)

    function parse_mask(entry, name::String)
        n = nvertices(P)
        entry isa AbstractVector || error("$(name) entries must be Bool masks (length n=$(n)).")
        length(entry) == n || error("$(name) mask must have length n=$(n)")
        mask = BitVector(undef, n)
        for i in 1:n
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

    H = FiniteFringe.FringeModule{QQ}(P, U, D, Phi)
    if return_pi && haskey(obj, "pi")
        return H, _pi_from_obj(P, obj["pi"])
    end
    return H
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
