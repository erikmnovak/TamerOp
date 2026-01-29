module Serialization
# -----------------------------------------------------------------------------
# JSON serialization for:
#   - Flange over Z^n (FlangeZn.Flange)
#   - Finite encoding (FiniteFringe.FinitePoset + FringeModule)
# Exactness: scalars are QQ encoded as "num/den".
# -----------------------------------------------------------------------------

using JSON3
using SparseArrays: spzeros
using ..CoreModules: QQ, rational_to_string, string_to_rational
import ..FlangeZn: Face, IndFlat, IndInj, Flange
import ..FiniteFringe: FinitePoset, FringeModule
using ..FiniteFringe

export save_flange_json, load_flange_json, save_encoding_json, load_encoding_json

# -------------------- Flange (Z^n) ---------------------------------------------
"""
    save_flange_json(path, FG::FlangeZn.Flange)

Schema:
{
  "kind": "FlangeZn",
  "n": n,
  "flats":      [ {"b":[...], "tau":[i1,i2,...]}, ... ],
  "injectives": [ {"b":[...], "tau":[...]} , ... ],
  "phi": [[ "num/den", ...], ...]   # rows = #injectives, cols = #flats
}
"""
function save_flange_json(path::AbstractString, FG::Flange)
    n = FG.n
    flats = [Dict("b"=>F.b, "tau"=>findall(identity, F.tau.coords)) for F in FG.flats]
    injectives = [Dict("b"=>E.b, "tau"=>findall(identity, E.tau.coords)) for E in FG.injectives]
    phi = [ [rational_to_string(FG.Phi[i,j]) for j in 1:length(FG.flats)]
                                       for i in 1:length(FG.injectives) ]
    obj = Dict("kind"=>"FlangeZn", "n"=>n, "flats"=>flats, "injectives"=>injectives, "phi"=>phi)
    open(path, "w") do io; JSON3.write(io, obj; allow_inf=true, indent=2); end
    path
end

"Inverse of `save_flange_json`."
function load_flange_json(path::AbstractString)
    obj = open(JSON3.read, path)
    @assert haskey(obj, "kind") && String(obj["kind"]) == "FlangeZn"
    n = Int(obj["n"])
    mkface(idxs) = Face(n, begin m=falses(n); for t in idxs; m[Int(t)] = true; end; m end)
    flats      = [ IndFlat(Vector{Int}(f["b"]), mkface(Vector{Int}(f["tau"])), :F) for f in obj["flats"] ]
    injectives = [ IndInj(Vector{Int}(e["b"]), mkface(Vector{Int}(e["tau"])), :E) for e in obj["injectives"] ]
    m = length(injectives); k = length(flats)
    Phi = Matrix{QQ}(undef, m, k)
    for i in 1:m, j in 1:k
        Phi[i,j] = string_to_rational(String(obj["phi"][i][j]))
    end
    Flange{QQ}(n, flats, injectives, Phi)
end

# -------------------- Finite encodings (P, H) -----------------------------------

# Save a finite poset and its fringe module (U, D, phi) to JSON.
# We store 'leq' as a dense boolean matrix, each Upset/Downset as a bit mask,
# and 'phi' with exact rationals as "num/den" strings.

function save_encoding_json(path::AbstractString, H::FringeModule{QQ})
    P = H.P
    # Serialize the partial order as a boolean matrix
    leq = [[P.leq[i,j] for j in 1:P.n] for i in 1:P.n]
    # Serialize upsets and downsets as bit vectors
    U_masks = [collect(Bool, U.mask) for U in H.U]
    D_masks = [collect(Bool, D.mask) for D in H.D]
    # Serialize phi exactly
    m, n = size(H.phi)
    phi = [ [rational_to_string(H.phi[i,j]) for j in 1:n] for i in 1:m ]
    obj = Dict(
        "kind" => "FiniteEncodingFringe",
        "poset" => Dict("n" => P.n, "leq" => leq),
        "U" => U_masks,
        "D" => D_masks,
        "phi" => phi
    )
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

# Load the same schema back into a FringeModule{QQ}.
# This loader is strict: it expects the schema emitted by save_encoding_json.
#
# Schema:
# {
#   "kind": "FiniteEncodingFringe",
#   "poset": { "n": n, "leq": [[Bool, ...], ...] },
#   "U": [[Bool, ...], ...],
#   "D": [[Bool, ...], ...],
#   "phi": [["a/b", ...], ...]
# }
function load_encoding_json(path::AbstractString)
    obj = open(JSON3.read, path)

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
        mask
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


end # module
