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
    flats      = [ IndFlat{QQ}(Vector{Int}(f["b"]), mkface(Vector{Int}(f["tau"])), :F) for f in obj["flats"] ]
    injectives = [ IndInj{QQ}(Vector{Int}(e["b"]), mkface(Vector{Int}(e["tau"])), :E) for e in obj["injectives"] ]
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
# Load a finite poset + fringe module from JSON.
# Supports BOTH:
#   (new)    { "kind":"FiniteEncodingFringe", "poset":{...}, "U":[Bool masks], "D":[Bool masks], "phi":[["a/b"]...] }
#   (legacy) { "P":{...}, "U":[[indices]...], "D":[[indices]...], "phi":[["a/b"]...] }
function load_encoding_json(path::AbstractString)
    obj = open(JSON3.read, path)

    # "kind" is optional (backwards compatibility).
    if haskey(obj, "kind")
        k = String(obj["kind"])
        @assert k == "FiniteEncodingFringe" "Unsupported JSON kind: $(k)"
    end

    # Accept either "poset" (new) or "P" (legacy)
    poset_obj =
        haskey(obj, "poset") ? obj["poset"] :
        haskey(obj, "P")     ? obj["P"] :
        error("Encoding JSON must have key \"poset\" (new schema) or \"P\" (legacy schema).")

    n = Int(poset_obj["n"])

    leq_any = poset_obj["leq"]
    leq = falses(n, n)

    # Accept either a flat vector of length n*n (column-major) or a vector-of-vectors (n rows).
    if isa(leq_any, AbstractVector) && length(leq_any) == n*n && !(leq_any[1] isa AbstractVector)
        leq_vec = Vector{Bool}(leq_any)
        for i in 1:n, j in 1:n
            leq[i, j] = leq_vec[i + (j - 1) * n]
        end
    else
        rows = Vector{Any}(leq_any)
        @assert length(rows) == n "poset.leq must have n rows"
        for i in 1:n
            row = Vector{Bool}(rows[i])
            @assert length(row) == n "poset.leq row length mismatch"
            for j in 1:n
                leq[i, j] = row[j]
            end
        end
    end

    P = FiniteFringe.FinitePoset(leq)

    # U/D entries can be either:
    #   - Bool masks of length n (new schema), OR
    #   - index lists (legacy schema): treated as generator lists and closed.
    parse_upset(entry) = begin
        if isa(entry, AbstractVector) && !isempty(entry) && (entry[1] isa Bool)
            mask = BitVector(Vector{Bool}(entry))
            length(mask) == P.n || error("U mask must have length n=$(P.n)")
            Uc = FiniteFringe.upset_closure(P, mask)
            Uc.mask == mask || error("U entry is not an upset mask; use generators (index list) instead.")
            return Uc
        else
            gens = (entry isa Integer) ? Int[ Int(entry) ] : Vector{Int}(entry)
            return FiniteFringe.upset_from_generators(P, gens)
        end
    end

    parse_downset(entry) = begin
        if isa(entry, AbstractVector) && !isempty(entry) && (entry[1] isa Bool)
            mask = BitVector(Vector{Bool}(entry))
            length(mask) == P.n || error("D mask must have length n=$(P.n)")
            Dc = FiniteFringe.downset_closure(P, mask)
            Dc.mask == mask || error("D entry is not a downset mask; use generators (index list) instead.")
            return Dc
        else
            gens = (entry isa Integer) ? Int[ Int(entry) ] : Vector{Int}(entry)
            return FiniteFringe.downset_from_generators(P, gens)
        end
    end

    U = [parse_upset(x) for x in obj["U"]]
    D = [parse_downset(x) for x in obj["D"]]

    # Parse QQ entries: prefer exact rationals "a/b", but allow integer JSON numbers too.
    parse_QQ(x) = begin
        if x isa Integer
            return QQ(BigInt(x))
        end
        s = String(x)
        if occursin("/", s)
            return string_to_rational(s)
        else
            # allow "2" as an integer string
            return QQ(parse(BigInt, strip(s)))
        end
    end

    m = length(D)
    k = length(U)
    Phi = zeros(QQ, m, k)

    phi_any = obj["phi"]
    @assert length(phi_any) == m "phi must have m=$(m) rows"
    for i in 1:m
        row = phi_any[i]
        @assert length(row) == k "phi row length mismatch (expected k=$(k))"
        for j in 1:k
            Phi[i, j] = parse_QQ(row[j])
        end
    end

    return FiniteFringe.FringeModule{QQ}(P, U, D, Phi)
end


end # module
