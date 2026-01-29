module M2SingularBridge
# ------------------------------------------------------------------------------
# Bridge: obtain flat/injective (flange) data from an external CAS (Macaulay2,
# Singular, ...) using a small JSON schema, then build FlangeZn.Flange{QQ}.
#
# We keep this module minimal and ASCII-only. It *consumes* the JSON produced by
# the CAS; unit tests can target this module without requiring the CAS runtime.
# ------------------------------------------------------------------------------

using JSON3
import ..FlangeZn: Face, IndFlat, IndInj, Flange, canonical_matrix
using ..CoreModules: QQ, string_to_rational

export parse_flange_json, flange_from_m2

"""
JSON schema expected from the CAS:

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
* Scalars in `phi` are interpreted in QQ (Rational{BigInt}).
"""
function parse_flange_json(json_src)::Flange{QQ}
    obj = JSON3.read(json_src)
    n = Int(obj["n"])

    # helper: Face from bools or from list of indices
    function _mkface(n::Int, tau_any)
        if tau_any isa AbstractVector{Bool}
            Face(n, BitVector(tau_any))
        else
            bits = falses(n)
            for t in tau_any; bits[Int(t)] = true; end
            Face(n, bits)
        end
    end

    flats = IndFlat{QQ}[]
    for f in obj["flats"]
        b   = Vector{Int}(f["b"])
        tau = _mkface(n, f["tau"])
        id  = Symbol(String(get(f, "id", "F")))
        push!(flats, IndFlat{QQ}(b, tau, id))
    end

    injectives = IndInj{QQ}[]
    for e in obj["injectives"]
        b   = Vector{Int}(e["b"])
        tau = _mkface(n, e["tau"])
        id  = Symbol(String(get(e, "id", "E")))
        push!(injectives, IndInj{QQ}(b, tau, id))
    end

    Phi = if haskey(obj, "phi")
        A = obj["phi"]
        m = length(injectives); ncol = length(flats)
        M = zeros(QQ, m, ncol)
        @assert length(A) == m "phi: wrong number of rows"
        for i in 1:m
            row = A[i]; @assert length(row) == ncol "phi: wrong number of cols"
            for j in 1:ncol
                val = row[j]
                if val isa String
                    M[i,j] = string_to_rational(val)
                elseif val isa Integer
                    M[i,j] = QQ(val)
                else
                    # last resort: try rationalize but warn the user
                    @warn "phi entry is a non-integer numeric; consider emitting strings \"num/den\" for exactness"
                    M[i,j] = QQ(val)  # may yield large denominators; acceptable as fallback
                end
            end
        end
        M
    else
        canonical_matrix(flats, injectives)
    end

    Flange{QQ}(n, flats, injectives, Phi)
end

"""
    flange_from_m2(cmd::Cmd; jsonpath=nothing) -> Flange{QQ}

Run a CAS command that prints (or writes) the JSON described in the docstring,
then parse it to a `Flange{QQ}`.
"""
function flange_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing)
    if jsonpath === nothing
        io = read(cmd, String)            # CAS prints JSON to stdout
        return parse_flange_json(io)
    else
        run(cmd)                          # CAS writes the JSON file
        open(jsonpath, "r") do io
            return parse_flange_json(io)
        end
    end
end

end # module
