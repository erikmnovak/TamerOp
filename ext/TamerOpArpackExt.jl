module TamerOpArpackExt

using Arpack
using SparseArrays

const TO = let pm = nothing
    if isdefined(Main, :TamerOp)
        pm = getfield(Main, :TamerOp)
    else
        @eval import TamerOp
        pm = TamerOp
    end
    pm
end

const FL = TO.FieldLinAlg
const CM = TO.CoreModules

function _svds_nullspace_impl(F::CM.RealField, A)
    S = A isa SparseMatrixCSC ? A : sparse(A)
    m, n = size(S)
    n == 0 && return zeros(Float64, 0, 0)

    tol = FL._float_tol(F, S)
    kmax = min(n, max(2, min(64, n)))
    k = min(kmax, max(2, min(16, n)))

    while true
        svdres = try
            Arpack.svds(S; nsv=k, which=:SM, ritzvec=true)
        catch
            return nothing
        end

        idx = findall(s -> abs(s) <= tol, svdres.S)
        if !isempty(idx)
            return Matrix(svdres.Vt[idx, :])'
        end

        if k >= kmax
            return nothing
        end
        k = min(kmax, 2 * k)
    end
end

FL._set_svds_impl!(_svds_nullspace_impl)

end # module
