module PosetModulesNemoExt

import PosetModules
import Nemo

const QQ = PosetModules.CoreModules.QQ

# Flip the runtime flag (optional, but useful for debugging / telemetry).
PosetModules.ExactQQ._NEMO_ENABLED[] = true

# Conversions between Matrix{QQ} and Nemo.fmpq_mat.
function _to_fmpq_mat(A::AbstractMatrix{QQ})
    m, n = size(A)
    S = Nemo.MatrixSpace(Nemo.QQ, m, n)
    M = S()
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = Nemo.QQ(A[i, j])
    end
    return M
end

function _from_fmpq_mat(M::Nemo.fmpq_mat)
    m, n = size(M)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = QQ(BigInt(Nemo.numerator(x)), BigInt(Nemo.denominator(x)))
    end
    return A
end

# Only convert dense strided matrices by default (avoids slow elementwise
# conversion of sparse matrices when Nemo is loaded).
@inline function PosetModules.ExactQQ._to_backend_matrix(A::StridedMatrix{QQ})
    return _to_fmpq_mat(A)
end

function PosetModules.ExactQQ._rref_backend(A::Nemo.fmpq_mat; pivots::Bool=true)
    _, R = Nemo.rref(A)
    M = _from_fmpq_mat(R)

    pivs = Int[]
    m, n = size(M)
    @inbounds for i in 1:m
        for j in 1:n
            if M[i, j] != 0
                push!(pivs, j)
                break
            end
        end
    end

    return pivots ? (M, Tuple(pivs)) : M
end

PosetModules.ExactQQ._rank_backend(A::Nemo.fmpq_mat) = Nemo.rank(A)

function PosetModules.ExactQQ._nullspace_backend(A::Nemo.fmpq_mat)
    _, N = Nemo.nullspace(A)
    return _from_fmpq_mat(N)
end

end # module
