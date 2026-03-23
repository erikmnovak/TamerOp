# utils.jl -- low-level utilities for DerivedFunctors

module Utils
    using LinearAlgebra
    using SparseArrays

    # Sibling modules under TamerOp (two levels up from this nested module).
    using ...CoreModules: AbstractCoeffField, RealField, field_from_eltype
    using ...FieldLinAlg
    using ...Modules: PModule, PMorphism, _get_cover_cache

    # ----------------------------
    # Basic utilities: morphism composition (local, explicit, reliable)
    # ----------------------------

    function compose(g::PMorphism{K}, f::PMorphism{K}) where {K}
        @assert f.cod === g.dom
        Q = f.dom.Q
        comps = Vector{Matrix{K}}(undef, Q.n)
        for i in 1:Q.n
            comps[i] = FieldLinAlg._matmul(g.comps[i], f.comps[i])
        end
        return PMorphism{K}(f.dom, g.cod, comps)
    end

    function is_zero_matrix(field::AbstractCoeffField, A::AbstractMatrix)
        if field isa RealField
            isempty(A) && return true
            maxabs = maximum(abs, A)
            tol = field.atol + field.rtol * maxabs
            return maxabs <= tol
        end
        return all(iszero, A)
    end

    # Solve A*X = B (particular solution, free vars set to 0).
    function solve_particular(field::AbstractCoeffField, A::AbstractMatrix, B::AbstractMatrix)
        A0 = Matrix(A)
        B0 = Matrix(B)
        m, n = size(A0)
        @assert size(B0, 1) == m
        Aug = hcat(A0, B0)
        R, pivs_all = FieldLinAlg.rref(field, Aug)
        rhs = size(B0, 2)
        for i in 1:m
            if all(R[i, 1:n] .== 0)
                if any(R[i, n+1:n+rhs] .!= 0)
                    error("solve_particular: inconsistent system")
                end
            end
        end
        pivs = Int[]
        for p in pivs_all
            p <= n && push!(pivs, p)
        end
        X = zeros(eltype(A0), n, rhs)
        for (row, pcol) in enumerate(pivs)
            X[pcol, :] = R[row, n+1:n+rhs]
        end
        return X
    end

end
