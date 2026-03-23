using Test
using Random
using SparseArrays
using LinearAlgebra

const DO_LONG = lowercase(get(ENV, "POSETMODULES_LONG_TESTS", "true")) in ("1", "true", "yes")

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

# Helper utilities shared across random stress testsets.
function random_chain_subposet(n::Int; p::Float64=0.35)
    leq = falses(n, n)
    for i in 1:n
        leq[i, i] = true
        for j in (i+1):n
            leq[i, j] = rand() < p
        end
    end
    # transitive closure
    for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq)
end

function random_upset(P::FF.FinitePoset)
    gens = [i for i in 1:P.n if rand() < 0.35]
    isempty(gens) && (gens = [rand(1:P.n)])
    return FF.upset_from_generators(P, gens)
end

function random_downset(P::FF.FinitePoset)
    gens = [i for i in 1:P.n if rand() < 0.35]
    isempty(gens) && (gens = [rand(1:P.n)])
    return FF.downset_from_generators(P, gens)
end

function random_fringe_module(P::FF.FinitePoset; mbound::Int=3, rbound::Int=3, density::Float64=0.6)
    m = rand(1:mbound)  # births
    r = rand(1:rbound)  # deaths
    U = [random_upset(P) for _ in 1:m]
    D = [random_downset(P) for _ in 1:r]

    Phi = spzeros(K, r, m)
    for j in 1:r, i in 1:m
        if FF.intersects(U[i], D[j]) && rand() < density
            Phi[j, i] = c(rand(-2:2))
        end
    end
    return FF.FringeModule{K}(P, U, D, Phi; field=field)
end

@testset "Random stress" begin
    if !DO_LONG
        @info "Skipping random stress tests. Set POSETMODULES_LONG_TESTS=true to enable."
        @test true
        return
    end

    Random.seed!(12345)

    for trial in 1:10
        P = random_chain_subposet(rand(4:7); p=0.35)
        M = random_fringe_module(P; mbound=4, rbound=4, density=0.6)

        m = length(M.U)
        r = length(M.D)

        # Invariant: fiber_dimension matches exact rank of the active submatrix at each vertex.
        for q in 1:P.n
            cols = [i for i in 1:m if M.U[i].mask[q]]
            rows = [j for j in 1:r if M.D[j].mask[q]]
            if isempty(cols) || isempty(rows)
                @test FF.fiber_dimension(M, q) == 0
            else
                Aq = Matrix(M.phi[rows, cols])
            @test FF.fiber_dimension(M, q) == TamerOp.FieldLinAlg.rank(field, Aq)
            end
        end

        # Encoding transport invariant: push forward then pull back recovers
        # the original generators exactly.
        enc = EN.build_uptight_encoding_from_fringe(M)
        pi = enc.pi
        Mpb = EN.pullback_fringe_along_encoding(EN.pushforward_fringe_along_encoding(M, pi), pi)
        @test [U.mask for U in Mpb.U] == [U.mask for U in M.U]
        @test [D.mask for D in Mpb.D] == [D.mask for D in M.D]
    end

    # A smaller number of random Ext-vs-Hom checks (can be expensive).
    if field isa CM.RealField
        @test true
    else
        for trial in 1:3
            P = random_chain_subposet(rand(4:6); p=0.4)
            M = random_fringe_module(P; mbound=3, rbound=3, density=0.5)
            N = random_fringe_module(P; mbound=3, rbound=3, density=0.5)

            extMN = DF.ext_dimensions_via_indicator_resolutions(M, N; maxlen=3)
            @test get(extMN, 0, 0) == FF.hom_dimension(M, N)
        end
    end
end

@testset "Random stress: Ext/Tor cross-checks (small degrees)" begin
    if !DO_LONG
        @test true
        return
    end

    if field isa CM.RealField
        @test true
        return
    end

    Random.seed!(12345)

    # Helper: random general poset on [1..n] by choosing random i<j relations
    # and closing transitively.
    function random_poset(n::Int; p_edge::Float64=0.25)
        leq = falses(n,n)
        for i in 1:n
            leq[i,i] = true
        end
        for i in 1:n-1, j in i+1:n
            if rand() < p_edge
                leq[i,j] = true
            end
        end
        # transitive closure
        for k in 1:n, i in 1:n, j in 1:n
            if leq[i,k] && leq[k,j]
                leq[i,j] = true
            end
        end
        return FF.FinitePoset(leq)
    end

    for trial in 1:8
        P = random_poset(rand(4:7); p_edge=0.25)
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

        Hm = random_fringe_module(P; mbound=4, rbound=3, density=0.4)
        Hn = random_fringe_module(P; mbound=4, rbound=3, density=0.4)
        M = IR.pmodule_from_fringe(Hm)
        N = IR.pmodule_from_fringe(Hn)

        # Ext dims: indicator-resolution method vs DerivedFunctors.Ext
        ext_dims = DF.ext_dimensions_via_indicator_resolutions(Hm, Hn; maxlen=3)
        E = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=2))
        for t in 0:2
            @test TO.dim(E,t) == get(ext_dims, t, 0)
        end

        # Ext dims: injective model agrees with projective model
        resN = DF.injective_resolution(N, TO.ResolutionOptions(maxlen=3))
        Einj = TO.ExtInjective(M, resN)
        for t in 0:2
            @test TO.dim(Einj,t) == TO.dim(E,t)
        end

        # Identity naturality on Ext
        idM = IR.id_morphism(M)
        idN = IR.id_morphism(N)
        for t in 0:2
            d = TO.dim(E,t)
            A = TO.ext_map_first(E, E, idM; t=t)
            B = TO.ext_map_second(E, E, idN; t=t)
            @test Matrix(A) == Matrix{K}(I,d,d)
            @test Matrix(B) == Matrix{K}(I,d,d)
        end

        # Tor sanity: boundary squares to zero, and identity naturality
        Hr = random_fringe_module(Pop; mbound=4, rbound=3, density=0.4)
        Hl = random_fringe_module(P;   mbound=4, rbound=3, density=0.4)
        Rop = IR.pmodule_from_fringe(Hr)
        L   = IR.pmodule_from_fringe(Hl)

        T = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=2))

        if length(T.bd) >= 2
            @test nnz(T.bd[1] * T.bd[2]) == 0
        end

        idR = IR.id_morphism(Rop)
        idL = IR.id_morphism(L)
        for s in 0:2
            d = TO.dim(T,s)
            A = TO.tor_map_first(T, T, idR; s=s)
            B = TO.tor_map_second(T, T, idL; s=s)
            @test Matrix(A) == Matrix{K}(I,d,d)
            @test Matrix(B) == Matrix{K}(I,d,d)
        end
    end
end

end # with_fields
