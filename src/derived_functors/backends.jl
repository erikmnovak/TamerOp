# backends.jl -- Zn/Rn workflow wrappers and backend glue

"""
Backends: Zn/Rn wrappers and boxing/glue utilities.

This submodule should host:
- ExtZn, ExtRn style wrappers
- pmodule_on_box and similar utilities
- any encoding-specific drivers needed to connect geometric encodings to
  derived-functor computations

If PL backends are optional in some environments, keep those imports guarded
when you actually start migrating code here.
"""
module Backends
    using LinearAlgebra
    using SparseArrays

    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache, field_from_eltype
    using ...Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions
    using ...Modules: PModule, PMorphism
    using ...ChainComplexes
    using ...IndicatorResolutions: pmodule_from_fringe
    import ...PLPolyhedra
    using ...PLPolyhedra: PLFringe

    using ...ZnEncoding
    using ...FlangeZn: Flange

    import ..Utils: compose
    import ..Resolutions: projective_resolution, injective_resolution
    import ..SpectralSequences: ExtDoubleComplex, ExtSpectralSequence
    import ..ExtTorSpaces: Ext, ExtInjective, Tor, TorSpace, dim, basis

    # ----------------------------
    # Zn and Rn workflow wrappers: encode -> compute on a finite poset
    # ----------------------------

    """
        pmodule_on_box(FG::Flange{K}; a::NTuple{N,Int}, b::NTuple{N,Int}) -> PModule{K}

    Restrict the Z^n module presented by a flange `FG` to the finite grid poset
    on the integer box [a,b] (inclusive) and return it as a finite-poset module.
    """
    function pmodule_on_box(FG::Flange{K}; a::NTuple{N,Int}, b::NTuple{N,Int}) where {K,N}
        return ZnEncoding.pmodule_on_box(FG; a=a, b=b)
    end

    """
        ExtZn(FG1, FG2, enc::EncodingOptions, df::DerivedFunctorOptions; method=:regions, a=nothing, b=nothing)

    Compute Ext^s(FG1, FG2) for Z^n modules given by flange presentations.

    This is a workflow wrapper:
    1. Encode the infinite module(s) onto a finite encoding poset P (controlled by `enc`),
    2. Run homological algebra on the resulting finite-poset modules (controlled by `df`).

    Keyword `method`:
    - `:regions` (default): encode FG1 and FG2 to a common finite encoding poset.
    - `:box`: restrict both to the finite integer box [a,b] and compute on that box.
    When `method=:box` you must provide integer tuples `a` and `b`.
    """
    function ExtZn(FG1::Flange{K}, FG2::Flange{K},
                enc::EncodingOptions, df::DerivedFunctorOptions;
                method::Symbol = :regions,
                a::Union{Nothing,Tuple{Vararg{Int}}} = nothing,
                b::Union{Nothing,Tuple{Vararg{Int}}} = nothing) where {K}

        if method == :box
            a === nothing && error("ExtZn(method=:box): missing keyword a")
            b === nothing && error("ExtZn(method=:box): missing keyword b")
            M = pmodule_on_box(FG1; a=a, b=b)
            N = pmodule_on_box(FG2; a=a, b=b)
            return Ext(M, N, df)
        elseif method == :regions
            P, Hs, pi = ZnEncoding.encode_from_flanges(FG1, FG2, enc)
            M = pmodule_from_fringe(Hs[1])
            N = pmodule_from_fringe(Hs[2])
            return Ext(M, N, df)
        else
            error("ExtZn: unknown method=$(method); expected :regions or :box")
        end
    end

    """
        projective_resolution_Zn(FG::Flange{K}, enc::EncodingOptions, res::ResolutionOptions; return_encoding=false)

    Compute a projective resolution of the Z^n module presented by FG by:
        FG -> (P, M, pi) via region encoding, then projective_resolution(M) on P.

    If return_encoding=true, return a named tuple:
        (res=res, P=P, pi=pi)

    This is a computation on the encoding poset P. Minimality and Betti-style data
    are minimality/Betti on P, not a Z^n-native commutative-algebra theory.
    """
    function projective_resolution_Zn(FG::Flange{K},
                                    enc::EncodingOptions,
                                    res::ResolutionOptions;
                                    return_encoding::Bool=false,
                                    threads::Bool = (Threads.nthreads() > 1)) where {K}
        P, H, pi = ZnEncoding.encode_from_flange(FG, enc)
        M = pmodule_from_fringe(H)
        R = projective_resolution(M, res; threads=threads)
        return return_encoding ? (res=R, P=P, pi=pi) : R
    end

    """
        injective_resolution_Zn(FG::Flange{K}, enc::EncodingOptions, res::ResolutionOptions; return_encoding=false)

    Compute an injective resolution of the Z^n module presented by FG by:
        FG -> (P, M, pi) via region encoding, then injective_resolution(M) on P.

        If return_encoding=true, return a named tuple:
        (res=res, P=P, pi=pi)

        This is a computation on the encoding poset P.
    """
    function injective_resolution_Zn(FG::Flange{K},
                                    enc::EncodingOptions,
                                    res::ResolutionOptions;
                                    return_encoding::Bool=false,
                                    threads::Bool = (Threads.nthreads() > 1)) where {K}
        P, H, pi = ZnEncoding.encode_from_flange(FG, enc)
        M = pmodule_from_fringe(H)
        R = injective_resolution(M, res; threads=threads)
        return return_encoding ? (res=R, P=P, pi=pi) : R
    end

    # ----------------------------
    # PL workflow wrappers (R^n)
    # ----------------------------

    """
        ExtRn(F1, F2, enc::EncodingOptions, df::DerivedFunctorOptions)

    Compute Ext^s(F1, F2) for modules over R^n given by PL fringe presentations.

    This follows the Ezra Miller pattern: do homological algebra on a finite encoding
    poset, not on the infinite poset R^n directly.

    Algorithm:
    1. Build a single common encoding poset P that simultaneously encodes the union of
    all birth/death shapes appearing in F1 and F2.
    2. Push both presentations down to fringe modules on P.
    3. Convert to P-modules and compute Ext on the finite poset.

    Options:
    - Encoding parameters must be passed via `enc` (an `EncodingOptions` object). In particular:
    * `enc.max_regions` caps the number of polyhedral regions/signatures enumerated by the backend.
    * `enc.strict_eps` is the rational slack used when forcing "outside" constraints.
    - Derived-functor parameters must be passed via `df` (a `DerivedFunctorOptions` object). In particular:
    * `df.maxdeg` controls the range 0 <= s <= maxdeg computed.
    """

    function ExtRn(F1::PLFringe, F2::PLFringe,
                enc::EncodingOptions, df::DerivedFunctorOptions)
        P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(F1, F2, enc)
        M = pmodule_from_fringe(Hs[1])
        N = pmodule_from_fringe(Hs[2])
        return Ext(M, N, df)
    end

    """
        projective_resolution_Rn(F, enc, res; return_encoding=false)

    Compute a projective resolution of the module over R^n presented by F by:
        F -> (P, M, pi) via encoding, then projective_resolution(M) on the finite poset P.

    If return_encoding=true, return a named tuple:
        (res=res, P=P, pi=pi)

    This is a computation on the encoding poset P. Minimality and Betti-style data
    are minimality/Betti on P, not an R^n-native commutative-algebra theory.
    """
    function projective_resolution_Rn(F::PLFringe,
                                    enc::EncodingOptions,
                                    res::ResolutionOptions;
                                    return_encoding::Bool=false,
                                    threads::Bool = (Threads.nthreads() > 1))
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, enc)
        M = pmodule_from_fringe(H)
        R = projective_resolution(M, res; threads=threads)
        return return_encoding ? (res=R, P=P, pi=pi) : R
    end

    """
        injective_resolution_Rn(F, enc, res; return_encoding=false)

    Compute an injective resolution of the module over R^n presented by F by:
        F -> (P, M, pi) via encoding, then injective_resolution(M) on the finite poset P.

    If return_encoding=true, return a named tuple:
        (res=res, P=P, pi=pi)

    This is a computation on the encoding poset P.
    """
    function injective_resolution_Rn(F::PLFringe,
                                    enc::EncodingOptions,
                                    res::ResolutionOptions;
                                    return_encoding::Bool=false,
                                    threads::Bool = (Threads.nthreads() > 1))
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, enc)
        M = pmodule_from_fringe(H)
        R = injective_resolution(M, res; threads=threads)
        return return_encoding ? (res=R, P=P, pi=pi) : R
    end

    # -------------------------------------------------------------------------
    # Ext bicomplex / spectral sequence wrappers for presentations.
    #
    # IMPORTANT DESIGN RULE:
    #   All encoding policy must be supplied via an EncodingOptions object.
    #   In particular, these wrappers do NOT accept ad hoc keywords like
    #   max_regions or strict_eps, and they do not construct EncodingOptions
    #   internally. This keeps the "single source of truth" for defaults in
    #   Options.EncodingOptions and enables options/threading/provenance
    #   at the workflow layer.
    # -------------------------------------------------------------------------

    # Internal sanity check: these wrappers call a specific encoding backend
    # (ZnEncoding for flanges and PLBackend for PLFringe). We therefore enforce
    # that the provenance tag `enc.backend` is compatible with that choice.
    function _require_encoding_backend(enc::EncodingOptions, wanted::Symbol, caller::AbstractString)
        if !(enc.backend == :auto || enc.backend == wanted)
            error(string(
                caller, ": expected enc.backend in (:auto, :", wanted, "), got ", enc.backend, ". ",
                "Pass EncodingOptions(backend=:", wanted, ", ...) (or backend=:auto)."
            ))
        end
        return nothing
    end

    """
        ExtDoubleComplex(FG1::Flange{K}, FG2::Flange{K};
                         method=:regions, a=nothing, b=nothing, maxlen=nothing)

    Build the Ext bicomplex for two Z^n modules given by flange presentations.

    Keyword `method` selects how the finite poset model is produced:

    - `method = :regions`: common-encode FG1 and FG2 by enumerating regions.
      This *requires* an explicit `EncodingOptions` argument; see the 3-argument
      method below.
    - `method = :box`: ignore region structure and restrict both modules to the
      integer box [a,b] (inclusive). When `method=:box` you must provide integer
      tuples `a` and `b`.

    This 2-argument method exists to give a helpful error message when users
    forget to pass an EncodingOptions for region encoding. It never constructs
    EncodingOptions internally.
    """
    function ExtDoubleComplex(FG1::Flange{K}, FG2::Flange{K};
                              method::Symbol = :regions,
                              a = nothing,
                              b = nothing,
                              maxlen::Union{Nothing,Int} = nothing,
                              cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        if method == :box
            if a === nothing || b === nothing
                error("ExtDoubleComplex(method=:box): missing keywords a and b (box corners).")
            end
            M1 = pmodule_on_box(FG1; a=a, b=b)
            M2 = pmodule_on_box(FG2; a=a, b=b)
            return ExtDoubleComplex(M1, M2; maxlen=maxlen, cache=cache)
        elseif method == :regions
            error("ExtDoubleComplex(method=:regions): pass EncodingOptions explicitly: ExtDoubleComplex(FG1, FG2, enc; method=:regions, ...).")
        else
            error("ExtDoubleComplex: unknown method=$(method); expected :box or :regions")
        end
    end

    """
        ExtDoubleComplex(FG1::Flange{K}, FG2::Flange{K},
                         enc::EncodingOptions;
                         method=:regions, a=nothing, b=nothing, maxlen=nothing)

    Z^n workflow wrapper: common-encode two flange presentations to a finite
    encoding poset (controlled by `enc`), then build the Ext bicomplex on that
    finite poset.

    The encoding policy must be supplied exclusively via `enc`. This wrapper
    intentionally does *not* accept ad hoc encoding keywords (such as
    `max_regions`) and it does not construct EncodingOptions internally.

    Keyword `method`:
    - `:regions` (default): common-encode FG1 and FG2 via region encoding.
    - `:box`: ignore `enc`, restrict to the integer box [a,b], and compute on it.
      When `method=:box` you must provide integer tuples `a` and `b`.
    """
    function ExtDoubleComplex(FG1::Flange{K}, FG2::Flange{K}, enc::EncodingOptions;
                              method::Symbol = :regions,
                              a = nothing,
                              b = nothing,
                              maxlen::Union{Nothing,Int} = nothing,
                              cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        if method == :box
            # Delegate to the 2-argument method for consistent error messages.
            return ExtDoubleComplex(FG1, FG2; method=:box, a=a, b=b, maxlen=maxlen, cache=cache)
        elseif method == :regions
            _require_encoding_backend(enc, :zn, "ExtDoubleComplex(Zn)")
            _, Hs, _ = ZnEncoding.encode_from_flanges(FG1, FG2, enc)
            M1 = pmodule_from_fringe(Hs[1])
            M2 = pmodule_from_fringe(Hs[2])
            return ExtDoubleComplex(M1, M2; maxlen=maxlen, cache=cache)
        else
            error("ExtDoubleComplex: unknown method=$(method); expected :box or :regions")
        end
    end

    """
        ExtSpectralSequence(FG1::Flange{K}, FG2::Flange{K};
                            first=:vertical,
                            method=:regions, a=nothing, b=nothing, maxlen=nothing)

    Convenience wrapper: build the Ext bicomplex for two Z^n flange
    presentations and return its spectral sequence.

    - For `method=:box`, this is a thin wrapper around the 2-argument
      `ExtDoubleComplex(FG1,FG2; ...)`.
    - For `method=:regions`, you must pass an EncodingOptions; see the
      3-argument method below.

    Keyword `first` is passed to `ChainComplexes.spectral_sequence`.
    """
    function ExtSpectralSequence(FG1::Flange{K}, FG2::Flange{K};
                                 first::Symbol = :vertical,
                                 method::Symbol = :regions,
                                 a = nothing,
                                 b = nothing,
                                 maxlen::Union{Nothing,Int} = nothing,
                                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        if method == :regions
            error("ExtSpectralSequence(method=:regions): pass EncodingOptions explicitly: ExtSpectralSequence(FG1, FG2, enc; method=:regions, ...).")
        end
        DC = ExtDoubleComplex(FG1, FG2; method=method, a=a, b=b, maxlen=maxlen, cache=cache)
        return ChainComplexes.spectral_sequence(DC; output=:full, first=first)
    end

    """
        ExtSpectralSequence(FG1::Flange{K}, FG2::Flange{K},
                            enc::EncodingOptions;
                            first=:vertical,
                            method=:regions, a=nothing, b=nothing, maxlen=nothing)

    Z^n workflow wrapper: common-encode FG1 and FG2 using `enc` (when
    `method=:regions`), build the Ext bicomplex, and return its spectral
    sequence.

    Encoding parameters are supplied exclusively via the EncodingOptions `enc`.
    """
    function ExtSpectralSequence(FG1::Flange{K}, FG2::Flange{K}, enc::EncodingOptions;
                                 first::Symbol = :vertical,
                                 method::Symbol = :regions,
                                 a = nothing,
                                 b = nothing,
                                 maxlen::Union{Nothing,Int} = nothing,
                                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        DC = ExtDoubleComplex(FG1, FG2, enc; method=method, a=a, b=b, maxlen=maxlen, cache=cache)
        return ChainComplexes.spectral_sequence(DC; output=:full, first=first)
    end

    """
        ExtDoubleComplex(PL1::PLFringe, PL2::PLFringe,
                         enc::EncodingOptions; maxlen=nothing)

    R^n workflow wrapper: common-encode two PL fringe presentations to a finite
    encoding poset (controlled by `enc`), then build the Ext bicomplex on that
    finite poset.

    Encoding parameters are supplied exclusively via `enc` (for example
    `enc.max_regions` and `enc.strict_eps`). This wrapper does not accept ad hoc
    encoding keywords and it does not construct EncodingOptions internally.
    """
    function ExtDoubleComplex(PL1::PLFringe, PL2::PLFringe, enc::EncodingOptions;
                              maxlen::Union{Nothing,Int} = nothing,
                              cache::Union{Nothing,ResolutionCache}=nothing)
        _require_encoding_backend(enc, :pl, "ExtDoubleComplex(Rn)")
        _, Hs, _ = PLPolyhedra.encode_from_PL_fringes(PL1, PL2, enc)
        M1 = pmodule_from_fringe(Hs[1])
        M2 = pmodule_from_fringe(Hs[2])
        return ExtDoubleComplex(M1, M2; maxlen=maxlen, cache=cache)
    end

    function ExtDoubleComplex(PL1::PLFringe, PL2::PLFringe;
                              maxlen::Union{Nothing,Int} = nothing,
                              cache::Union{Nothing,ResolutionCache}=nothing)
        error("ExtDoubleComplex(PL1, PL2; ...): pass EncodingOptions explicitly: ExtDoubleComplex(PL1, PL2, enc; maxlen=maxlen).")
    end

    """
        ExtSpectralSequence(PL1::PLFringe, PL2::PLFringe,
                            enc::EncodingOptions;
                            first=:vertical, maxlen=nothing)

    R^n workflow wrapper: common-encode PL1 and PL2 using `enc`, build the Ext
    bicomplex, and return its spectral sequence.

    Encoding parameters are supplied exclusively via the EncodingOptions `enc`.
    """
    function ExtSpectralSequence(PL1::PLFringe, PL2::PLFringe, enc::EncodingOptions;
                                 first::Symbol = :vertical,
                                 maxlen::Union{Nothing,Int} = nothing,
                                 cache::Union{Nothing,ResolutionCache}=nothing)
        DC = ExtDoubleComplex(PL1, PL2, enc; maxlen=maxlen, cache=cache)
        return ChainComplexes.spectral_sequence(DC; output=:full, first=first)
    end

    function ExtSpectralSequence(PL1::PLFringe, PL2::PLFringe;
                                 first::Symbol = :vertical,
                                 maxlen::Union{Nothing,Int} = nothing,
                                 cache::Union{Nothing,ResolutionCache}=nothing)
        error("ExtSpectralSequence(PL1, PL2; ...): pass EncodingOptions explicitly: ExtSpectralSequence(PL1, PL2, enc; first=first, maxlen=maxlen).")
    end

end
