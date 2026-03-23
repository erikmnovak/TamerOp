module TamerOpCSVExt

using CSV

const TO = let pm = nothing
    if isdefined(Main, :TamerOp)
        pm = getfield(Main, :TamerOp)
    else
        @eval import TamerOp
        pm = TamerOp
    end
    pm
end

const FEA = TO.Featurizers

@inline function _wide_table_from_featureset(fs::FEA.FeatureSet;
                                             ids_col::Symbol=:id,
                                             include_ids::Bool=true)
    nms = Symbol[]
    vecs = Vector{Any}()
    if include_ids
        push!(nms, ids_col)
        push!(vecs, copy(fs.ids))
    end
    @inbounds for j in eachindex(fs.names)
        push!(nms, fs.names[j])
        push!(vecs, collect(view(fs.X, :, j)))
    end
    return NamedTuple{Tuple(nms)}(Tuple(vecs))
end

@inline function _long_table_from_featureset(fs::FEA.FeatureSet;
                                             include_sample_index::Bool=true)
    ns = FEA.nsamples(fs)
    nf = FEA.nfeatures(fs)
    n = ns * nf
    ids = Vector{String}(undef, n)
    feats = Vector{Symbol}(undef, n)
    vals = Vector{eltype(fs.X)}(undef, n)
    sidx = include_sample_index ? Vector{Int}(undef, n) : Int[]

    t = 0
    @inbounds for i in 1:ns
        idi = fs.ids[i]
        for j in 1:nf
            t += 1
            ids[t] = idi
            feats[t] = fs.names[j]
            vals[t] = fs.X[i, j]
            include_sample_index && (sidx[t] = i)
        end
    end

    if include_sample_index
        return (id=ids, feature=feats, value=vals, sample_index=sidx)
    end
    return (id=ids, feature=feats, value=vals)
end

@inline function _csv_columntable(path::AbstractString; kwargs...)
    file = CSV.File(path; kwargs...)
    nms = collect(propertynames(file))
    vecs = Vector{Any}(undef, length(nms))
    @inbounds for i in eachindex(nms)
        vecs[i] = collect(getproperty(file, nms[i]))
    end
    return NamedTuple{Tuple(nms)}(Tuple(vecs))
end

@inline function _add_ids_column(cols, ids_col::Symbol)
    nms0 = collect(propertynames(cols))
    nrows = isempty(nms0) ? 0 : length(getproperty(cols, nms0[1]))
    ids = String[string(i) for i in 1:nrows]
    nms = Symbol[ids_col]
    append!(nms, nms0)
    vecs = Vector{Any}(undef, length(nms))
    vecs[1] = ids
    @inbounds for i in eachindex(nms0)
        vecs[i + 1] = collect(getproperty(cols, nms0[i]))
    end
    return NamedTuple{Tuple(nms)}(Tuple(vecs))
end

@inline function _reorder_featureset(fs::FEA.FeatureSet, ordered_names::Vector{Symbol})
    fs.names == ordered_names && return fs
    pos = Dict{Symbol,Int}(nm => i for (i, nm) in enumerate(fs.names))
    perm = Vector{Int}(undef, length(ordered_names))
    @inbounds for i in eachindex(ordered_names)
        nm = ordered_names[i]
        haskey(pos, nm) || throw(ArgumentError("load_features_csv: metadata feature $(nm) missing from CSV columns"))
        perm[i] = pos[nm]
    end
    X = fs.X[:, perm]
    return FEA.FeatureSet(X, ordered_names, fs.ids, fs.meta)
end

function FEA.save_features_csv(path::AbstractString,
                              fs::FEA.FeatureSet;
                              format::Symbol=:wide,
                              layout::Symbol=:samples_by_features,
                              include_metadata::Bool=true,
                              metadata_path::Union{Nothing,AbstractString}=nothing,
                              ids_col::Symbol=:id,
                              include_ids::Bool=true,
                              include_sample_index::Bool=true,
                              kwargs...)
    layout in (:samples_by_features, :features_by_samples) ||
        throw(ArgumentError("save_features_csv: layout must be :samples_by_features or :features_by_samples"))
    mode = format
    mode in (:wide, :long) || throw(ArgumentError("save_features_csv: format must be :wide or :long"))

    fs_out = if layout == :samples_by_features
        fs
    else
        XF = permutedims(fs.X)
        feature_ids = String[string(nm) for nm in fs.names]
        sample_names = Symbol["sample_$(i)" for i in eachindex(fs.ids)]
        FEA.FeatureSet(XF, sample_names, feature_ids, fs.meta)
    end

    if mode == :wide
        tbl = _wide_table_from_featureset(fs_out; ids_col=ids_col, include_ids=include_ids)
        CSV.write(path, tbl; kwargs...)
    else
        tbl = _long_table_from_featureset(fs_out; include_sample_index=include_sample_index)
        CSV.write(path, tbl; kwargs...)
    end

    if include_metadata
        md = FEA.feature_metadata(fs; format=mode)
        md["layout"] = String(layout)
        md["csv_ids_col"] = String(ids_col)
        md["csv_include_ids"] = include_ids
        md["csv_include_sample_index"] = include_sample_index
        mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
        FEA.save_metadata_json(mpath, md)
    end
    return path
end

function FEA.load_features_csv(path::AbstractString;
                              format::Symbol=:wide,
                              layout::Union{Nothing,Symbol}=nothing,
                              ids_col::Symbol=:id,
                              metadata_path::Union{Nothing,AbstractString}=nothing,
                              require_metadata::Bool=false,
                              validate_feature_schema::Bool=true,
                              kwargs...)
    mode = format
    mode in (:wide, :long) || throw(ArgumentError("load_features_csv: format must be :wide or :long"))
    cols = _csv_columntable(path; kwargs...)

    mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
    md = if isfile(mpath)
        FEA.load_metadata_json(mpath; validate_feature_schema=validate_feature_schema)
    elseif require_metadata
        throw(ArgumentError("load_features_csv: metadata sidecar not found at $(mpath)"))
    else
        nothing
    end

    layout0 = if layout === nothing
        md === nothing ? :samples_by_features : Symbol(String(FEA._obj_get(md, "layout", "samples_by_features")))
    else
        layout
    end
    layout0 in (:samples_by_features, :features_by_samples) ||
        throw(ArgumentError("load_features_csv: layout must be :samples_by_features or :features_by_samples"))

    cols0 = if mode == :wide && !(ids_col in propertynames(cols))
        _add_ids_column(cols, ids_col)
    else
        cols
    end
    meta = md === nothing ? NamedTuple() : (metadata=md,)
    fs0 = FEA._featureset_from_columntable(cols0; format=mode, ids_col=ids_col, meta=meta)

    fs1 = if md !== nothing && FEA._obj_haskey(md, "feature_names")
        ord = Symbol[Symbol(String(s)) for s in md["feature_names"]]
        _reorder_featureset(fs0, ord)
    else
        fs0
    end

    if layout0 == :features_by_samples
        X = permutedims(fs1.X)
        feature_ids = String[string(nm) for nm in fs1.names]
        names2 = Symbol["sample_$(i)" for i in eachindex(fs1.ids)]
        return FEA.FeatureSet(X, names2, feature_ids, fs1.meta)
    end
    return fs1
end

end # module
