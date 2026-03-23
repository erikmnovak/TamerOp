module TamerOpParquet2Ext

using Parquet2
using Tables

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

function _parquet_col(v::AbstractVector)
    T = eltype(v)
    if T <: Symbol
        return String[string(x) for x in v]
    elseif T <: Union{Missing,Symbol}
        U = Base.nonmissingtype(T)
        U <: Symbol || return v
        out = Vector{Union{Missing,String}}(undef, length(v))
        @inbounds for i in eachindex(v)
            vi = v[i]
            out[i] = ismissing(vi) ? missing : String(vi)
        end
        return out
    end
    return v
end

function _parquet_compatible_table(tbl)
    cols = Tables.columntable(tbl)
    nms = propertynames(cols)
    vecs = Vector{Any}(undef, length(nms))
    @inbounds for i in eachindex(nms)
        vecs[i] = _parquet_col(collect(getproperty(cols, nms[i])))
    end
    return NamedTuple{Tuple(nms)}(Tuple(vecs))
end

function FEA.save_features_parquet(path::AbstractString,
                                  fs::FEA.FeatureSet;
                                  format::Symbol=:wide,
                                  include_metadata::Bool=true,
                                  metadata_path::Union{Nothing,AbstractString}=nothing,
                                  kwargs...)
    tbl = FEA.feature_table(fs; format=format)
    Parquet2.writefile(path, _parquet_compatible_table(tbl); kwargs...)

    if include_metadata
        mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
        FEA.save_metadata_json(mpath, FEA.feature_metadata(fs; format=format))
    end
    return path
end

function FEA.load_features_parquet(path::AbstractString;
                                  format::Symbol=:wide,
                                  ids_col::Symbol=:id,
                                  metadata_path::Union{Nothing,AbstractString}=nothing,
                                  require_metadata::Bool=false,
                                  kwargs...)
    ds = Parquet2.Dataset(path; kwargs...)
    cols = Tables.columntable(ds)
    mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
    md = if isfile(mpath)
        FEA.load_metadata_json(mpath; validate_feature_schema=true)
    elseif require_metadata
        throw(ArgumentError("load_features_parquet: metadata sidecar not found at $(mpath)"))
    else
        nothing
    end
    meta = md === nothing ? NamedTuple() : (metadata=md,)
    return FEA._featureset_from_columntable(cols; format=format, ids_col=ids_col, meta=meta)
end

end # module
