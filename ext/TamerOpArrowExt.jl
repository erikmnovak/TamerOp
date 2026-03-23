module TamerOpArrowExt

using Arrow
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

function FEA.save_features_arrow(path::AbstractString,
                                fs::FEA.FeatureSet;
                                format::Symbol=:wide,
                                include_metadata::Bool=true,
                                metadata_path::Union{Nothing,AbstractString}=nothing,
                                compress=nothing)
    tbl = FEA.feature_table(fs; format=format)
    if compress === nothing
        Arrow.write(path, tbl)
    else
        Arrow.write(path, tbl; compress=compress)
    end

    if include_metadata
        mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
        FEA.save_metadata_json(mpath, FEA.feature_metadata(fs; format=format))
    end
    return path
end

function FEA.load_features_arrow(path::AbstractString;
                                format::Symbol=:wide,
                                ids_col::Symbol=:id,
                                metadata_path::Union{Nothing,AbstractString}=nothing,
                                require_metadata::Bool=false)
    table = Arrow.Table(path)
    cols = Tables.columntable(table)
    mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
    md = if isfile(mpath)
        FEA.load_metadata_json(mpath; validate_feature_schema=true)
    elseif require_metadata
        throw(ArgumentError("load_features_arrow: metadata sidecar not found at $(mpath)"))
    else
        nothing
    end
    meta = md === nothing ? NamedTuple() : (metadata=md,)
    return FEA._featureset_from_columntable(cols; format=format, ids_col=ids_col, meta=meta)
end

end # module
