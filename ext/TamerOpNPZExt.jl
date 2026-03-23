module TamerOpNPZExt

using NPZ
using JSON3

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

@inline function _json_bytes(x)
    return Vector{UInt8}(codeunits(JSON3.write(x; allow_inf=true)))
end

@inline function _decode_json_bytes(v, key::AbstractString)
    if !(v isa AbstractVector{<:Unsigned})
        throw(ArgumentError("load_features_npz: key $(key) must store UTF-8 bytes (Vector{UInt8})"))
    end
    return String(Vector{UInt8}(v))
end

@inline function _ensure_numeric_matrix(X)
    X isa AbstractMatrix || throw(ArgumentError("save_features_npz: X must be a matrix"))
    T = eltype(X)
    if T <: Number
        return Matrix(X)
    elseif T <: Union{Missing,<:Number}
        throw(ArgumentError("save_features_npz: NPZ export does not support missing-valued feature matrices"))
    end
    throw(ArgumentError("save_features_npz: NPZ export requires numeric matrix eltype, got $(T)"))
end

function FEA.save_features_npz(path::AbstractString,
                              fs::FEA.FeatureSet;
                              x_key::AbstractString="X",
                              names_key::AbstractString="names",
                              ids_key::AbstractString="ids",
                              meta_key::AbstractString="meta_json",
                              format::Symbol=:wide,
                              layout::Symbol=:samples_by_features,
                              include_metadata::Bool=true,
                              metadata_path::Union{Nothing,AbstractString}=nothing)
    layout in (:samples_by_features, :features_by_samples) ||
        throw(ArgumentError("save_features_npz: layout must be :samples_by_features or :features_by_samples"))

    X0 = _ensure_numeric_matrix(fs.X)
    X = layout == :samples_by_features ? X0 : permutedims(X0)
    names_json = [String(nm) for nm in fs.names]
    ids_json = [String(id) for id in fs.ids]

    md = include_metadata ? FEA.feature_metadata(fs; format=format) : FEA.feature_schema_header(format=format)
    md["layout"] = String(layout)
    md_json = JSON3.write(md; allow_inf=true)

    payload = Dict{String,Any}(
        String(x_key) => X,
        String(names_key) => _json_bytes(names_json),
        String(ids_key) => _json_bytes(ids_json),
        String(meta_key) => Vector{UInt8}(codeunits(md_json)),
    )
    NPZ.npzwrite(path, payload)

    if include_metadata
        mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
        FEA.save_metadata_json(mpath, md)
    end
    return path
end

function FEA.load_features_npz(path::AbstractString;
                              x_key::AbstractString="X",
                              names_key::AbstractString="names",
                              ids_key::AbstractString="ids",
                              meta_key::AbstractString="meta_json",
                              format::Symbol=:wide,
                              layout::Union{Nothing,Symbol}=nothing,
                              metadata_path::Union{Nothing,AbstractString}=nothing,
                              require_metadata::Bool=false,
                              validate_feature_schema::Bool=true)
    raw = NPZ.npzread(path)
    kx = String(x_key)
    kn = String(names_key)
    ki = String(ids_key)
    km = String(meta_key)
    haskey(raw, kx) || throw(ArgumentError("load_features_npz: missing key $(kx)"))
    haskey(raw, kn) || throw(ArgumentError("load_features_npz: missing key $(kn)"))
    haskey(raw, ki) || throw(ArgumentError("load_features_npz: missing key $(ki)"))
    haskey(raw, km) || throw(ArgumentError("load_features_npz: missing key $(km)"))

    Xraw = raw[kx]
    Xraw isa AbstractMatrix || throw(ArgumentError("load_features_npz: key $(kx) must contain a matrix"))
    X0 = Matrix(Xraw)

    names = String[]
    ids = String[]
    try
        names = String.(JSON3.read(_decode_json_bytes(raw[kn], kn)))
        ids = String.(JSON3.read(_decode_json_bytes(raw[ki], ki)))
    catch err
        throw(ArgumentError("load_features_npz: failed to decode names/ids JSON payloads ($(err))"))
    end

    md = nothing
    try
        md = JSON3.read(_decode_json_bytes(raw[km], km))
    catch err
        throw(ArgumentError("load_features_npz: failed to decode metadata JSON payload ($(err))"))
    end
    validate_feature_schema && FEA.validate_feature_metadata_schema(md)

    # Optional sidecar support for parity with Arrow/Parquet paths.
    mpath = metadata_path === nothing ? FEA.default_feature_metadata_path(path) : String(metadata_path)
    if isfile(mpath)
        md = FEA.load_metadata_json(mpath; validate_feature_schema=validate_feature_schema)
    elseif require_metadata
        throw(ArgumentError("load_features_npz: metadata sidecar not found at $(mpath)"))
    end

    layout0 = layout
    if layout0 === nothing
        layout_raw = FEA._obj_get(md, "layout", "samples_by_features")
        layout0 = Symbol(String(layout_raw))
    end
    layout0 in (:samples_by_features, :features_by_samples) ||
        throw(ArgumentError("load_features_npz: unsupported layout $(layout0)"))

    X = layout0 == :samples_by_features ? X0 : permutedims(X0)
    size(X, 1) == length(ids) ||
        throw(DimensionMismatch("load_features_npz: X sample dimension $(size(X,1)) does not match ids length $(length(ids))"))
    size(X, 2) == length(names) ||
        throw(DimensionMismatch("load_features_npz: X feature dimension $(size(X,2)) does not match names length $(length(names))"))

    meta = (metadata=md, format=format)
    return FEA.FeatureSet(X, Symbol.(names), ids, meta)
end

end # module
