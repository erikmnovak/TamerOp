module TamerOpTablesExt

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
const Inv = TO.Invariants

@inline function _named_cols(names::Vector{Symbol}, cols::Vector)
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

function _feature_wide_columntable(fs::FEA.FeatureSet)
    ns, nf = size(fs.X)
    names = Vector{Symbol}(undef, nf + 1)
    cols = Vector{Any}(undef, nf + 1)
    names[1] = :id
    cols[1] = copy(fs.ids)
    @inbounds for j in 1:nf
        names[j + 1] = fs.names[j]
        cols[j + 1] = collect(view(fs.X, :, j))
    end
    return _named_cols(names, cols)
end

function _feature_long_columntable(fs::FEA.FeatureSet)
    ns, nf = size(fs.X)
    n = ns * nf
    sample_index = Vector{Int}(undef, n)
    id = Vector{String}(undef, n)
    feat = Vector{Symbol}(undef, n)
    value = Vector{eltype(fs.X)}(undef, n)
    k = 1
    @inbounds for i in 1:ns
        ids = fs.ids[i]
        for j in 1:nf
            sample_index[k] = i
            id[k] = ids
            feat[k] = fs.names[j]
            value[k] = fs.X[i, j]
            k += 1
        end
    end
    return (sample_index=sample_index, id=id, feature=feat, value=value)
end

function _euler_surface_long_columntable(t::FEA.EulerSurfaceLongTable)
    nx = length(t.x)
    ny = length(t.y)
    n = nx * ny
    id = Vector{String}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    value = Vector{eltype(t.values)}(undef, n)
    k = 1
    @inbounds for i in 1:nx
        xi = t.x[i]
        for j in 1:ny
            id[k] = t.id
            x[k] = xi
            y[k] = t.y[j]
            value[k] = t.values[i, j]
            k += 1
        end
    end
    return (id=id, x=x, y=y, value=value)
end

function _persistence_image_long_columntable(t::FEA.PersistenceImageLongTable)
    PI = t.image
    nx = length(PI.xgrid)
    ny = length(PI.ygrid)
    n = nx * ny
    id = Vector{String}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    value = Vector{Float64}(undef, n)
    k = 1
    @inbounds for ix in 1:nx
        xv = PI.xgrid[ix]
        for iy in 1:ny
            id[k] = t.id
            x[k] = xv
            y[k] = PI.ygrid[iy]
            value[k] = PI.values[iy, ix]
            k += 1
        end
    end
    return (id=id, x=x, y=y, value=value)
end

function _mp_landscape_long_columntable(t::FEA.MPLandscapeLongTable{D,O}) where {D,O}
    L = t.landscape
    nd, no, kmax, nt = size(L.values)
    n = nd * no * kmax * nt
    id = Vector{String}(undef, n)
    dir_index = Vector{Int}(undef, n)
    offset_index = Vector{Int}(undef, n)
    direction = Vector{D}(undef, n)
    offset = Vector{O}(undef, n)
    layer = Vector{Int}(undef, n)
    tgrid = Vector{Float64}(undef, n)
    value = Vector{Float64}(undef, n)
    weight = Vector{Float64}(undef, n)
    p = 1
    @inbounds for i in 1:nd, j in 1:no, k in 1:kmax, l in 1:nt
        id[p] = t.id
        dir_index[p] = i
        offset_index[p] = j
        direction[p] = L.directions[i]
        offset[p] = L.offsets[j]
        layer[p] = k
        tgrid[p] = L.tgrid[l]
        value[p] = L.values[i, j, k, l]
        weight[p] = L.weights[i, j]
        p += 1
    end
    return (id=id, dir_index=dir_index, offset_index=offset_index,
            direction=direction, offset=offset, layer=layer,
            t=tgrid, value=value, weight=weight)
end

function _point_signed_measure_long_columntable(t::FEA.PointSignedMeasureLongTable{N,T,W}) where {N,T,W}
    pm = t.measure
    n = length(pm)
    id = fill(t.id, n)
    inds = Vector{NTuple{N,Int}}(undef, n)
    coords = Vector{NTuple{N,T}}(undef, n)
    wts = Vector{W}(undef, n)
    @inbounds for i in 1:n
        idx = pm.inds[i]
        inds[i] = idx
        coords[i] = ntuple(d -> pm.axes[d][idx[d]], N)
        wts[i] = pm.wts[i]
    end
    return (id=id, indices=inds, coords=coords, weight=wts)
end

# ---------------------------------------------------------------------------
# FeatureSet tables
# ---------------------------------------------------------------------------

Tables.istable(::Type{<:FEA.FeatureSet}) = true
Tables.columnaccess(::Type{<:FEA.FeatureSet}) = true
Tables.columns(fs::FEA.FeatureSet) = _feature_wide_columntable(fs)
Tables.schema(fs::FEA.FeatureSet) = Tables.schema(Tables.columns(fs))

Tables.istable(::Type{<:FEA.FeatureSetWideTable}) = true
Tables.columnaccess(::Type{<:FEA.FeatureSetWideTable}) = true
Tables.columns(t::FEA.FeatureSetWideTable) = _feature_wide_columntable(t.fs)
Tables.schema(t::FEA.FeatureSetWideTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:FEA.FeatureSetLongTable}) = true
Tables.columnaccess(::Type{<:FEA.FeatureSetLongTable}) = true
Tables.columns(t::FEA.FeatureSetLongTable) = _feature_long_columntable(t.fs)
Tables.schema(t::FEA.FeatureSetLongTable) = Tables.schema(Tables.columns(t))

# ---------------------------------------------------------------------------
# Optional long-form invariant tables
# ---------------------------------------------------------------------------

Tables.istable(::Type{<:FEA.EulerSurfaceLongTable}) = true
Tables.columnaccess(::Type{<:FEA.EulerSurfaceLongTable}) = true
Tables.columns(t::FEA.EulerSurfaceLongTable) = _euler_surface_long_columntable(t)
Tables.schema(t::FEA.EulerSurfaceLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:FEA.PersistenceImageLongTable}) = true
Tables.columnaccess(::Type{<:FEA.PersistenceImageLongTable}) = true
Tables.columns(t::FEA.PersistenceImageLongTable) = _persistence_image_long_columntable(t)
Tables.schema(t::FEA.PersistenceImageLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:FEA.MPLandscapeLongTable}) = true
Tables.columnaccess(::Type{<:FEA.MPLandscapeLongTable}) = true
Tables.columns(t::FEA.MPLandscapeLongTable) = _mp_landscape_long_columntable(t)
Tables.schema(t::FEA.MPLandscapeLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:FEA.PointSignedMeasureLongTable}) = true
Tables.columnaccess(::Type{<:FEA.PointSignedMeasureLongTable}) = true
Tables.columns(t::FEA.PointSignedMeasureLongTable) = _point_signed_measure_long_columntable(t)
Tables.schema(t::FEA.PointSignedMeasureLongTable) = Tables.schema(Tables.columns(t))

# Direct convenience: table view defaults for core invariant objects.
Tables.istable(::Type{<:Inv.PersistenceImage1D}) = true
Tables.columnaccess(::Type{<:Inv.PersistenceImage1D}) = true
Tables.columns(pi::Inv.PersistenceImage1D) = Tables.columns(FEA.persistence_image_table(pi))
Tables.schema(pi::Inv.PersistenceImage1D) = Tables.schema(Tables.columns(pi))

Tables.istable(::Type{<:Inv.MPLandscape}) = true
Tables.columnaccess(::Type{<:Inv.MPLandscape}) = true
Tables.columns(L::Inv.MPLandscape) = Tables.columns(FEA.mp_landscape_table(L))
Tables.schema(L::Inv.MPLandscape) = Tables.schema(Tables.columns(L))

Tables.istable(::Type{<:Inv.PointSignedMeasure}) = true
Tables.columnaccess(::Type{<:Inv.PointSignedMeasure}) = true
Tables.columns(pm::Inv.PointSignedMeasure) = Tables.columns(FEA.point_signed_measure_table(pm))
Tables.schema(pm::Inv.PointSignedMeasure) = Tables.schema(Tables.columns(pm))

end # module
