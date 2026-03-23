module TamerOpDataFramesExt

using DataFrames

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

"""
    DataFrame(fs::FeatureSet; format=:wide, copycols=true)

Convenience DataFrame constructor for feature outputs.
"""
function DataFrames.DataFrame(fs::FEA.FeatureSet; format::Symbol=:wide, copycols::Bool=true)
    return DataFrames.DataFrame(FEA.feature_table(fs; format=format); copycols=copycols)
end

DataFrames.DataFrame(t::FEA.FeatureSetWideTable; copycols::Bool=true) =
    DataFrames.DataFrame(FEA.feature_table(t.fs; format=:wide); copycols=copycols)

DataFrames.DataFrame(t::FEA.FeatureSetLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(FEA.feature_table(t.fs; format=:long); copycols=copycols)

DataFrames.DataFrame(t::FEA.EulerSurfaceLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::FEA.PersistenceImageLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::FEA.MPLandscapeLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::FEA.PointSignedMeasureLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

end # module
