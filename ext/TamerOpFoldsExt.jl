module TamerOpFoldsExt

using Folds

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

function _foreach_indexed(n::Int, f; chunk_size::Int=0, deterministic::Bool=true)
    n <= 1 && return nothing
    if chunk_size > 0
        Folds.foreach(1:n; basesize=chunk_size) do i
            f(i)
        end
    else
        Folds.foreach(1:n) do i
            f(i)
        end
    end
    return nothing
end

FEA._set_batch_impl!((foreach_indexed=_foreach_indexed,))

end # module
