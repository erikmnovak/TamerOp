module TamerOpProgressLoggingExt

using Logging
using ProgressLogging
using UUIDs: uuid4, UUID

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

mutable struct _ProgressState
    total::Int
    done::Base.Threads.Atomic{Int}
    lock::ReentrantLock
    id::UUID
    label::String
end

function _init(total::Int, label::AbstractString)
    state = _ProgressState(max(total, 1),
                           Base.Threads.Atomic{Int}(0),
                           ReentrantLock(),
                           uuid4(),
                           String(label))
    Logging.@logmsg(ProgressLogging.ProgressLevel, state.label, progress=nothing, _id=state.id)
    return state
end

function _step!(state::_ProgressState, delta::Int=1)
    ndone = Base.Threads.atomic_add!(state.done, delta) + delta
    frac = min(1.0, ndone / state.total)
    Base.lock(state.lock)
    try
        Logging.@logmsg(ProgressLogging.ProgressLevel, state.label, progress=frac, _id=state.id)
    finally
        Base.unlock(state.lock)
    end
    return nothing
end

function _finish!(state::_ProgressState)
    Base.lock(state.lock)
    try
        Logging.@logmsg(ProgressLogging.ProgressLevel, state.label, progress="done", _id=state.id)
    finally
        Base.unlock(state.lock)
    end
    return nothing
end

FEA._set_progress_impl!((init=_init, step!=_step!, finish!=_finish!))

end # module
