# Convert non-isbits fields of structs (e.g. sequences, trajectories)
# to CuArrays.
# Inspired by KomaMRI.jl (KomaMRI.jl/src/simulation/GPUFunctions.jl),
# which in turn is inspired by Flux.jl (Flux.jl/src/functor.jl)

"""
    gpu(x)

Move `x` to CUDA device. It uses `Functors.fmap` to recursively
traverse the fields of the struct `x`, converting `<:AbstractArrays` to `CuArrays`,
and ignoring isbitsarrays. For custom structs (e.g. `<:BlochSimulator` or `<:AbstractTrajectory`),
it is required that `typeof(x)` be made a `Functors.@functor` (e.g. `@functor FISP`).
"""
gpu(x) = fmap(_gpu, x; exclude=_isleaf)

_gpu(x) = x
_gpu(x::AbstractArray) = CUDA.CuArray(x) # no automatic conversion to Float32
_gpu(x::AbstractTissueProperties) = x
_gpu(x::StaticArray) = x

# make custom _isleaf function to include isbitsarrays
_isbitsarray(::AbstractArray{<:Real}) = true
_isbitsarray(::AbstractArray{T}) where {T} = isbitstype(T)
_isbitsarray(x) = false
_isleaf(x) = _isbitsarray(x) || isleaf(x)

# Convert underlying arrays of a `StructArray` to `CuArray`s while keeping the `StructArray` wrapper intact
gpu(x::StructArray) = StructArray{eltype(x)}(gpu(StructArrays.components(x)))
gpu(x::AbstractVector{<:StructArray}) = [gpu(x[i]) for i in eachindex(x)]

export gpu

"""
    _all_arrays_are_cuarrays(x)

Returns `true` if all `AbstractArray` fields in `x` are `CuArray`s and `false` otherwise. Will also `return` if `x` does not have any `AbstractArray` fields.
"""
function _all_arrays_are_cuarrays(x)
    for name in propertynames(x)
        property = getproperty(x, name)
        if (property isa AbstractArray) && !(property isa CuArray)
            return false
        end
    end
    return true
end