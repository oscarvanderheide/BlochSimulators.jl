
@inline global_id() = (blockIdx().x - 1) * blockDim().x + threadIdx().x

# Convert non-isbits fields of structs (e.g. sequences, trajectories)
# to CuArrays.
# Inspired by KomaMRI.jl (KomaMRI.jl/src/simulation/GPUFunctions.jl),
# which in turn is inspired by Flux.jl (Flux.jl/src/functor.jl)

"""
    gpu(x)

Move x to CUDA device. It uses Functors.fmap to recursively
traverse the fields of the struct x, converting <:AbstractArrays to CuArrays,
and ignoring isbitsarrays. For custom structs (e.g. <:BlochSimulator or <:AbstractTrajectory),
it is required that typeof(x) be made a Functors.@functor (e.g. @functor FISP).
"""
gpu(x) = fmap(_gpu, x; exclude = _isleaf)

_gpu(x) = x
_gpu(x::AbstractArray) = CUDA.CuArray(x) # no automatic conversion to Float32
_gpu(x::AbstractTissueParameters) = x
_gpu(x::StaticArray) = x

# make custom _isleaf function to include isbitsarrays
_isbitsarray(::AbstractArray{<:Real}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false
_isleaf(x) = _isbitsarray(x) || isleaf(x)

export gpu