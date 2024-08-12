# Change precision of custom structs
# Inspired by KomaMRI.jl (KomaMRI.jl/src/simulation/GPUFunctions.jl),
# which in turn is inspired by Flux.jl (Flux.jl/src/functor.jl)

export f32, f64

"""
    f32(x)

Change precision of `x` to `Float32`. It uses `Functors.fmap` to recursively
traverse the fields of the struct `x`. For custom structs (e.g. `<:BlochSimulator`
or `<:AbstractTrajectory`), it is required that `typeof(x)` be made a
`Functors.@functor` (e.g. `@functor FISP`).

It may be necessary to add new adapt rules (by adding new methods to adapt_storage)
if new structs with complicated nested fields are introduced.
"""
f32(x) = fmap(y -> _change_precision(Float32, y), x)

"""
    f64(x)

Change precision of `x` to `Float64`. It uses `Functors.fmap` to recursively
traverse the fields of the struct `x`. For custom structs (e.g. `<:BlochSimulator`
or `<:AbstractTrajectory`), it is required that `typeof(x)` be made a
`Functors.@functor` (e.g. `@functor FISP`).

It may be necessary to add new adapt rules (by adding new methods to `adapt_storage`)
if new structs with complicated nested fields are introduced.
"""
f64(x) = fmap(y -> _change_precision(Float64, y), x)

# default: do nothing
_change_precision(::Type{T}, x) where {T} = x
# rule for single real number
_change_precision(::Type{T}, x::AbstractFloat) where {T} = T(x)
# rule for single complex number
_change_precision(::Type{T}, x::Complex{<:AbstractFloat}) where {T} = Complex{T}(x)
# rule for real array
_change_precision(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T} = convert(AbstractArray{T}, x)
# rule for complex array
_change_precision(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T} = convert(AbstractArray{Complex{T}}, x)
# rule for tuple (not needed it seems)
# _change_precision(::Type{T}, x::NTuple{N}) where {N,T} = NTuple{N,T}(x)
# rule for namedtuple (not needed either it seems)
# _change_precision(::Type{T}, x::NamedTuple{M,NTuple{N}}) where {M,N,T} = NamedTuple{M, NTuple{N,T}}(x)
# rule for AbstractTissueProperties
_change_precision(::Type{T}, x::P) where {T,P<:AbstractTissueProperties} = P.name.wrapper{T}(x)