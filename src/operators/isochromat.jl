### TYPES ###

"""
    struct Isochromat{T<:Real} <: FieldVector{3,T}
        x::T
        y::T
        z::T
    end

Holds the x,y,z components of a spin isochromat in a FieldVector, which is a `StaticVector`
(from the package `StaticArrays`) with custom fieldnames.
"""
struct Isochromat{T<:Real} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

# Ensures that operations with/on Isochromats return Isochromats. See the FieldVector
# example from the StaticArrays documentation.
StaticArrays.similar_type(::Type{Isochromat{T}}, ::Type{T}, s::Size{(3,)}) where {T<:Real} = Isochromat{T}

### METHODS

# Initialize States

"""
    initialize_states(::AbstractResource, ::IsochromatSimulator{T}) where T

Initialize a spin isochromat to be used throughout a simulation of the sequence.

This may seem redundant but to is necessary to share the same programming interface with
`EPGSimulators`.
"""
@inline function initialize_states(::AbstractResource, ::IsochromatSimulator{T}) where {T}
    return Isochromat{T}(0, 0, 0)
end

# Initial conditions

"""
    initial_conditions(m::Isochromat{T}) where T

Return a spin isochromat with `(x,y,z) = (0,0,1)`.
"""
# Set initial conditions

@inline function initial_conditions(m::Isochromat{T}) where {T}
    return Isochromat{T}(0, 0, 1)
end

# Rotate

"""
    rotate(m::Isochromat{T}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties, ΔtΔω=zero(T)) where T

RF, gradient and/or ΔB₀ induced rotation of Isochromat computed using Rodrigues' rotation
formula.

# Arguments
- `m`: Input isochromat state.
- `γΔtRF`: Complex value representing RF pulse effect (radians). `B₁` scaling from `p` is
  applied internally if `hasB₁(p)`.
- `γΔtGR`: Tuple/Vector representing gradient effect `γ * G * Δt` (radians/centimeter).
- `r`: Position vector `(x,y,z)` [cm].
- `Δt`: Time step duration (seconds).
- `p`: Tissue properties (`AbstractTissueProperties`). `B₀` effects from `p` are applied
  internally if `hasB₀(p)`.
- `ΔtΔω`: Additional off-resonance phase accumulated during `Δt` (radians). Often `Δt * Δω`,
  where `Δω` is in rad/s.

# Returns
- Rotated isochromat state.
"""
@inline function rotate(m::Isochromat{T}, γΔtRF, γΔtGR, r, Δt, p::AbstractTissueProperties, ΔtΔω=zero(T)) where {T}

    # Determine rotation vector a
    aˣ = real(γΔtRF)
    aʸ = -imag(γΔtRF)
    aᶻ = -(γΔtGR ⋅ r + ΔtΔω)

    hasB₁(p) && (aˣ *= p.B₁)
    hasB₁(p) && (aʸ *= p.B₁)
    hasB₀(p) && (aᶻ -= Δt * 2 * π * p.B₀)

    a = SVector{3,T}(aˣ, aʸ, aᶻ)

    # Angle of rotation is norm of rotation vector
    θ = norm(a)

    if !iszero(θ)
        # Normalize rotation vector
        k = inv(θ) * a
        # Perform rotation (Rodrigues formula)
        sinθ, cosθ = sincos(θ)
        m = (cosθ * m) + (sinθ * k × m) + (k ⋅ m * (one(T) - cosθ) * k)
    end

    return m
end

"""
    rotate(m::Isochromat, γΔtGRz, z, Δt, p::AbstractTissueProperties)

Rotation of Isochromat without RF (so around z-axis only) due to gradients and B0 (i.e.
refocussing slice select gradient).
"""
@inline function rotate(m::Isochromat, γΔtGR, r, Δt, p::AbstractTissueProperties)
    # Determine rotation angle θ
    θ = -γΔtGR ⋅ r
    hasB₀(p) && (θ -= Δt * π * p.B₀ * 2)
    # Perform rotation in xy plane
    sinθ, cosθ = sincos(θ)
    return rotxy(sinθ, cosθ, m)
end

# Decay

"""
    decay(m::Isochromat{T}, E₁, E₂) where T

Apply T₂ decay to transverse component and T₁ decay to longitudinal component of
`Isochromat`.
"""
@inline function decay(m::Isochromat{T}, E₁, E₂) where {T}
    return m .* Isochromat{T}(E₂, E₂, E₁)
end

# Regrowth

"""
    regrowth(m::Isochromat{T}, E₁) where T

Apply T₁ regrowth to longitudinal component of `Isochromat`.
"""
@inline function regrowth(m::Isochromat{T}, E₁) where {T}
    return m + Isochromat{T}(0, 0, 1 - E₁)
end

# Invert

"""
    invert(m::Isochromat{T}, p::AbstractTissueProperties) where T

Invert z-component of `Isochromat` (assuming spoiled transverse magnetization so
xy-component zero).
"""
@inline function invert(m::Isochromat{T}, p::AbstractTissueProperties) where {T}
    # Determine rotation angle θ
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    return Isochromat{T}(0, 0, cos(θ) * m.z)
end

"""
    invert(m::Isochromat{T}, p::AbstractTissueProperties) where T

Invert `Isochromat` with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline invert(m::Isochromat{T}) where {T} = Isochromat{T}(0, 0, -m.z)

# Sample

"""
    sample!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)

Sample transverse magnetization from `Isochromat`. The "+=" is needed for 2D sequences where
slice profile is taken into account.
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)
    @inbounds output[index] += complex(m.x, m.y)
end

"""
    sample_xyz!(output, index::Union{Integer,CartesianIndex}, m::Isochromat)

Sample m.x, m.y and m.z components from `Isochromat`. The "+=" is needed for 2D sequences
where slice profile is taken into account.
"""
@inline function sample_xyz!(output::AbstractArray{<:S}, index::Union{Integer,CartesianIndex}, m::Isochromat) where {S}
    @inbounds output[index] += S(m.x, m.y, m.z)
end
