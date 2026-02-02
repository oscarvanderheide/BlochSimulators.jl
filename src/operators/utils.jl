"""
    rotxy(sinθ, cosθ, m::Isochromat{T}) where {T}

Rotate the x,y-components of a 3D vector (Isochromat) with angle θ around the z-axis.

# Arguments
- `sinθ`: Sine of rotation angle (dimensionless)
- `cosθ`: Cosine of rotation angle (dimensionless)
- `m`: Isochromat to rotate

# Returns
- Rotated isochromat
"""
@inline function rotxy(sinθ, cosθ, (x, y, z)::Isochromat{T}) where {T}
    u = cosθ * x - sinθ * y
    v = cosθ * y + sinθ * x
    w = z
    return Isochromat{T}(u, v, w)
end

"""
    off_resonance_rotation(Δt::T, p::AbstractTissueProperties) where {T}

Calculate complex rotation factor due to off-resonance (B₀).

# Arguments
- `Δt`: Time duration in **seconds**
- `p`: Tissue properties. If `hasB₀(p)`, uses `p.B₀` (in **Hz**)

# Returns
- Complex rotation factor `exp(im * 2π * B₀ * Δt)` (dimensionless)
"""
@inline function off_resonance_rotation(Δt::T, p::AbstractTissueProperties) where {T}
    if hasB₀(p)
        θ = π * Δt * p.B₀ * 2
        return exp(im * θ)
    else
        return one(T)
    end
end

"""
    E₁(Δt, T₁)

Calculate T₁ relaxation factor.

# Arguments
- `Δt`: Time duration in **seconds**
- `T₁`: Longitudinal relaxation time constant in **seconds**

# Returns
- Relaxation factor `exp(-Δt/T₁)` (dimensionless)
"""
@inline E₁(Δt, T₁) = exp(-Δt * inv(T₁))

"""
    E₂(Δt, T₂)

Calculate T₂ relaxation factor.

# Arguments
- `Δt`: Time duration in **seconds**
- `T₂`: Transverse relaxation time constant in **seconds**

# Returns
- Relaxation factor `exp(-Δt/T₂)` (dimensionless)
"""
@inline E₂(Δt, T₂) = exp(-Δt * inv(T₂))

# Add methods that accept state (either Isochromat or AbstractConfigurationStates) as argument. The state is not used in these computations but makes it possible to implement manual AD that dispatches on the state
@inline E₁(state::S, Δt, T₁) where {S<:Union{Isochromat,AbstractConfigurationStates}} = E₁(Δt, T₁)
@inline E₂(state::S, Δt, T₂) where {S<:Union{Isochromat,AbstractConfigurationStates}} = E₂(Δt, T₂)
@inline off_resonance_rotation(state::S, Δt, p) where {S<:Union{Isochromat,AbstractConfigurationStates}} = off_resonance_rotation(Δt, p)