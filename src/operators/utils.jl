# Rotate the x,y-components of a 3D vector with angle θ.
@inline function rotxy(sinθ, cosθ, (x, y, z)::Isochromat{T}) where {T}
    u = cosθ * x - sinθ * y
    v = cosθ * y + sinθ * x
    w = z
    return Isochromat{T}(u, v, w)
end

@inline function off_resonance_rotation(Δt::T, p::AbstractTissueProperties) where {T}
    if hasB₀(p)
        θ = π * Δt * p.B₀ * 2
        return exp(im * θ)
    else
        return one(T)
    end
end

@inline E₁(Δt, T₁) = exp(-Δt * inv(T₁))
@inline E₂(Δt, T₂) = exp(-Δt * inv(T₂))

# Add methods that accept state (either Isochromat or AbstractConfigurationStates) as argument. The state is not used in these computations but makes it possible to implement manual AD that dispatches on the state
@inline E₁(state::S, Δt, T₁) where {S<:Union{Isochromat,AbstractConfigurationStates}} = E₁(Δt, T₁)
@inline E₂(state::S, Δt, T₂) where {S<:Union{Isochromat,AbstractConfigurationStates}} = E₂(Δt, T₂)
@inline off_resonance_rotation(state::S, Δt, p) where {S<:Union{Isochromat,AbstractConfigurationStates}} = off_resonance_rotation(Δt, p)