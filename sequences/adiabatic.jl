"""
    AdiabaticInversion{T<:Real, V<:AbstractVector} <: IsochromatSimulator{T}

This struct is used to simulate an adiabatic inversion pulse. This struct itself
could be used as field in other sequence structs.

# Fields
- `γΔtA::V`: Time-dependent amplitude modulation. Units: **radians** (γ·B₁·Δt)
- `Δω::V`: Time-dependent frequency modulation. Units: **rad/s** (angular frequency offset)
- `Δt::T`: Time discretization step in **seconds**, assumed constant
"""
struct AdiabaticInversion{T<:Real,V<:AbstractVector} <: IsochromatSimulator{T}
    γΔtA::V
    Δω::V
    Δt::T
end
# Methods needed to allocate an output array of the correct size and type
output_size(sequence::AdiabaticInversion) = 1 # Only record the magnetization at the end of the inversion pulse
output_eltype(sequence::AdiabaticInversion{T,V}) where {T,V} = Isochromat{T}

# To be able to change precision and send to CUDA device
@functor AdiabaticInversion
@adapt_structure AdiabaticInversion

## Sequence implementation
function simulate_magnetization!(output, sequence::AdiabaticInversion, m, p::AbstractTissueProperties)

    T₁, T₂ = p.T₁, p.T₂
    E₁, E₂ = BlochSimulators.E₁(m, sequence.Δt, T₁), BlochSimulators.E₂(m, sequence.Δt, T₂)

    m = initial_conditions(m)

    γΔtA, Δω, Δt = sequence.γΔtA, sequence.Δω, sequence.Δt

    𝟘 = zero(Δt)

    @inbounds for t in eachindex(sequence.γΔtA)

        m = rotate(m, γΔtA[t], 𝟘, 𝟘, Δt, p, Δt * Δω[t])
        m = decay(m, E₁, E₂)
        m = regrowth(m, E₁)

    end

    sample_xyz!(output, 1, m)
end