"""
    Generic2D{T,V,M,S} where {T<:AbstractFloat, V<:AbstractVector, M<:AbstractMatrix, S} <: IsochromatSimulator{T}

Simulate a generic 2D sequence defined by arrays containing RF and gradient waveforms. Contains a
loop over z locations to take into account slice profile effects. The Δt vector stores the time intervals
for the waveforms.

# Fields
- `RF::V{Complex{T}}`: Vector with (complex) RF values during each time interval
- `GR::M{T}`: Matrix with GRx, GRy and GRz values during each time interval
- `sample::S`: Vector with Bool's to indicate the sample points
- `Δt::V{T}`: Vector with time intervals
- `z::V{T}`: Vector with different positions along the slice direction 
"""
struct Generic2D{T,V<:AbstractVector{Complex{T}},W<:AbstractVector{T},M<:AbstractMatrix{T},S} <: IsochromatSimulator{T}
    RF::V
    GR::M
    sample::S
    Δt::W
    z::W
end

# To be able to change precision and send to CUDA device
@functor Generic2D
@adapt_structure Generic2D

export Generic2D

# Methods needed to allocate an output array of the correct size and type
output_dimensions(sequence::Generic2D) = sum(sequence.sample)
output_eltype(sequence::Generic2D) = Isochromat{eltype(sequence.GR)}

@inline function simulate_echos!(output, sequence::Generic2D{T}, m, (p::AbstractTissueParameters)) where T

    Δt = sequence.Δt
    GR = sequence.GR
    RF = sequence.RF

    γ = 26753.0
    T₁, T₂ = p.T₁, p.T₂

    nr_timesteps = length(Δt)

    for z in sequence.z # loop over z locations

        # set magnetization to (0,0,1)
        m = initial_conditions(m)

        m = invert(m)

        # reset sample counter that is used to index the output array when storing the magnetization
        sample_counter = 1

        for t in 1:nr_timesteps

            γΔtGR = @. γ*Δt[t]*(GR[1,t], GR[2,t], GR[3,t])
            γΔtRF = γ*Δt[t]*RF[t]

            # GR, RF and B₀ induced rotation
            m = rotate(m, γΔtRF, γΔtGR, (p.x, p.y, z), Δt[t], p)

            # T₁ and T₂ decay, T₁ regrowth
            E₁, E₂ = exp(-Δt[t]/T₁), exp(-Δt[t]/T₂)

            m = decay(m, E₁, E₂)
            m = regrowth(m, E₁)

            # sample magnetization
            if sequence.sample[t]
                sample_xyz!(output, sample_counter, m)
                sample_counter += 1
            end
        end
    end

    return nothing
end
