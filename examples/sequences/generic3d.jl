"""
    Generic3D{T,V<:AbstractVector{Complex{T}},W<:AbstractVector{T},M<:AbstractMatrix{T},S} <: IsochromatSimulator{T}

Simulate a generic sequence defined by arrays containing RF and gradient waveforms. Unlike the Generic2D sequence, it is assumed that the excitation is homogenous over the voxel and therefore no summation over a slice direction is applied. The Δt vector stores the time intervals for the waveforms.

# Fields
- `RF::V{Complex{T}}`: Vector with (complex) RF values during each time interval
- `GR::M{T}`: Matrix with GRx, GRy and GRz values during each time interval
- `sample::S`: Vector with Bool's to indicate the sample points
- `Δt::V{T}`: Vector with time intervals
"""
struct Generic3D{T,V<:AbstractVector{Complex{T}},W<:AbstractVector{T},M<:AbstractMatrix{T},S} <: IsochromatSimulator{T}
    RF::V
    GR::M
    sample::S
    Δt::W
end

# To be able to change precision and send to CUDA device
@functor Generic3D
@adapt_structure Generic3D

export Generic3D

# Methods needed to allocate an output array of the correct size and type
output_size(sequence::Generic3D) = sum(sequence.sample)
output_eltype(sequence::Generic3D) = Isochromat{eltype(sequence.GR)}

@inline function simulate_magnetization!(output, sequence::Generic3D{T}, m, (p::AbstractTissueParameters)) where T

    Δt = sequence.Δt
    GR = sequence.GR
    RF = sequence.RF

    γ = T(26753.0)
    T₁, T₂ = p.T₁, p.T₂

    nr_timesteps = length(Δt)

    x,y,z = p.x, p.y, p.z

    # set magnetization to (0,0,1)
    m = initial_conditions(m)

    m = invert(m)

    # sample counter that is used to index the output array when storing the magnetization
    sample_counter = 1

    for t in 1:nr_timesteps

        γΔtGR = @. γ*Δt[t]*(GR[1,t], GR[2,t], GR[3,t])
        γΔtRF = γ*Δt[t]*RF[t]

        # GR, RF and B₀ induced rotation
        m = rotate(m, γΔtRF, γΔtGR, (x,y,z), Δt[t], p)

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

    return nothing
end
