# # Custom Sequence Struct

# In this example we are going to generate a custom sequence
# struct for a multi-echo sequence with 90 degree excitation
# and full relaxation in between shots

using StaticArrays
using ComputationalResources
using BlochSimulators

import BlochSimulators: @functor, @adapt_structure, output_dimensions, output_eltype, simulate!, Isochromat, rotate, E₁, E₂, off_resonance_rotation, initial_conditions, rotxy, decay, sample_transverse!, regrowth

struct MultiShot{T,nTE} <: IsochromatSimulator{T}
    nshots::Int
    TE::SVector{nTE,T}
    TR::T
    flipangle::T
end

@functor MultiShot
@adapt_structure MultiShot

output_dimensions(s::MultiShot) = s.nshots * length(s.TE)
output_eltype(s::MultiShot{T,nTE}) where {T,nTE} = Complex{T}

function simulate!(echos, sequence::MultiShot, m::Isochromat, parameters::AbstractTissueParameters)

    T₁ = parameters.T₁
    T₂ = parameters.T₂
    TE = sequence.TE
    TR = sequence.TR
    nTE = length(sequence.TE)

    excite(m) = rotate(m, deg2rad(sequence.flipangle), 0, 0, 0, parameters) # no gradients

    E₁ᵀᴿ = E₁(m, TR, T₁) # E₁ for whole TR
    E₁ᵀᴱ = E₁.((m,), TE, T₁) # E₁ for each ech TR

    E₂ᵀᴿ = E₂(m, TR, T₂)
    E₂ᵀᴱ = E₂.((m,), TE, T₂) # SVector

    B₀rotᵀᴿ = off_resonance_rotation(m, TR, parameters)
    B₀rotᵀᴱ = off_resonance_rotation.((m,), TE, (parameters,))

    idx = 1

    m = initial_conditions(m)

    @inbounds for shot in 1:sequence.nshots

        m = excite(m)

        for i in 1:nTE

            # apply T₂ decay and B₀ rotation for this echo time
            mᵀᴱ = rotxy(reim(B₀rotᵀᴱ[i])..., m)
            mᵀᴱ = decay(mᵀᴱ, E₁ᵀᴱ[i], E₂ᵀᴱ[i])

            # sample the transverse magnetization
            sample_transverse!(echos, idx, mᵀᴱ)

            idx += 1
        end

        # apply T₂ decay, T₁ regrowth and B₀ rotation for entire TR
        m = rotxy(reim(B₀rotᵀᴿ)..., m)
        m = decay(m, E₁ᵀᴿ, E₂ᵀᴿ)
        m = regrowth(m, E₁ᵀᴿ)

    end
end

nshots = 16
nTE = 16
TE = SVector{nTE}(1:16) .* (5*10^-3)
TR = 100 * 10^-3
flipangle = complex(40.0)

sequence = MultiShot(nshots, TE, TR, 90.0)
parameters = T₁T₂B₀(1.0,0.1,20.0)

echos = simulate(CPU1(), sequence, [parameters])

pp = [T₁T₂B₀(1.0,0.1,20.0) for i = 1:100000];
echos = simulate(CUDALibs(), gpu(f32(sequence)), gpu(f32(pp[1:10])))

echos = simulate(CUDALibs(), gpu(f32(sequence)), gpu(f32(pp)))
