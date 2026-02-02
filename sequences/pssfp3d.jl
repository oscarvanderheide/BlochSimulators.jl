"""
    pSSFP3D{T<:AbstractFloat,N,U<:AbstractVector{Complex{T}},V<:Number} <: IsochromatSimulator{T}

This struct is used to simulate an inversion-recovery, gradient-balanced, transient-state sequence with varying flip angle scheme based on the isochromat model. The TR and TE are fixed throughout the sequence. The RF excitation waveform can be discretized
in time but no slice profile mechanism is provided. The sequence also uses an 'α/2' prepulse after the inversion.

Within each TR, multiple time steps are used to simulate the RF excitation. Then, in one time step we go from the
end of the RF excitation to the echo time (applying slice refocussing gradient, T₂ decay and B₀ rotation), and again in one time step from the echo time to the start of the next RF excitation.

# Fields
- `RF_train::U`: Vector with flip angle for each TR. `abs.(RF_train)` are RF flip angles in **degrees** and
    `angle.(RF_train)` are RF phases in **radians**.
- `TR::T`: Repetition time in **seconds**, assumed constant during the sequence
- `γΔtRF::SVector{N}{V}`: Time-discretized RF waveform, normalized to flip angle of 1 degree. Units: **radians**
- `Δt::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}`: Time intervals in **seconds** for each sample of excitation pulse (ex),
    inversion delay (inv) and time between RF and TE (pr)
"""
struct pSSFP3D{T<:AbstractFloat,N,U<:AbstractVector{Complex{T}},V<:Number} <: IsochromatSimulator{T}
    RF_train::U
    TR::T
    γΔtRF::SVector{N}{V}
    Δt::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}
end

# To be able to change precision and send to CUDA device
@functor pSSFP3D
@adapt_structure pSSFP3D

export pSSFP3D

# Methods needed to allocate an output array of the correct size and type
output_size(sequence::pSSFP3D) = length(sequence.RF_train)
output_eltype(sequence::pSSFP3D) = eltype(sequence.RF_train)

# Sequence implementation
@inline function simulate_magnetization!(magnetization, sequence::pSSFP3D, m, p::AbstractTissueProperties)

    T₁, T₂ = p.T₁, p.T₂

    γΔtRFᵉˣ = sequence.γΔtRF
    Δtᵉˣ = sequence.Δt.ex
    E₁ᵉˣ, E₂ᵉˣ = E₁(m, Δtᵉˣ, T₁), E₂(m, Δtᵉˣ, T₂)

    Δtᵖʳ = sequence.Δt.pr
    E₁ᵖʳ, E₂ᵖʳ = E₁(m, Δtᵖʳ, T₁), E₂(m, Δtᵖʳ, T₂)

    E₁ⁱⁿᵛ, E₂ⁱⁿᵛ = E₁(m, sequence.Δt.inv, T₁), E₂(m, sequence.Δt.inv, T₂)

    𝟘 = zero(T₁)

    # Simulate excitation with flip angle θ using hard pulse approximation of the normalized RF-waveform γΔtRF
    excite = @inline function (m, θ)
        for ⚡ in (θ * γΔtRFᵉˣ)
            m = rotate(m, ⚡, 𝟘, 𝟘, Δtᵉˣ, p)
            m = decay(m, E₁ᵉˣ, E₂ᵉˣ)
            m = regrowth(m, E₁ᵉˣ)
        end
        return m
    end

    # Slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth
    precess = @inline function (m)
        m = rotate(m, 𝟘, 𝟘, Δtᵖʳ, p)
        m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
        m = regrowth(m, E₁ᵖʳ)
        return m
    end

    # reset spin to initial conditions
    m = initial_conditions(m)

    # apply inversion pulse
    m = invert(m, p)

    m = decay(m, E₁ⁱⁿᵛ, E₂ⁱⁿᵛ)
    m = regrowth(m, E₁ⁱⁿᵛ)

    # apply "alpha over two" pulse
    θ₀ = -sequence.RF_train[1] / 2
    m = excite(m, θ₀)

    # slice select re- & prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
    m = rotate(m, 𝟘, 𝟘, Δtᵖʳ, p)
    m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
    m = regrowth(m, E₁ᵖʳ)

    # simulate pSSFP3D sequence with varying flipangles
    for (TR, θ) ∈ enumerate(sequence.RF_train)
        # simulate RF pulse and slice-selection gradient
        m = excite(m, θ)
        # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until TE
        m = precess(m)
        # sample magnetization at echo time
        sample_transverse!(magnetization, TR, m)
        # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
        m = precess(m)
    end

    return nothing
end

# add method to getindex to reduce sequence length with convenient syntax (idx is something like 1:nr_of_readouts)
Base.getindex(seq::pSSFP3D, idx) = typeof(seq)(seq.RF_train[idx], seq.TR, seq.γΔtRF, seq.Δt, seq.γΔtGRz, seq.z)

# Nicer printing of sequence information in REPL
# Base.show(io::IO, ::MIME"text/plain", seq::pSSFP3D) = begin
Base.show(io::IO, seq::pSSFP3D) = begin
    println("")
    println(io, "pSSFP3D sequence")
    println(io, "RF_train: ", typeof(seq.RF_train), " (degrees)")
    println(io, "TR:       ", seq.TR, " s")
    println(io, "γΔtRF:    ", "SVector{$(length(seq.γΔtRF))}{$(eltype(seq.γΔtRF))} (radians)")
    println(io, "Δt:       ", seq.Δt, " (s)")
end

