"""
    pSSFP2D{T<:AbstractFloat,N,M,U<:AbstractVector{Complex{T}},V<:Number} <: IsochromatSimulator{T}

This struct is used to simulate a inversion-recovery, gradient-balanced transient-state sequence with varying flip angle scheme based on the isochromat model. The TR and TE are fixed throughout the sequence. The TR and TE are fixed throughout the sequence. Slice profile correction
is done by discretizing the RF excitation waveform in time and using multiple `Isochromat`s with different
positions along the slice direction (`z`) per voxel. The sequence also uses an 'α/2' prepulse after the inversion.

Within each TR, multiple time steps are used to simulate the RF excitation. Then, in one time step we go from the
end of the RF excitation to the echo time (applying slice refocussing gradient, T₂ decay and B₀ rotation), and again in one time step from the echo time to the start of the next RF excitation.

# Fields
- `RF_train::U`: Vector with flip angle for each TR. `abs.(RF_train)` are RF flip angles in **degrees** and
    `angle.(RF_train)` are RF phases in **radians**.
- `TR::T`: Repetition time in **seconds**, assumed constant during the sequence
- `γΔtRF::SVector{N}{V}`: Time-discretized RF waveform, normalized to flip angle of 1 degree. Units: **radians**
- `Δt::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}`: Time intervals in **seconds** for each sample of excitation pulse (ex),
    inversion delay (inv) and time between RF and TE (pr)
- `γΔtGRz::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}`: Slice select gradients for ex, inv and pr. Units: **rad/cm** (γ·G·Δt; multiplied by z position in cm gives phase in radians)
- `z::SVector{M}{T}`: Vector with different positions along the slice direction in **cm** (centimeters)
"""
struct pSSFP2D{T<:AbstractFloat,N,M,U<:AbstractVector{Complex{T}},V<:Number} <: IsochromatSimulator{T}
    RF_train::U
    TR::T
    γΔtRF::SVector{N}{V}
    Δt::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}
    γΔtGRz::NamedTuple{(:ex, :inv, :pr),NTuple{3,T}}
    z::SVector{M}{T}
end

# To be able to change precision and send to CUDA device
@functor pSSFP2D
@adapt_structure pSSFP2D

export pSSFP2D

# Methods needed to allocate an output array of the correct size and type
output_size(sequence::pSSFP2D) = length(sequence.RF_train)
output_eltype(sequence::pSSFP2D) = eltype(sequence.RF_train)

# Sequence implementation
@inline function simulate_magnetization!(magnetization, sequence::pSSFP2D, m, p::AbstractTissueProperties)

    T₁, T₂ = p.T₁, p.T₂

    γΔtRFᵉˣ = sequence.γΔtRF
    γΔtGRzᵉˣ = sequence.γΔtGRz.ex
    Δtᵉˣ = sequence.Δt.ex
    E₁ᵉˣ, E₂ᵉˣ = E₁(m, Δtᵉˣ, T₁), E₂(m, Δtᵉˣ, T₂)

    γΔtGRzᵖʳ = sequence.γΔtGRz.pr
    Δtᵖʳ = sequence.Δt.pr
    E₁ᵖʳ, E₂ᵖʳ = E₁(m, Δtᵖʳ, T₁), E₂(m, Δtᵖʳ, T₂)

    E₁ⁱⁿᵛ, E₂ⁱⁿᵛ = E₁(m, sequence.Δt.inv, T₁), E₂(m, sequence.Δt.inv, T₂)

    # Simulate excitation with flip angle θ using hard pulse approximation of the normalized RF-waveform γΔtRF
    excite = @inline function (m, θ, z)
        for ⚡ in (θ * γΔtRFᵉˣ)
            m = rotate(m, ⚡, γΔtGRzᵉˣ, z, Δtᵉˣ, p)
            m = decay(m, E₁ᵉˣ, E₂ᵉˣ)
            m = regrowth(m, E₁ᵉˣ)
        end
        return m
    end

    # Slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth
    precess = @inline function (m, z)
        m = rotate(m, γΔtGRzᵖʳ, z, Δtᵖʳ, p)
        m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
        m = regrowth(m, E₁ᵖʳ)
        return m
    end

    @inbounds for z in sequence.z

        # reset spin to initial conditions
        m = initial_conditions(m)

        # apply inversion pulse
        m = invert(m, p)
        m = decay(m, E₁ⁱⁿᵛ, E₂ⁱⁿᵛ)
        m = regrowth(m, E₁ⁱⁿᵛ)

        # apply "alpha over two" pulse
        θ₀ = -sequence.RF_train[1] / 2
        m = excite(m, θ₀, z)

        # slice select re- & prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
        m = rotate(m, 2 * γΔtGRzᵖʳ, z, Δtᵖʳ, p)
        m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
        m = regrowth(m, E₁ᵖʳ)

        # simulate pSSFP2D sequence with varying flipangles
        for (TR, θ) ∈ enumerate(sequence.RF_train)
            # simulate RF pulse and slice-selection gradient
            m = excite(m, θ, z)
            # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until TE
            m = precess(m, z)
            # sample magnetization at echo time (sum over slice direction)
            sample_transverse!(magnetization, TR, m)
            # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
            m = precess(m, z)
        end
    end

    return nothing
end

# add method to getindex to reduce sequence length with convenient syntax (idx is something like 1:nr_of_readouts)
Base.getindex(seq::pSSFP2D, idx) = typeof(seq)(seq.RF_train[idx], seq.TR, seq.γΔtRF, seq.Δt, seq.γΔtGRz, seq.z)

# Nicer printing of sequence information in REPL
# Base.show(io::IO, ::MIME"text/plain", seq::pSSFP2D) = begin
Base.show(io::IO, seq::pSSFP2D) = begin
    println("")
    println(io, "pSSFP2D sequence")
    println(io, "RF_train: ", typeof(seq.RF_train), " (degrees)")
    println(io, "TR:       ", seq.TR, " s")
    println(io, "γΔtRF:    ", "SVector{$(length(seq.γΔtRF))}{$(eltype(seq.γΔtRF))} (radians)")
    println(io, "Δt:       ", seq.Δt, " (s)")
    println(io, "γΔtGRz:   ", seq.γΔtGRz, " (rad/cm)")
    println(io, "z:        ", "SVector{$(length(seq.z))}{$(eltype(seq.z))} (cm)")
end

