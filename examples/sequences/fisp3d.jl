"""
    FISP3D{T, Ns, U<:AbstractVector} <: EPGSimulator{T,Ns}

This struct is used to simulate gradient-spoiled sequence with varying flip angle
scheme and adiabatic inversion prepulse using the EPG model for 3D sequences. The TR and TE are fixed throughout
the sequence. Unlike the 2D case, no slice profile correction is applied. However, for
3D acquisitions, slab correction may be necessary. To accomplish this, the B₁ parameter
can be modified per voxel instead. For 3D, the sequence will likely be repeated many times. After
a certain number of repetitions (usually five is enough), a "hyper" steady state is reached. We only sample
in this hyper steady state.

# Fields
- `RF_train::U` Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    angle.(RF_train) should be the RF phases in degrees.
- `sliceprofiles::V` # Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
- `TR::T`: Repetition time in seconds, assumed constant during the sequence
- `TE::T`: Echo time in seconds, assumed constant during the sequence
- `max_state::Val{Ns}`: Maximum number of states to keep track of in EPG simulation
- `TI::T`: Inversion delay after the inversion prepulse in seconds
- `TW::T`: Waiting time between repetitions in seconds
- `repetitions::Int`: Number of repetitions
- `inversion_prepulse::Bool`: With or without inversion prepulse at the start of every repetition
- 'wait_spoiling::Bool': Spoiling is assumed before the start of a next cycle
"""
# Create struct that holds parameters necessary for performing FISP simulations based on the EPG model
struct FISP3D{T<:AbstractFloat, Ns, U<:AbstractVector{Complex{T}}} <: EPGSimulator{T,Ns}
    RF_train::U
    TR::T
    TE::T
    max_state::Val{Ns}
    TI::T
    TW::T
    repetitions::Int
    inversion_prepulse::Bool
    wait_spoiling::Bool
end
# provide default value of wait_spoiling for backward compatibility
FISP3D(RF_train,TR,TE,max_state,TI,TW,repetitions,inversion_prepulse) =
                    FISP3D(RF_train,TR,TE,max_state,TI,TW,repetitions,inversion_prepulse,false)

# To be able to change precision and send to CUDA device
@functor FISP3D
@adapt_structure FISP3D

# Methods needed to allocate an output array of the correct size and type
output_dimensions(sequence::FISP3D) = length(sequence.RF_train)
output_eltype(sequence::FISP3D) = unitless(eltype(sequence.RF_train))

# If RF doesn't have phase, configuration states will be real
Ω_eltype(sequence::FISP3D) = unitless(eltype(sequence.RF_train))

# Sequence implementation
@inline function simulate_echos!(echos, sequence::FISP3D, Ω, p::AbstractTissueParameters)

    # Slab profile is simulated by modifying the B₁ value of parameters depending on their z-location

    # This needs to be modified~ (Hanna)
    T₁, T₂ = p.T₁, p.T₂
    TR, TE, TI = sequence.TR, sequence.TE, sequence.TI
    TW = sequence.TW

    E₁ᵀᴱ, E₂ᵀᴱ          = E₁(Ω, TE, T₁),    E₂(Ω, TE, T₂)
    E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ    = E₁(Ω, TR-TE, T₁), E₂(Ω, TR-TE, T₂)
    E₁ᵀᴵ, E₂ᵀᴵ          = E₁(Ω, TI, T₁),    E₂(Ω, TI, T₂)
    E₁ᵂ, E₂ᵂ =  E₁(Ω, TW, T₁),    E₂(Ω, TW, T₂)  # waiting between sequence repetitions

    eⁱᴮ⁰⁽ᵀᴱ⁾    = off_resonance_rotation(Ω, TE, p)
    eⁱᴮ⁰⁽ᵀᴿ⁻ᵀᴱ⁾ = off_resonance_rotation(Ω, TR-TE, p)
    eⁱᴮ⁰⁽ᵀᵂ⁾    = off_resonance_rotation(Ω, TW, p)    # waiting between sequence repetitions

    @inline precess(Ω, E₁, E₂, eⁱᶿ) = begin
        Ω = rotate_decay(Ω, E₁, E₂, eⁱᶿ)
        Ω = regrowth(Ω, E₁)
    end

    # Ω = initial_conditions(Ω)
    initial_conditions!(Ω)

    for repetition in (1:sequence.repetitions)

        # apply inversion pulse
        if sequence.inversion_prepulse
            invert!(Ω)
            spoil!(Ω)
            decay!(Ω, E₁ᵀᴵ, E₂ᵀᴵ)
            regrowth!(Ω, E₁ᵀᴵ)
        end

        for (TR,RF) in enumerate(sequence.RF_train)
            # mix states
            excite!(Ω, RF, p)
            # T2 decay F states, T1 decay Z states, B0 rotation until TE
            rotate_decay!(Ω, E₁ᵀᴱ, E₂ᵀᴱ, eⁱᴮ⁰⁽ᵀᴱ⁾)
            regrowth!(Ω, E₁ᵀᴱ)
            # sample F0+
            if (repetition == sequence.repetitions) # sample at the last repetition
                sample_transverse!(echos, TR, Ω)
            end
            # T2 decay F states, T1 decay Z states, B0 rotation until next RF excitation
            rotate_decay!(Ω, E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ, eⁱᴮ⁰⁽ᵀᴿ⁻ᵀᴱ⁾)
            regrowth!(Ω, E₁ᵀᴿ⁻ᵀᴱ)
            # shift F states due to dephasing gradients
            dephasing!(Ω)
        end
        # Ω = decay(Ω, E₁ᵂ, E₂ᵂ) # waiting between repetitions
        # Waiting between repetitions
        if sequence.wait_spoiling
            spoil!(Ω)
        end
        rotate_decay!(Ω, E₁ᵂ, E₂ᵂ, eⁱᴮ⁰⁽ᵀᵂ⁾)
        regrowth!(Ω, E₁ᵂ)
    end

    return nothing
end

# The _value_ of max_state needs to be part of the type, not its type (<:Int)
# That's what the Val{Ns} thing does. Because it's easy to forget doing Val(max_state) when constructing FISP,
# here's a constructor that takes care of it in case you forget.
FISP3D(RF_train, TR, TE, max_state::Int, TI, TW, repetitions, inversion_prepulse, wait_spoiling) =
                        FISP3D(RF_train, TR, TE, Val(max_state), TI, TW, repetitions, inversion_prepulse, wait_spoiling)

# Add method to getindex to reduce sequence length with convenient syntax (idx is something like 1:nr_of_readouts)
Base.getindex(seq::FISP3D, idx) = typeof(seq)(seq.RF_train[idx], seq.TR, seq.TE, seq.max_state, seq.TI, seq.TW, seq.repetitions, seq.inversion_prepulse, seq.wait_spoiling)

# Nicer printing of sequence in REPL
# Base.show(io::IO, ::MIME"text/plain", seq::FISP) = begin
Base.show(io::IO, seq::FISP3D) = begin
    println("")
    println(io, "FISP sequence")
    println(io, "RF_train:     ", typeof(seq.RF_train), " $(length(seq.RF_train)) flip angles")
    println(io, "TR:           ", seq.TR)
    println(io, "TE:           ", seq.TE)
    println(io, "max_state:    ", seq.max_state)
    println(io, "TI:           ", seq.TI)
    println(io, "TW:           ", seq.TW)
    println(io, "repetitions:  ", seq.repetitions)
    println(io, "inversion_prepulse: ", seq.inversion_prepulse)
    println(io, "wait_spoiling: ", seq.wait_spoiling)
end

export FISP3D