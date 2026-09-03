# EPG state containers
# --------------------
# Purpose: give the CPU and GPU versions of the EPG state matrix a common
# type. Functions can then accept `AbstractConfigurationStates` instead of
# having separate CPU and GPU implementations.
#
# For example, `decay!(Ω, E₁, E₂)` works with either container:
#
#   ConfigurationStates       -> CPU state matrix stored in an MMatrix
#   ConfigurationStatesSubset -> one GPU thread's states stored in an SMatrix
#
# MMatrix is mutable, so CPU values can be changed directly. SMatrix is
# immutable, so the GPU wrapper must be mutable: updating a GPU state means
# replacing its `matrix` field with a new SMatrix.
abstract type AbstractConfigurationStates{T} <: AbstractMatrix{T} end

struct ConfigurationStates{T,M<:AbstractMatrix{T}} <: AbstractConfigurationStates{T}
    matrix::M
end

ConfigurationStates(m::Matrix) = ConfigurationStates(MMatrix{size(m)...}(m))

mutable struct ConfigurationStatesSubset{T,M<:AbstractMatrix{T}} <: AbstractConfigurationStates{T}
    matrix::M
end

# Basic matrix interface
# ----------------------
# Purpose: make a state container usable like an ordinary Julia matrix.
# These overloads forward reads and queries to the matrix stored inside Ω.
#
# Examples:
#
#   size(Ω)       # calls size(Ω.matrix)
#   Ω[3, 1]       # reads Ω.matrix[3, 1]
#   view(Ω, 1, :) # views the F₊ row on CPU
Base.size(Ω::AbstractConfigurationStates) = size(Ω.matrix)
Base.getindex(Ω::AbstractConfigurationStates, indices...) = Ω.matrix[indices...]
Base.view(Ω::AbstractConfigurationStates, indices...) = view(Ω.matrix, indices...)

# Assignment differs because MMatrix is mutable while SMatrix is not:
#
#   Ω[3, 1] = 1
#
# changes the existing MMatrix on CPU, but creates an updated SMatrix and
# stores it back in the wrapper on GPU.
@inline Base.setindex!(Ω::ConfigurationStates, value, indices...) =
    setindex!(Ω.matrix, value, indices...)

@inline function Base.setindex!(Ω::ConfigurationStatesSubset, value, indices...)
    Ω.matrix = setindex(Ω.matrix, value, indices...)
    return value
end

# Whole-state mutation
# --------------------
# Purpose: support operations which replace every value in Ω while hiding
# the CPU/GPU storage difference.
#
# Examples:
#
#   copyto!(Ω, result) # store a newly calculated state matrix
#   fill!(Ω, 0)        # set every configuration state to zero
#
# On CPU these write into the existing MMatrix. On GPU they construct and
# store a new SMatrix.
@inline Base.copyto!(Ω::ConfigurationStates, source::AbstractMatrix) =
    (copyto!(Ω.matrix, source); Ω)
@inline Base.copyto!(Ω::ConfigurationStatesSubset{T,M}, source::AbstractArray) where {T,M} =
    (Ω.matrix = M(source); Ω)

@inline Base.fill!(Ω::ConfigurationStates, value) = (fill!(Ω.matrix, value); Ω)
@inline Base.fill!(Ω::ConfigurationStatesSubset, value) =
    (Ω.matrix = map(_ -> value, Ω.matrix); Ω)

# In-place broadcasting
# ---------------------
# Purpose: allow the EPG operators to use normal dotted Julia expressions.
# For example, relaxation can be written as:
#
#   Ω .*= (E₂, E₂, E₁)
#
# Julia represents that statement internally as a lazy `Broadcasted` object.
# That object contains Ω itself. `_unwrap_state` replaces Ω with Ω.matrix so
# Julia performs the calculation using the actual MMatrix or SMatrix.
@inline _unwrap_state(x) = x
@inline _unwrap_state(Ω::AbstractConfigurationStates) = Ω.matrix
@inline function _unwrap_state(broadcast::Base.Broadcast.Broadcasted)
    args = map(_unwrap_state, broadcast.args)
    return Base.Broadcast.broadcasted(broadcast.f, args...)
end

@inline Base.copyto!(Ω::ConfigurationStates, broadcast::Base.Broadcast.Broadcasted) =
    (copyto!(Ω.matrix, _unwrap_state(broadcast)); Ω)
@inline Base.copyto!(Ω::ConfigurationStatesSubset{T,M},
    broadcast::Base.Broadcast.Broadcasted) where {T,M} =
    (Ω.matrix = M(copy(_unwrap_state(broadcast))); Ω)

# Julia handles scalar assignment separately, so `Ω .= 0` needs these two
# additional methods. Both delegate to the `fill!` methods above.
@inline Base.copyto!(Ω::ConfigurationStates,
    broadcast::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}) =
    fill!(Ω, copy(_unwrap_state(broadcast)))
@inline Base.copyto!(Ω::ConfigurationStatesSubset,
    broadcast::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}) =
    fill!(Ω, copy(_unwrap_state(broadcast)))

# Matrix multiplication
# ---------------------
# Purpose: apply an RF rotation using standard in-place matrix syntax:
#
#   mul!(Ω, R, Ω) # equivalent to Ω = R * Ω
#
# Ω is both an input and the output here. Therefore the multiplication is
# completed before `copyto!` stores the result back in Ω.
@inline LinearAlgebra.mul!(Ω::AbstractConfigurationStates, A::AbstractMatrix,
    B::AbstractConfigurationStates) = copyto!(Ω, A * B.matrix)

"""
    F₊(Ω)

View into the first row of the configuration state matrix `Ω`, corresponding to the `F₊`
states.
"""
F₊(Ω) = OffsetVector(view(Ω, 1, :), 0:size(Ω, 2)-1)
"""
    F̄₋(Ω)

View into the second row of the configuration state matrix `Ω`, corresponding to the `F̄₋`
states.
"""
F̄₋(Ω) = OffsetVector(view(Ω, 2, :), 0:size(Ω, 2)-1)
"""
    Z(Ω)

View into the third row of the configuration state matrix `Ω`, corresponding to the `Z`
states.
"""
Z(Ω) = OffsetVector(view(Ω, 3, :), 0:size(Ω, 2)-1)

## KERNELS ###

# Initialize States

"""
    Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T}

By default, configuration states are complex. For some sequences, they will only ever be
real (no RF phase, no complex slice profile correction) and for these sequences a method
needs to be added to this function.

"""
@inline Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T}

"""
    initialize_states(::AbstractResource, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an `MMatrix` of EPG states on CPU to be used throughout the simulation.
"""
@inline function initialize_states(::AbstractResource, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    Ω = zeros(Ω_eltype(sequence), 3, Ns)

    return ConfigurationStates(Ω)
end

"""
    initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an array of EPG states on a CUDA GPU to be used throughout the simulation.
"""
@inline function initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    # # request shared memory in which configuration states are stored
    # # (all threads request for the entire threadblock)
    # Ω_shared = CUDA.CuStaticSharedArray(Ω_eltype(sequence), (3, Ns, THREADS_PER_BLOCK))
    # # get view for configuration states of this thread's voxel
    # # note that this function gets called inside a CUDA kernel
    # # so it has has access to threadIdx
    # Ω_view = view(Ω_shared, :, :, threadIdx().x)
    # # wrap in a ConfigurationStates object
    # Ω = MMatrix{3,Ns}(Ω_view) Ω = ConfigurationStates(Ω)

    # is Ns is not a multiple of 32, error
    if (Ns % WARPSIZE != 0)
        error("Number of states must be a multiple of THREADS_PER_BLOCK")
    end

    # Each thread holds (Ns ÷ WARPSIZE) columns of Ω
    #
    # Note this is stored as an *immutable* SMatrix, mutated by replacing the
    # `matrix` field of the (mutable) ConfigurationStatesSubset wrapper,
    # rather than as a mutable MMatrix mutated in place. MMatrix/MArray
    # implement setindex! via unsafe_store! through a pointer, which forces
    # the whole array into (very slow) GPU local memory. Once enough
    # operations in the per-TR loop touch it, the compiler ends up spilling
    # heavily -- this was the root cause of a >10x slowdown going from
    # max_state=32 to max_state=64. Reassigning an immutable SMatrix instead
    # keeps everything representable as plain SSA values, which the
    # compiler can keep in registers.
    num_states_per_thread = Ns ÷ WARPSIZE
    Ω = @SMatrix zeros(Ω_eltype(sequence), 3, num_states_per_thread)

    return ConfigurationStatesSubset(Ω)
end

#
"""
    initial_conditions!(Ω::AbstractConfigurationStates)

Set all components of all states to 0, except the Z-component of the 0th state which is set
to 1.
"""
@inline function initial_conditions!(Ω::AbstractConfigurationStates)
    fill!(Ω, zero(eltype(Ω)))
    @inbounds Z(Ω)[0] = 1
    return nothing
end

# GPU version: only lane 1 owns global state 0 -- each lane holds a
# different strided subset of the states, so unlike CPU, not every lane
# should touch Z[0] (see initialize_states(::CUDALibs, ...)).
@inline function initial_conditions!(Ω::ConfigurationStatesSubset)
    fill!(Ω, zero(eltype(Ω)))
    if laneid() == 1
        @inbounds Ω[3, 1] = 1
    end
    return nothing
end

# RF excitation

"""
    excite!(Ω::AbstractConfigurationStates, RF, p::AbstractTissueProperties)

Apply RF pulse rotation to the EPG states `Ω`.

# Arguments
- `Ω`: The configuration state matrix.
- `RF`: Complex RF pulse value. `abs(RF)` is the flip angle (degrees), `angle(RF)` is the
  pulse phase (radians). `B₁` scaling from `p` is applied internally if `hasB₁(p)`.
- `p`: Tissue properties (`AbstractTissueProperties`).
"""
@inline function excite!(
    Ω::AbstractConfigurationStates,
    RF::T,
    p::AbstractTissueProperties
) where {T<:Union{Complex,Quantity{<:Complex}}}

    α = flip_angle(RF, p)

    if iszero(α)
        return nothing
    end

    # apply rotation matrix to each state
    mul!(Ω, RF_rotation_matrix(RF, α), Ω)
    return nothing
end

"""
    flip_angle(RF, p::AbstractTissueProperties)

Flip angle in **radians** of an RF pulse, including `B₁` scaling if
`hasB₁(p)`. For a complex `RF` the nominal flip angle in degrees is
`abs(RF)` (its phase is handled separately); a real `RF` *is* the nominal
flip angle in degrees, signed.
"""
@inline function flip_angle(RF::T, p::AbstractTissueProperties) where {T<:Union{Complex,Quantity{<:Complex}}}
    α = deg2rad(abs(RF))
    hasB₁(p) && (α *= p.B₁)
    return α
end

@inline function flip_angle(RF::T, p::AbstractTissueProperties) where {T<:Union{Real,Quantity{<:Real}}}
    α = deg2rad(RF)
    hasB₁(p) && (α *= p.B₁)
    return α
end

"""
    RF_rotation_matrix(RF, α)

The 3x3 rotation matrix applied to each configuration state by an RF pulse
with flip angle `α` (radians) and phase `angle(RF)`. Split out of
[`excite!`](@ref) so that forward-sensitivity propagation can differentiate
it w.r.t. the flip angle (see `∂RF_rotation_matrix∂B₁` in
src/derivatives/forward_sensitivity.jl).
"""
@inline function RF_rotation_matrix(RF::T, α) where {T<:Union{Complex,Quantity{<:Complex}}}
    x = α / 2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2 * sinx * cosx, 2 * cos²x - one(α)
    # phase stuff
    cosφ, sinφ = reim(normalize(RF))
    # again double angle formula
    sin2φ, cos2φ = 2 * sinφ * cosφ, 2 * cosφ^2 - one(α)
    # complex exponentials
    ℯⁱᵠ = complex(cosφ, sinφ)
    ℯ²ⁱᵠ = complex(cos2φ, sin2φ)
    ℯ⁻ⁱᵠ = conj(ℯⁱᵠ)
    ℯ⁻²ⁱᵠ = conj(ℯ²ⁱᵠ)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, ℯ²ⁱᵠ * sin²x, -im * ℯⁱᵠ * sinα
    R₂₁, R₂₂, R₂₃ = ℯ⁻²ⁱᵠ * sin²x, cos²x, 1im * ℯ⁻ⁱᵠ * sinα #im gives issues with CUDA profiling, 1im works
    R₃₁, R₃₂, R₃₃ = -im * ℯ⁻ⁱᵠ * sinα / 2, 1im * ℯⁱᵠ * sinα / 2, cosα
    # assemble static matrix
    return SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃)
end

"""
    excite!(Ω::AbstractConfigurationStates, RF, p::AbstractTissueProperties) where {T<:Union{Real, Quantity{<:Real}}}

Apply RF pulse rotation to the EPG states `Ω` (version for real-valued RF pulse, assuming
zero phase).

# Arguments
- `Ω`: The configuration state matrix.
- `RF`: Real RF pulse value representing the flip angle (degrees). `B₁` scaling from `p` is
  applied internally if `hasB₁(p)`.
- `p`: Tissue properties (`AbstractTissueProperties`).
"""
@inline function excite!(
    Ω::AbstractConfigurationStates,
    RF::T,
    p::AbstractTissueProperties
) where {T<:Union{Real,Quantity{<:Real}}}

    α = flip_angle(RF, p)

    if iszero(α)
        return nothing
    end

    # apply rotation matrix to each state
    mul!(Ω, RF_rotation_matrix(RF, α), Ω)

    return nothing
end

"""
    RF_rotation_matrix(RF::Real, α)

Version for a real-valued RF pulse (zero phase). See the complex version above.
"""
@inline function RF_rotation_matrix(RF::T, α) where {T<:Union{Real,Quantity{<:Real}}}
    x = α / 2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2 * sinx * cosx, 2 * cos²x - one(α)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, -sin²x, -sinα
    R₂₁, R₂₂, R₂₃ = -sin²x, cos²x, -sinα
    R₃₁, R₃₂, R₃₃ = sinα / 2, sinα / 2, cosα
    # assemble static matrix
    return SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃)
end

"""
    rotate!(Ω::AbstractConfigurationStates, eⁱᶿ)

Apply phase accrual due to off-resonance to the transverse EPG states (`F₊`, `F̄₋`).

# Arguments
- `Ω`: The configuration state matrix.
- `eⁱᶿ`: Complex rotation factor, typically `exp(-im * Δω * Δt)`, where `Δω` is the
  off-resonance frequency (rad/s, potentially derived from `p.B₀`) and `Δt` is the time
  duration (seconds).
"""
@inline function rotate!(Ω::AbstractConfigurationStates, eⁱᶿ::T) where {T}
    Ω .*= (eⁱᶿ, conj(eⁱᶿ), one(eⁱᶿ))
    return nothing
end

# Decay and diffuse

"""
    decay!(Ω::AbstractConfigurationStates, E₁, E₂)

Apply T₁ and T₂ relaxation effects to the EPG states `Ω`.

# Arguments
- `Ω`: The configuration state matrix.
- `E₁`: T₁ relaxation factor, `exp(-Δt/T₁)`, where `Δt` is the time duration (seconds) and
  `T₁` is from the tissue properties (seconds).
- `E₂`: T₂ relaxation factor, `exp(-Δt/T₂)`, where `Δt` is the time duration (seconds) and
  `T₂` is from the tissue properties (seconds).
"""
@inline function decay!(Ω::AbstractConfigurationStates, E₁, E₂)
    Ω .*= (E₂, E₂, E₁)
    return nothing
end

"""
    rotate_decay!(Ω::AbstractConfigurationStates, E₁, E₂, eⁱᶿ)

Apply combined off-resonance rotation and T₁/T₂ relaxation to the EPG states `Ω`.

# Arguments
- `Ω`: The configuration state matrix.
- `E₁`: T₁ relaxation factor (`exp(-Δt/T₁)`).
- `E₂`: T₂ relaxation factor (`exp(-Δt/T₂)`).
- `eⁱᶿ`: Complex off-resonance rotation factor (`exp(-im * Δω * Δt)`). (See `rotate!` and
`decay!` for details on arguments).
"""
@inline function rotate_decay!(Ω::AbstractConfigurationStates, E₁, E₂, eⁱᶿ)
    Ω .*= (E₂ * eⁱᶿ, E₂ * conj(eⁱᶿ), complex(E₁))
    return nothing
end


"""
    diffusion_decay_matrix(Ω::EPGSAbstractConfigurationStatestates, D)

Pre-calculate diffusion decay according to state number. Store in a matrix with the same
size/type as Ω s.t. later on the decay can be applied by element-wise multiplication.

# Arguments
- `Ω`: The configuration state matrix
- `D`: The diffusion coefficient
"""
@inline function diffusion_decay_matrix(Ω::AbstractConfigurationStates, D::T) where {T<:Real}
    # println("hi")
    expbD = similar(real(Ω.matrix))

    # expbD = @MMatrix zeros(real(eltype(Ω.matrix)), size(3, num_states_per_thread)

    # return ConfigurationStatesSubset(Ω)
    for state in 0:size(Ω, 2)-1
        bᵀD = T(((state + 0.5)^2 + 1.0 / 12.0) * D)
        bᴸD = T((state^2) * D)
        expbᵀD = exp(-bᵀD)
        expbᴸD = exp(-bᴸD)
        F₊(expbD)[state] = expbᵀD
        F̄₋(expbD)[state] = expbᵀD
        Z(expbD)[state] = expbᴸD
    end
    return ConfigurationStates(expbD)
end

"""
On GPU, when each thread only holds parts of the configuration state matrix, each thread
only computes the corresponding columns of the diffusion decay matrix.
"""
@inline function diffusion_decay_matrix(Ω::ConfigurationStatesSubset, D::T) where {T<:Real}
    expbD = similar(real(Ω.matrix))
    for idx in 1:size(Ω, 2)
        # Calculate the state this thread is responsible for
        state = laneid() - 1 + (idx - 1) * WARPSIZE
        bᵀD = T(((state + 0.5)^2 + 1.0 / 12.0) * D)
        bᴸD = T((state^2) * D)
        expbᵀD = exp(-bᵀD)
        expbᴸD = exp(-bᴸD)
        expbD[1, idx] = expbᵀD
        expbD[2, idx] = expbᵀD
        expbD[3, idx] = expbᴸD
    end
    return ConfigurationStatesSubset(SMatrix(expbD))
end
"""
diffuse!(Ω::AbstractConfigurationStatestates, diffusion_decay)

Apply diffusion decay according to state number by element-wise multiplication with
the pre-computed diffusion decay matrix.

- `Ω`: The configuration state matrix
- `diffusion_decay`: The pre-calculated diffusion decay terms
"""
@inline function diffuse!(Ω::AbstractConfigurationStates, diffusion_decay)
    Ω .*= diffusion_decay.matrix
    return nothing
end

# Regrowth

"""
    regrowth!(Ω::AbstractConfigurationStates, E₁)

Apply T₁ regrowth to the longitudinal magnetization equilibrium state (`Z₀`).

# Arguments
- `Ω`: The configuration state matrix.
- `E₁`: T₁ relaxation factor, `exp(-Δt/T₁)`, where `Δt` is the time duration (seconds) and
  `T₁` is from the tissue properties (seconds). The regrowth amount is `(1 - E₁)`.
"""
@inline function regrowth!(Ω::AbstractConfigurationStates, E₁)
    @inbounds Z(Ω)[0] += (1 - E₁)
end

# GPU version: same update as above, guarded to only lane 1 -- see the
# comment on initial_conditions!(Ω::ConfigurationStatesSubset) for why.
@inline function regrowth!(Ω::ConfigurationStatesSubset, E₁)
    if laneid() == 1
        @inbounds Ω[3, 1] += (1 - E₁)
    end
    return nothing
end

# Dephasing

"""
    dephasing!(Ω::AbstractConfigurationStates)

Shift states around due to dephasing gradient: The `F₊` go up one, the `F̄₋` go down one and
`Z` do not change
"""
@inline function dephasing!(Ω::AbstractConfigurationStates)
    shift_down!(F̄₋(Ω))
    shift_up!(F₊(Ω), F̄₋(Ω))
end

# shift down the F- states, set highest state to 0
@inline function shift_down!(F̄₋)
    for i = 0:lastindex(F̄₋)-1
        @inbounds F̄₋[i] = F̄₋[i+1]
    end
    @inbounds F̄₋[end] = 0
end

# shift up the F₊ states and let F₊[0] be conj(F₋[0])
@inline function shift_up!(F₊, F̄₋)
    for i = lastindex(F₊):-1:1
        @inbounds F₊[i] = F₊[i-1]
    end
    @inbounds F₊[0] = conj(F̄₋[0])
end

# GPU dephasing works directly on Ω. Its specialized setindex! above turns
# every apparent element mutation into a functional SMatrix update.
@inline function dephasing!(Ω::ConfigurationStatesSubset)
    shuffle_down!(Ω)
    shuffle_up!(Ω)
    return nothing
end

# shuffle down the F- states, set highest state to 0
@inline function shuffle_down!(Ω::ConfigurationStatesSubset)
    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() + 1, WARPSIZE)
    for column in axes(Ω, 2)
        @inbounds F̄₋ᵏ = Ω[2, column]
        if laneid() == Int32(1) && column < size(Ω, 2)
            @inbounds F̄₋ᵏ = Ω[2, column+1]
        end
        F̄₋ᵏ = CUDA.shfl_sync(mask, F̄₋ᵏ, src_lane)  # Broadcast value from the first lane
        @inbounds Ω[2, column] = F̄₋ᵏ
    end
    if laneid() == WARPSIZE
        @inbounds Ω[2, end] = 0
    end
    return nothing
end

# shuffle up the F₊ states and let F₊[0] be conj(F₋[0])
@inline function shuffle_up!(Ω::ConfigurationStatesSubset)
    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() - Int32(1), WARPSIZE)
    last_column = Int32(size(Ω, 2))
    for column in last_column:Int32(-1):Int32(1)
        @inbounds F₊ᵏ = Ω[1, column]
        if laneid() == WARPSIZE && column > Int32(1)
            @inbounds F₊ᵏ = Ω[1, column-1]
        end
        @inbounds Ω[1, column] = CUDA.shfl_sync(mask, F₊ᵏ, src_lane)
    end
    if laneid() == Int32(1)
        @inbounds Ω[1, 1] = conj(Ω[2, 1])
    end
    return nothing
end

# Invert

"""
    invert!(Ω::AbstractConfigurationStates, p::AbstractTissueProperties)

Invert `Z`-component of states of all orders. *Assumes fully spoiled transverse
magnetization*.
"""
@inline function invert!(Ω::AbstractConfigurationStates, p::AbstractTissueProperties)
    # inversion angle
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    cosθ = cos(θ)
    Ω .*= (one(cosθ), one(cosθ), cosθ)
end

"""
    invert!(Ω::AbstractConfigurationStates)

Invert with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline function invert!(Ω::AbstractConfigurationStates)
    Ω .*= (1, 1, -1)
end

# Spoil

"""
    spoil!(Ω::AbstractConfigurationStates)

Perfectly spoil the transverse components of all states.
"""
# Shared by CPU and GPU; see the comment on invert!(Ω, p) above.
@inline function spoil!(Ω::AbstractConfigurationStates)
    Ω .*= (0, 0, 1)
end

# Sample

"""
    sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::AbstractConfigurationStates)

Sample the measurable transverse magnetization, that is, the `F₊` component of the 0th
state. The `+=` is needed for 2D sequences where slice profile is taken into account.
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::AbstractConfigurationStates)
    @inbounds output[index] += F₊(Ω)[0]
end

@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::ConfigurationStatesSubset)
    if laneid() == 1
        @inbounds output[index] += Ω[1, 1]
    end
end

"""
    sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::AbstractConfigurationStates)

Sample the entire configuration state matrix `Ω`. The `+=` is needed for 2D sequences where
slice profile is taken into account.
"""
@inline function sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::AbstractConfigurationStates)
    @inbounds output[index] .+= Ω
end

@inline function sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::ConfigurationStatesSubset)
    if laneid() == 1
        @inbounds output[index] .+= Ω
    end
end
