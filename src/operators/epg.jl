abstract type AbstractConfigurationStates{T} <: AbstractMatrix{T} end
struct ConfigurationStates{T,M<:AbstractMatrix{T}} <: AbstractConfigurationStates{T}
    matrix::M
end

ConfigurationStates(m::Matrix) = ConfigurationStates(MMatrix{size(m)...}(m))

struct ConfigurationStatesSubset{T,M<:AbstractMatrix{T}} <: AbstractConfigurationStates{T}
    matrix::M
end

# Make the AbstractConfigurationStates satisfy the AbstractMatrix interface
Base.size(Ω::AbstractConfigurationStates) = size(Ω.matrix)
Base.getindex(Ω::AbstractConfigurationStates, i::Int) = Ω.matrix[i]
Base.getindex(Ω::AbstractConfigurationStates, I::Vararg{Int,N}) where {N} = Ω.matrix[I...]
Base.setindex!(Ω::AbstractConfigurationStates, v, i::Int) = setindex!(Ω.matrix, v, i)
Base.setindex!(Ω::AbstractConfigurationStates, v, I::Vararg{Int,N}) where {N} = setindex!(Ω.matrix, v, I...)
Base.view(Ω::AbstractConfigurationStates, inds...) = view(Ω.matrix, inds...)

# # The three methods below are used to make it possible to write Ω .= R * Ω instead of Ω.matrix .= R * Ω.matrix

# # Overload * operator
# function Base.:*(R::AbstractMatrix, Ω::AbstractConfigurationStates) # Assuming simple
#     matrix multiplication for demonstration return typeof(Ω)(R * Ω.matrix) end

# Overload broadcasted assignment for CustomMatrixType
function Base.broadcasted(::typeof(identity), Ω::AbstractConfigurationStates)
    # This allows .= to work by returning the object itself in a broadcasted context
    return Ω
end

function Base.copyto!(Ω::AbstractConfigurationStates, result::AbstractConfigurationStates)
    # Implement how the result of R * Ω should be stored back into Ω
    Ω.matrix .= result.matrix
    return Ω
end


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

    Ω = @MMatrix zeros(Ω_eltype(sequence), 3, Ns ÷ WARPSIZE)

    return ConfigurationStatesSubset(Ω)
end

#
"""
    initial_conditions!(Ω::AbstractConfigurationStates)

Set all components of all states to 0, except the Z-component of the 0th state which is set
to 1.
"""
@inline function initial_conditions!(Ω::AbstractConfigurationStates)
    @. Ω = 0
    @inbounds Z(Ω)[0] = 1
    return nothing
end

@inline function initial_conditions!(Ω::ConfigurationStatesSubset)
    @. Ω = 0
    if laneid() == 1
        @inbounds Z(Ω)[0] = 1
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

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(abs(RF))
    hasB₁(p) && (α *= p.B₁)

    if iszero(α)
        return nothing
    end

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
    R = SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃)
    # apply rotation matrix to each state
    Ω.matrix .= R * Ω.matrix
    return nothing
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

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(RF)
    hasB₁(p) && (α *= p.B₁)

    if iszero(α)
        return nothing
    end

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
    R = SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃)
    # apply rotation matrix to each state
    Ω.matrix .= R * Ω.matrix

    return nothing
end

"""
    rotate!(Ω::AbstractConfigurationStates, eⁱᶿ)

Apply phase accrual due to off-resonance to the transverse EPG states (`F₊`, `F̄₋`).

# Arguments
- `Ω`: The configuration state matrix.
- `eⁱᶿ`: Complex rotation factor, typically `exp(im * Δω * Δt)`, where `Δω` is the
  off-resonance frequency (rad/s, potentially derived from `p.B₀`) and `Δt` is the time
  duration (seconds).
"""
@inline function rotate!(Ω::AbstractConfigurationStates, eⁱᶿ::T) where {T}
    @. Ω.matrix[1:2, :] *= (eⁱᶿ, conj(eⁱᶿ))
end

# Decay

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
    @. Ω.matrix *= (E₂, E₂, E₁)
end

"""
    rotate_decay!(Ω::AbstractConfigurationStates, E₁, E₂, eⁱᶿ)

Apply combined off-resonance rotation and T₁/T₂ relaxation to the EPG states `Ω`.

# Arguments
- `Ω`: The configuration state matrix.
- `E₁`: T₁ relaxation factor (`exp(-Δt/T₁)`).
- `E₂`: T₂ relaxation factor (`exp(-Δt/T₂)`).
- `eⁱᶿ`: Complex off-resonance rotation factor (`exp(im * Δω * Δt)`). (See `rotate!` and
`decay!` for details on arguments).
"""
@inline function rotate_decay!(Ω::AbstractConfigurationStates, E₁, E₂, eⁱᶿ)
    @. Ω.matrix *= (E₂ * eⁱᶿ, E₂ * conj(eⁱᶿ), complex(E₁))
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

@inline function regrowth!(Ω::ConfigurationStatesSubset, E₁)
    if laneid() == 1
        @inbounds Z(Ω)[0] += (1 - E₁)
    end
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

@inline function dephasing!(Ω::ConfigurationStatesSubset)
    shuffle_down!(F̄₋(Ω))
    shuffle_up!(F₊(Ω), F̄₋(Ω))
end

# shuffle down the F- states, set highest state to 0
@inline function shuffle_down!(F̄₋)

    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() + 1, WARPSIZE)
    for k in eachindex(F̄₋)
        @inbounds F̄₋ᵏ = F̄₋[k]

        if laneid() == Int32(1)
            if k < lastindex(F̄₋)
                @inbounds F̄₋ᵏ = F̄₋[k+1]
            end
        end
        F̄₋ᵏ = CUDA.shfl_sync(mask, F̄₋ᵏ, src_lane)  # Broadcast value from the first lane
        @inbounds F̄₋[k] = F̄₋ᵏ

    end

    if laneid() == WARPSIZE
        @inbounds F̄₋[end] = 0
    end

    return nothing
end

# shuffle up the F₊ states and let F₊[0] be conj(F₋[0])
@inline function shuffle_up!(F₊, F̄₋)

    # Mask that excludes the last thread in each warp
    mask = CUDA.FULL_MASK
    src_lane = mod1(laneid() - Int32(1), WARPSIZE)

    # Shuffle down the F- values
    kk = Int32(lastindex(F₊))
    for k in kk:Int32(-1):Int32(0)
        @inbounds F₊ᵏ = F₊[k]

        if laneid() == WARPSIZE
            if k > Int32(0)
                @inbounds F₊ᵏ = F₊[k-1]
            end
        end

        @inbounds F₊[k] = CUDA.shfl_sync(mask, F₊ᵏ, src_lane)
    end
    if laneid() == Int32(1)
        @inbounds F₊[begin] = conj(F̄₋[begin])
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
    Z(Ω) .*= cos(θ)
end

"""
    invert!(Ω::AbstractConfigurationStates)

Invert with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline function invert!(Ω::AbstractConfigurationStates)
    Z(Ω) .*= -1
end

# Spoil

"""
    spoil!(Ω::AbstractConfigurationStates)

Perfectly spoil the transverse components of all states.
"""
@inline function spoil!(Ω::AbstractConfigurationStates)
    F₊(Ω) .= 0
    F̄₋(Ω) .= 0
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
        @inbounds output[index] += F₊(Ω)[0]
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
