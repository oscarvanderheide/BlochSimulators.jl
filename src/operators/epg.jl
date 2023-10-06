"""
    EPGStates = Union{MMatrix{3}, SizedMatrix{3}}

In the EPG model, the configuration state matrix `Ω` will be updated inplace.
On CPU, we use `StaticArrays.MMatrix`. The MMatrix will not escape `_simulate!`
and therefore should not result in allocations.
On GPU, we use shared memory (`CUDA.CuStaticSharedArray`). The shared memory is allocated
for all threads within a block simultaneously. We then take a `@view` and wrap it in a `SizedArray`.
Without the `SizedArray`, the type would be long and unreadable. By wrapping, we can then
simply dispatch on a `SizedArray` instead.
"""
const EPGStates = Union{MMatrix{3}, SizedMatrix{3}}

"""
    F₊(Ω)

View into the first row of the configuration state matrix `Ω`,
corresponding to the `F₊` states.
"""
F₊(Ω) = OffsetVector(view(Ω,1,:), 0:size(Ω,2)-1)
"""
    F̄₋(Ω)

View into the second row of the configuration state matrix `Ω`,
corresponding to the `F̄₋` states.
"""
F̄₋(Ω) = OffsetVector(view(Ω,2,:), 0:size(Ω,2)-1)
"""
    Z(Ω)

View into the third row of the configuration state matrix `Ω`,
corresponding to the `Z` states.
"""
Z(Ω) = OffsetVector(view(Ω,3,:), 0:size(Ω,2)-1)

## KERNELS ###

# Initialize States

"""
    Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T}

By default, configuration states are complex. For some sequences, they
will only ever be real (no RF phase, no complex slice profile correction)
and for these sequences a method needs to be added to this function.

"""
@inline Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T}


"""
    initialize_states(::AbstractResource, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an `MMatrix` of EPG states on CPU to be used throughout the simulation.
"""
@inline function initialize_states(::AbstractResource, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    Ω = @MMatrix zeros(Ω_eltype(sequence),3,Ns)
end

"""
    initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an array of EPG states on a CUDA GPU to be used throughout the simulation.
"""
@inline function initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    # request shared memory in which configuration states are stored
    # (all threads request for the entire threadblock)
    Ω_shared = CUDA.CuStaticSharedArray(Ω_eltype(sequence), (3, Ns, THREADS_PER_BLOCK))
    # get view for configuration states of this thread's voxel
    # note that this function gets called inside a CUDA kernel
    # so it has has access to threadIdx
    Ω_view = view(Ω_shared,:,:,threadIdx().x)
    # wrap in a SizedMatrix
    Ω = SizedMatrix{3,Ns}(Ω_view)
    return Ω
end

# Initial conditions

"""
    initial_conditions!(Ω::EPGStates)

Set all components of all states to 0, except the Z-component of the 0th state which is set to 1.
"""
@inline function initial_conditions!(Ω::EPGStates)
    @. Ω = 0
    Z(Ω)[0] = 1
    return nothing
end

# RF excitation

"""
    excite!(Ω::EPGStates, RF::Complex, p::AbstractTissueParameters)

Mixing of states due to RF pulse. Magnitude of RF is the flip angle in degrees.
Phase of RF is the phase of the pulse. If RF is real, the computations simplify a little bit.
"""
@inline function excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Complex, Quantity{<:Complex}}

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(abs(RF))
    hasB₁(p) && (α *= p.B₁)
    # phase of RF pulse
    φ = angle(RF)

    x = α/2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2*sinx*cosx, 2*cos²x - one(α)
    # phase stuff
    sinφ, cosφ = sincos(φ)
    # again double angle formula
    sin2φ, cos2φ = 2*sinφ*cosφ, 2*cosφ^2 - one(α)
    # complex exponentials
    ℯⁱᵠ  = complex(cosφ,  sinφ)
    ℯ²ⁱᵠ = complex(cos2φ, sin2φ)
    ℯ⁻ⁱᵠ  = conj(ℯⁱᵠ)
    ℯ⁻²ⁱᵠ = conj(ℯ²ⁱᵠ)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, ℯ²ⁱᵠ * sin²x, -im * ℯⁱᵠ * sinα
    R₂₁, R₂₂, R₂₃ = ℯ⁻²ⁱᵠ * sin²x, cos²x, 1im * ℯ⁻ⁱᵠ * sinα #im gives issues with CUDA profiling, 1im works
    R₃₁, R₃₂, R₃₃ = -im * ℯ⁻ⁱᵠ * sinα / 2, 1im * ℯⁱᵠ * sinα / 2, cosα
    # assemble static matrix
    R = SMatrix{3,3}(R₁₁,R₂₁,R₃₁,R₁₂,R₂₂,R₃₂,R₁₃,R₂₃,R₃₃)
    # apply rotation matrix to each state
    Ω .= R * Ω
    return nothing
end
"""
    excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Real, Quantity{<:Real}}

If RF is real, the calculations simplify (and probably Ω is real too, reducing memory (access) requirements).
"""
@inline function excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Real, Quantity{<:Real}}

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(RF)
    hasB₁(p) && (α *= p.B₁)
    x = α/2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2*sinx*cosx, 2*cos²x - one(α)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, -sin²x, -sinα
    R₂₁, R₂₂, R₂₃ = -sin²x, cos²x, -sinα
    R₃₁, R₃₂, R₃₃ = sinα / 2, sinα / 2, cosα
    # assemble static matrix
    R = SMatrix{3,3}(R₁₁,R₂₁,R₃₁,R₁₂,R₂₂,R₃₂,R₁₃,R₂₃,R₃₃)
    # apply rotation matrix to each state
    Ω .= R * Ω

    return nothing
end

"""
    rotate!(Ω::EPGStates, eⁱᶿ::T) where T

Rotate `F₊` and `F̄₋` states under the influence of `eⁱᶿ = exp(i * ΔB₀ * Δt)`
"""
@inline function rotate!(Ω::EPGStates, eⁱᶿ::T) where T
    @. Ω[1:2,:] *= (eⁱᶿ,conj(eⁱᶿ))
end

# Decay

"""
    decay!(Ω::EPGStates, E₁, E₂)

T₂ decay for F-components, T₁ decay for `Z`-component of each state.
"""
@inline function decay!(Ω::EPGStates, E₁, E₂)
    @. Ω *= (E₂, E₂, E₁)
end

"""
    rotate_decay!(Ω::EPGStates, E₁, E₂, eⁱᶿ)

Rotate and decay combined
"""
@inline function rotate_decay!(Ω::EPGStates, E₁, E₂, eⁱᶿ)
    @. Ω *= (E₂*eⁱᶿ, E₂*conj(eⁱᶿ), complex(E₁))
end

# Regrowth

"""
    regrowth!(Ω::EPGStates, E₁)

T₁ regrowth for Z-component of 0th order state.
"""
@inline function regrowth!(Ω::EPGStates, E₁)
    Z(Ω)[0] += (1 - E₁)
end

# Dephasing

"""
    dephasing!(Ω::EPGStates)

Shift states around due to dephasing gradient:
The `F₊` go up one, the `F̄₋` go down one and `Z` do not change
"""
@inline function dephasing!(Ω::EPGStates)
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

# Invert

"""
    invert!(Ω::EPGStates, p::AbstractTissueParameters)

Invert `Z`-component of states of all orders. *Assumes fully spoiled transverse magnetization*.
"""
@inline function invert!(Ω::EPGStates, p::AbstractTissueParameters)
    # inversion angle
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    Z(Ω) .*= cos(θ)
end

"""
    invert!(Ω::EPGStates)

Invert with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline function invert!(Ω::EPGStates)
    Z(Ω) .*= -1
end

# Spoil

"""
    spoil!(Ω::EPGStates)

Perfectly spoil the transverse components of all states.
"""
@inline function spoil!(Ω::EPGStates)
    F₊(Ω) .= 0
    F̄₋(Ω) .= 0
end

# Sample

"""
    sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)

Sample the measurable transverse magnetization, that is, the `F₊` component of the 0th state.
The `+=` is needed for 2D sequences where slice profile is taken into account.
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] += F₊(Ω)[0]
end

"""
    sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)

Sample the entire configuration state matrix `Ω`. The `+=` is needed
for 2D sequences where slice profile is taken into account.
"""
@inline function sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] .+= Ω
end