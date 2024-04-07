### Type definition

"""
    SpokesTrajectory{T} <: AbstractTrajectory{T}

Typical Cartesian and radial trajectories have a lot in common: a readout
can be described by a starting point in k-space and a Δk per sample point.
To avoid code repetition, both type of trajectories are made a subtype of
SpokesTrajectory such that some methods that would be the same for both
trajectories otherwise are written for SpokesTrajectory instead.
"""
abstract type SpokesTrajectory{T} <: AbstractTrajectory{T} end # Cartesian & Radial

export SpokesTrajectory

"""
    CartesianTrajectory{T,I,U,V} <: SpokesTrajectory{T}

Struct that is used to implement a typical Cartesian gradient trajectory.
The trajectory is described in a compact fashion by only storing
the starting position in k-space (`k_start_readout`) for each readout
as well as the step in k-space per readout point `Δk_adc`.

Note that CartesianTrajectory and RadialTrajectory are essentially the same in when
using when using this compact description. A SpokesTrajectory struct is therefore
defined as a supertype of both and methods are defined for SpokesTrajectory instead
to avoid code repetition.

The type parameters are intentionally left vague. The `J`, for example, may be an
integer for sequences where each readout has the same number of samples, but for
sequences with different numbers of samples per readout it may be a vector of integers.

# Fields
- `nreadouts::I`: The total number of readouts for this trajectory
- `nsamplesperreadout::I`: The total number of samples per readout
- `Δt::T`: Time between sample points
- `k_start_readout::U`: Starting position in k-space for each readout
- `Δk_adc::U`: k-space step Δkₓ per sample point (same for all readouts)
- `py::V`: Phase encoding index for each readout
- `readout_oversampling::I`: Readout oversampling factor
"""
struct CartesianTrajectory{T<:Real,I<:Integer,U<:AbstractVector,V<:AbstractVector} <: SpokesTrajectory{T}
    nreadouts::I
    nsamplesperreadout::I
    Δt::T # time between sample points
    k_start_readout::U # starting position in k-space for each readout
    Δk_adc::T # Δk during ADC for each readout
    py::V # phase encoding order, not needed but nice to store
    readout_oversampling::I # readout oversampling factor
end

@functor CartesianTrajectory
@adapt_structure CartesianTrajectory

export CartesianTrajectory

"""
    magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

Arguments
- `magnetization`:          Matrix{Complex} of size (# readouts, # voxels) with phase-encoded
                    magnetization at echo times.
- `parameters`:     Tissue parameters of all voxels, including spatial coordinates.
- `trajectory`:     Cartesian trajectory struct.
- `coordinates`:    Vector{Coordinates} with spatial coordinates for each voxel.
- `parameters`:     Matrix{Complex} of size (# voxels, # coils) with coil sensitivities.

Output:
- `signal`: Vector of length (# coils) with each element a Matrix{Complex}
        of size (# readouts, # samples per readout)

Description:

As noted in the description of the simulate_signal function (see `src/simulate/signal.jl`),
we simulate the MR signal at timepoint `t` from coil `i` as:
signalᵢ[t] = sum(m[t,v] * cᵢ[v] * ρ[v]  for v in 1:(# voxels)),
where `cᵢ`is the coil sensitivity profile of coil `i`, `ρ` is the proton density
map and `m` the matrix with the magnetization at all timepoints for each voxel
obtained through Bloch simulations.

The output (signalᵢ) for each coil is in principle a `Vector{Complex}`` of
length (# samples per readout) * (# readouts). If we reshape the output into a
`Matrix{Complex}` of size (# samples per readout, # readouts) instead, and do
something similar for `m`, then the signal value associated with the s-th sample
point of the r-th readout can be expressed as
signalᵢ[r,s] = sum( m[r,s,v]] * cᵢ[v] * ρ[v]  for v in 1:(# voxels)).

The problem here is that we typically cannot store the full m. Instead, we compute the
magnetization at echo times only. The reason is that, if mᵣ is the magnetization at
the r-th echo time in some voxel, and E = exp(-Δt*R₂[v]) * exp(im*(Δkₓ*x[v]))
is the change per sample point (WHICH FOR CARTESIAN SEQUENCES IS THE SAME
FOR ALL READOUTS AND SAMPLES), then the magnetization at the s-th sample
relative the the echo time can can be computed as mₛ = mᵣ * E[v]^s

Therefore we can write

signalⱼ[r,s] = sum( echos[r,v] * E[v]^s * ρ[v] * cⱼ[v] for v in 1:(# voxels))
signalⱼ[r,s] = echos[r,:] * (E.^s .* ρ .* cⱼ)

Because the (E.^s .* ρ .* cⱼ)-part is the same for all readouts, we can simply
perform this computation for all readouts simultaneously as
signalⱼ[:,s] = echos * (E.^s .* ρ .* cⱼ)

If we define the matrix Eˢ as E .^ (-(ns÷2):(ns÷2)-1), then we can do the computation
for all different sample points at the same time as well using a single matrix-matrix
multiplication:
signalⱼ = echos * (Eˢ .* (ρ .* cⱼ))

For the final output, we need to do this for each coil j.

Note that this implementation relies entirely on vectorized code
and works on both CPU and GPU. The matrix-matrix multiplications
are - I think - already multi-threaded so a separate multi-threaded
implementation is not needed.
"""
function echos_to_signal(::Union{CPU1,CPUThreads,CUDALibs}, echos, parameters, trajectory::CartesianTrajectory, coil_sensitivities)

    # Sanity checks
    @assert size(echos) == (trajectory.nreadouts, length(parameters))
    @assert size(coil_sensitivities,1) == length(parameters)

    # Load constants
    # Maybe should start using StructArrays to get rid of the map stuff
    T₂  = map(p->p.T₂, parameters) |> vec
    ρ  = map(p->complex(p.ρˣ,p.ρʸ), parameters) |> vec
    x  = map(p->p.x, parameters) |> vec
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    ns = trajectory.nsamplesperreadout
    nr = trajectory.nreadouts

    # Dynamics during readout (gradient encoding, T2 decay, B0 rotation)
    # are different per voxel but the same for each readout and sample point
    # Pre-compute vector with dynamic change from one sample point to the next
    E = @. exp(-Δt*inv(T₂)) * exp(im*(Δkₓ*x)) # todo: B0
    # To compute signal at all sample points with one matrix-matrix multiplication,
    # we pre-compute a readout_dynamics matrix instead
    Eˢ = @. E^(-(ns÷2):(ns÷2)-1)'
    # Perform main computation
    signal = map(eachcol(coil_sensitivities)) do c
        magnetization * (Eˢ .* (ρ .* c))
    end

    # Final checks
    @assert length(signal) == size(coil_sensitivities,2)
    @assert size(signal[1]) == (nr,ns)

    return signal
end


### Interface requirements

@inline nreadouts(t::SpokesTrajectory) = t.nreadouts
@inline nsamplesperreadout(t::SpokesTrajectory) = t.nsamplesperreadout
@inline nsamplesperreadout(t::SpokesTrajectory, readout) = t.nsamplesperreadout

function phase_encoding!(magnetization, trajectory::CartesianTrajectory, coordinates)
    y  = last.(coordinates) |> vec
    kʸ = imag.(trajectory.k_start_readout)
    @. magnetization *= exp(im * kʸ * y')
    return nothing
end

# perhaps do this with metaprogramming instead (iteratate over all subtypes of AbstractTrajectory)
function phase_encoding!(magnetization::DArray, trajectory::CartesianTrajectory, coordinates::DArray)

    @sync for p in workers()
        @async begin
            @spawnat p phase_encoding!(localpart(magnetization), trajectory, localpart(coordinates))
        end
    end

    return nothing
end

@inline function to_sample_point(mₑ, trajectory::CartesianTrajectory, readout_idx, sample_idx, p)

    # Note that m has already been phase-encoded

    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    x = p.x
    # There are ns samples per readout, echo time is assumed to occur
    # at index (ns÷2)+1. Now compute sample index relative to the echo time
    s = sample_idx - ((ns ÷ 2) + 1)
    # Apply readout gradient, T₂ decay and B₀ rotation
    E₂ = exp(-Δt * s * R₂)
    θ = Δkₓ * x
    hasB₀(p) && (θ += π * p.B₀ * Δt * 2)
    E₂eⁱᶿ = E₂ * exp(im * s * θ)
    mₛ = E₂eⁱᶿ * mₑ

    return mₛ
end

### Utility functions (not used in signal simulations)

# Convenience constructor to quickly generate Cartesian trajectory
# with nr readouts and ns samples per readout
CartesianTrajectory(nr, ns) = CartesianTrajectory(nr, ns, 10^-5, complex.(ones(nr)), 1.0, (1:nr) .- (nr ÷ 2), 2)

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::CartesianTrajectory, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.k_start_readout[idx], tr.Δk_adc, tr.py[idx], tr.readout_oversampling)

# Nicer printing in REPL
Base.show(io::IO, tr::CartesianTrajectory) = begin
    println("")
    println(io, "Cartesian trajectory")
    println(io, "nreadouts:            ", tr.nreadouts)
    println(io, "nsamplesperreadout:   ", tr.nsamplesperreadout)
    println(io, "Δt:                   ", tr.Δt)
    println(io, "k_start_readout:      ", typeof(tr.k_start_readout))
    println(io, "Δk_adc:               ", tr.Δk_adc)
    println(io, "py:                   ", typeof(tr.py))
    println(io, "readout_oversampling: ", tr.readout_oversampling)
end

"""
    kspace_coordinates(tr::CartesianTrajectory)

Return matrix (nrsamplesperreadout, nrreadouts) with kspace coordinates for the trajectory. Needed for nuFFT reconstructions.
"""
function kspace_coordinates(tr::CartesianTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout
    k = [tr.k_start_readout[r] + (s - 1) * tr.Δk_adc for s in 1:ns, r in 1:nr]

    return k
end

"""
    sampling_mask(tr::CartesianTrajectory)

For undersampled Cartesian trajectories, the gradient trajectory can also be described by a sampling mask.
"""
function sampling_mask(tr::CartesianTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout

    min_py = minimum(tr.py)

    sampling_mask = [CartesianIndices((1:ns, tr.py[r]-min_py+1:tr.py[r]-min_py+1)) for r in 1:nr]

    return sampling_mask
end
