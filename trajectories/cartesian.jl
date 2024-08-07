"""
    CartesianTrajectory2D{T,I,U,V} <: CartesianTrajectory{T}

Struct that is used to implement a typical Cartesian 2D gradient trajectory.
The trajectory is described in a compact fashion by only storing
the starting position in k-space (`k_start_readout`) for each readout
as well as the step in k-space per readout point `Δk_adc`.

Note that CartesianTrajectory2D and RadialTrajectory2D are essentially the same when using this compact description. A SpokesTrajectory struct is therefore
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
struct CartesianTrajectory2D{T<:Real,I<:Integer,U<:AbstractVector,V<:AbstractVector} <: SpokesTrajectory{T}
    nreadouts::I
    nsamplesperreadout::I
    Δt::T # time between sample points
    k_start_readout::U # starting position in k-space for each readout
    Δk_adc::T # Δk during ADC for each readout
    py::V # phase encoding order, not needed but nice to store
    readout_oversampling::I # readout oversampling factor
end

@functor CartesianTrajectory2D
@adapt_structure CartesianTrajectory2D

export CartesianTrajectory2D

"""
    magnetization_to_signal(::Union{CPU1,CPUThreads,CUDALibs}, magnetization, parameters, trajectory::CartesianTrajectory2D, coordinates, coil_sensitivities)

# Arguments
- `magnetization`:          Matrix{Complex} of size (# readouts, # voxels) with phase-encoded
                    magnetization at echo times.
- `parameters`:     Tissue parameters of all voxels, including spatial coordinates.
- `trajectory`:     Cartesian trajectory struct.
- `coordinates`:    Vector{Coordinates} with spatial coordinates for each voxel.
- `coil_sensitivities`:     Matrix{Complex} of size (# voxels, # coils) with coil sensitivities.

# Returns
- `signal`: Vector of length (# coils) with each element a Matrix{Complex}
        of size (# readouts, # samples per readout)

# Extended help
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

signalⱼ[r,s] = sum( magnetization[r,v] * E[v]^s * ρ[v] * cⱼ[v] for v in 1:(# voxels))
signalⱼ[r,s] = magnetization[r,:] * (E.^s .* ρ .* cⱼ)

Because the (E.^s .* ρ .* cⱼ)-part is the same for all readouts, we can simply
perform this computation for all readouts simultaneously as
signalⱼ[:,s] = magnetization * (E.^s .* ρ .* cⱼ)

If we define the matrix Eˢ as E .^ (-(ns÷2):(ns÷2)-1), then we can do the computation
for all different sample points at the same time as well using a single matrix-matrix
multiplication:
signalⱼ = magnetization * (Eˢ .* (ρ .* cⱼ))

The signalⱼ array is of size (# readouts, # samples per readout). We prefer to have it transposed, therefore we compute
signalⱼ = transpose(Eˢ .* (ρ .* cⱼ)) * transpose(magnetization)
instead.

For the final output, we do this calculation for each coil j and get a vector of signal matrices (one matrix for each coil) as a result.

Note that this implementation relies entirely on vectorized code
and works on both CPU and GPU. The matrix-matrix multiplications
are - I think - already multi-threaded so a separate multi-threaded
implementation is not needed.
"""
function magnetization_to_signal(
    ::Union{CPU1,CPUThreads,CUDALibs},
    magnetization,
    parameters::SimulationParameters,
    trajectory::CartesianTrajectory2D,
    coordinates::StructArray{<:Coordinates},
    coil_sensitivities)

    # Sanity checks
    @assert size(magnetization) == (trajectory.nreadouts, length(parameters))
    @assert size(coil_sensitivities, 1) == length(parameters)

    # Load constants
    T₂ = parameters.T₂
    ρ = complex.(parameters.ρˣ, parameters.ρʸ)
    x = coordinates.x
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    ns = trajectory.nsamplesperreadout
    nr = trajectory.nreadouts
    nc = size(coil_sensitivities, 2)

    # Dynamics during readout (gradient encoding, T2 decay, B0 rotation)
    # are different per voxel but the same for each readout and sample point
    # Pre-compute vector with dynamic change from one sample point to the next

    # Gradient rotation per sample point
    θ = Δkₓ * x
    # Add B₀ rotation per sample point
    CUDA.@allowscalar if hasB₀(first(parameters))
        @. θ += Δt * π * parameters.B₀ * 2
    end
    # Combined rotation/decay per sample point
    E = @. exp(-Δt * inv(T₂) + im * θ)
    # To compute signal at all sample points with one matrix-matrix multiplication,
    # we pre-compute a readout_dynamics matrix instead
    Eˢ = @. E^(-(ns ÷ 2):(ns÷2)-1)'
    # Allocate output array
    signal = similar(magnetization, (ns, nr, nc))
    # Perform main computations
    for j in 1:nc
        signalⱼ = @view signal[:, :, j]
        coilⱼ = @view coil_sensitivities[:, j]
        mul!(signalⱼ, transpose((Eˢ .* (ρ .* coilⱼ))), transpose(magnetization))
    end

    return signal
end

### Interface requirements

function phase_encoding!(magnetization, trajectory::CartesianTrajectory2D, coordinates::StructArray{<:Coordinates})
    y = coordinates.y |> vec
    kʸ = imag.(trajectory.k_start_readout)
    @. magnetization *= exp(im * kʸ * y')
    return nothing
end

# perhaps do this with metaprogramming instead (iteratate over all subtypes of AbstractTrajectory)
function phase_encoding!(magnetization::DArray, trajectory::CartesianTrajectory2D, coordinates::DArray)

    @sync for p in workers()
        @async begin
            @spawnat p phase_encoding!(localpart(magnetization), trajectory, localpart(coordinates))
        end
    end

    return nothing
end

@inline function to_sample_point(mₑ::Complex, trajectory::CartesianTrajectory2D, readout_idx, sample_idx, r::Coordinates, p::AbstractTissueProperties)

    # Note that m has already been phase-encoded

    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    x = coordinates.x
    y = coordinates.y
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
CartesianTrajectory2D(nr, ns) = CartesianTrajectory2D(nr, ns, 10^-5, complex.(ones(nr)), 1.0, (1:nr) .- (nr ÷ 2), 2)

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::CartesianTrajectory2D, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.k_start_readout[idx], tr.Δk_adc, tr.py[idx], tr.readout_oversampling)

# Nicer printing in REPL
Base.show(io::IO, tr::CartesianTrajectory2D) = begin
    println("")
    println(io, "Cartesian trajectory 2D")
    println(io, "nreadouts:            ", tr.nreadouts)
    println(io, "nsamplesperreadout:   ", tr.nsamplesperreadout)
    println(io, "Δt:                   ", tr.Δt)
    println(io, "k_start_readout:      ", typeof(tr.k_start_readout))
    println(io, "Δk_adc:               ", tr.Δk_adc)
    println(io, "py:                   ", typeof(tr.py))
    println(io, "readout_oversampling: ", tr.readout_oversampling)
end

"""
    kspace_coordinates(tr::CartesianTrajectory2D)

Return matrix (nrsamplesperreadout, nrreadouts) with kspace coordinates for the trajectory. Needed for nuFFT reconstructions.
"""
function kspace_coordinates(tr::CartesianTrajectory2D)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout
    k = [tr.k_start_readout[r] + (s - 1) * tr.Δk_adc for s in 1:ns, r in 1:nr]

    return k
end

"""
    sampling_mask(tr::CartesianTrajectory2D)

For undersampled Cartesian trajectories, the gradient trajectory can also be described by a sampling mask.
"""
function sampling_mask(tr::CartesianTrajectory2D)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout

    min_py = minimum(tr.py)

    sampling_mask = [CartesianIndices((1:ns, tr.py[r]-min_py+1:tr.py[r]-min_py+1)) for r in 1:nr]

    return sampling_mask
end
