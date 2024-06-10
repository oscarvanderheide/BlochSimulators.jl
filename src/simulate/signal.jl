"""
    simulate_signal(resource, sequence, parameters, trajectory, coil_sensitivities)

Simulate the MR signal at timepoint `t` from coil `i` as: `sᵢ(t) = ∑ⱼ cᵢⱼρⱼmⱼ(t)`,
where `cᵢⱼ`is the coil sensitivity of coil `i` at position of voxel `j`, `ρⱼ` is the proton density of voxel `j` and `mⱼ(t)` the (normalized) transverse magnetization in voxel `j` obtained through Bloch simulations.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with tissue parameters for each voxel
- `trajectory::AbstractTrajectory`: Custom trajectory struct
- `coordinates::AbstractVector{<:Coordinates}`: Vector with spatial coordinates for each voxel
- `coil_sensitivities::AbstractMatrix`: Sensitivity of coil `j` in voxel `v` is given by `coil_sensitivities[v,j]`

# Returns
- `signal::AbstractArray{<:Complex}`: Simulated MR signal for the `sequence` and `trajectory`. The array is of size (# samples per readout, # readouts, # coils).
"""
function simulate_signal(
    resource::AbstractResource,
    sequence::BlochSimulator{T},
    parameters::AbstractVector{<:AbstractTissueParameters{N,T}},
    trajectory::AbstractTrajectory{T},
    coordinates::AbstractArray{<:Coordinates{T}},
    coil_sensitivities::AbstractMatrix{Complex{T}}) where {N,T}

    @assert length(parameters) == size(coil_sensitivities, 1)
    # check that proton density is part of parameters
    @assert :ρˣ ∈ fieldnames(eltype(parameters))
    @assert :ρʸ ∈ fieldnames(eltype(parameters))

    # Compute magnetization response (at echo times) in all voxels
    magnetization = simulate_magnetization(resource, sequence, parameters)

    # Apply phase encoding (typically only relevant for Cartesian trajectories)
    phase_encoding!(magnetization, trajectory, coordinates)

    # Compute signal from (phase-encoded) magnetization at echo times
    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    return signal
end

"""
    magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

Allocates memory for the signal and computes the signal for each coil separately using the `_signal_per_coil!` function. 
    
# Implementation details
The `_signal_per_coil!` function has different implementations depending on the computational resources (i.e. the type of `resource`). The default implementations loop over all time points and compute the volume integral of the transverse magnetization in each voxel for each time point separately. This loop order is not necessarily optimal (and performance may be) across all trajectories and computational resources. If a better implementation is available, add new methods to this function for those specific combinations of resources and trajectories.

The "voxels" are assumed to be distributed over the workers. Each worker computes performs a volume integral over the voxels that it owns only (for all time points) using the CPU1() code. The results are then summed up across all workers.

# Note
When using multiple CPU's, the "voxels" are distributed over the workers. Each worker computes the signal for its own voxels in parallel and the results are summed up across all workers.
"""
function magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    # Allocate memory for the signal
    signal = _allocate_signal_array(resource, trajectory, coil_sensitivities)

    # Determine the number of receive coils
    num_coils = size(coil_sensitivities, 2)

    # Compute signal for each receive coil separately
    for j in 1:num_coils
        coil_sensitivitiesⱼ = @view coil_sensitivities[:, j]
        signalⱼ = @view signal[:, j]
        _signal_per_coil!(signalⱼ, resource, magnetization, parameters, trajectory, coordinates, coil_sensitivitiesⱼ)
    end

    return signal
end

function magnetization_to_signal(
    ::CPUProcesses,
    dmagnetization::DArray,
    dparameters::DArray,
    trajectory,
    dcoordinates::DArray,
    dcoil_sensitivities::DArray)

    signal = @distributed (+) for p in workers()
        magnetization_to_signal(
            CPU1(),
            localpart(dmagnetization),
            localpart(dparameters),
            trajectory,
            localpart(dcoordinates),
            localpart(dcoil_sensitivities)
        )
    end

    return signal
end

"""
    _signal_per_coil!(signal, resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

Compute the signal for a given coil by calculating a volume integral of the transverse magnetization in each voxel for each time point separately (using the `signal_at_time_point!` function). Each time point is computed in parallel for multi-threaded CPU computation and on the GPU for CUDA computation.
"""
function _signal_per_coil!(signal, resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    error("This method for _signal_per_coil! should never be called. It's only there for the docstring.")
end

function _signal_per_coil!(signal, ::CPU1, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    for time_point in 1:nsamples(trajectory)
        signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    end
end

function _signal_per_coil!(signal, ::CPUThreads, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    Threads.@threads for time_point in 1:nsamples(trajectory)
        signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    end
end

function _signal_per_coil!(signal, ::CUDALibs, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    # define kernel function to be run by each thread on gpu
    magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities) = begin

        time_point = (blockIdx().x - 1) * blockDim().x + threadIdx().x # global time point index

        if time_point <= nsamples(trajectory)
            signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
        end
        return nothing
    end

    # launch kernels, threads per block hardcoded for now
    CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
end

"""
    signal_at_time_point!(signal, time, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

If `to_sample_point` has been defined for the provided trajectory, this (generic but not optimized)
function computes the signal at timepoint `time_point` for all receive coils. It does so by computing
the readout- and sample indices for the given `time_point`, reading in the magnetization at echo
time of the `r`-th readout, using `to_sample_point` to compute the magnetization at the `s`-th sample
index, and then it integrates over all voxels (while scaling the magnetization with the proper coil
sensitivity and proton density).

Better performance can likely be achieved by incorporating more trajectory-specific information together with different loop orders.
"""

@inline function signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    # compute readout and sample indices for time point t
    readout, sample = _get_readout_and_sample_idx(trajectory, time_point)

    # each element of parameters contains tissue parameters for a single voxel
    num_voxels = length(parameters)

    # accumulator for signal at time index t
    s = zero(eltype(signal))

    for voxel = 1:num_voxels

        # load parameters and spatial coordinates
        @inbounds p = parameters[voxel]
        # load coil sensitivity for coil i in this voxel (SVector of length (# coils))
        @inbounds c = coil_sensitivities[voxel]
        # load magnetization in voxel at echo time of the r-th readout
        @inbounds m = magnetization[readout, voxel]
        # load the spatial coordinates of the voxel
        @inbounds xyz = coordinates[voxel]
        # compute magnetization at s-th sample of r-th readout
        mₛ = to_sample_point(m, trajectory, readout, sample, xyz, p)
        # add magnetization from this voxel, scaled with proton density
        # and coil sensitivity, to signal accumulator s
        ρ = complex(p.ρˣ, p.ρʸ)
        s += mₛ * (ρ * c)
    end

    # store signal for each coil at time t
    @inbounds signal[time_point] = s

    return nothing
end

"""
    _allocate_signal_array(resource, trajectory::AbstractTrajectory, coil_sensitivities)

Allocate an array to store the output of the signal simulation (all readout points,
integrated over all voxels).
"""
function _allocate_signal_array(resource, trajectory, coil_sensitivities::AbstractMatrix{T}) where {T<:Complex}

    num_samples = nsamples(trajectory)
    num_coils = size(coil_sensitivities, 2)

    return _allocate_array_on_resource(resource, T, (num_samples, num_coils))
end

### Convenience functions

"""
When coil sensitivities are not provided, use a single coil with sensitivity = 1 everywhere
"""
function simulate_signal(resource, sequence::BlochSimulator, parameters::AbstractArray{<:AbstractTissueParameters{N,T}}, trajectory, coordinates) where {N,T}

    # use one coil with sensitivity 1 everywhere
    coil_sensitivities = ones(Complex{T}, length(parameters), 1)

    # send to GPU if necessary
    if resource == CUDALibs()
        coil_sensitivities = gpu(coil_sensitivities)
    elseif resource == CPUProcesses()
        coil_sensitivities = distribute(coil_sensitivities)
    end

    # simulate signal and use only because there's only one coil anyway
    signal = simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)

    return signal
end