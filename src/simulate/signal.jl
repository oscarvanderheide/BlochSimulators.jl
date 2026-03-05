#=========================================================================================
# Signal Simulation
#
# This file implements the MR signal simulation functionality for BlochSimulators.jl
#
# ## High-Level Workflow
#
# 1. User calls `simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)`
# 2. The magnetization response is computed at echo times using `simulate_magnetization`
# 3. Phase encoding is applied (typically only for Cartesian trajectories)
# 4. The signal is computed by integrating over all voxels at each time point:
#    sᵢ(t) = ∑ⱼ cᵢⱼρⱼmⱼ(t), where:
#    - cᵢⱼ = coil sensitivity of coil i at voxel j
#    - ρⱼ = proton density of voxel j (complex: ρˣ + i·ρʸ)
#    - mⱼ(t) = transverse magnetization in voxel j at time t
# 5. Results are returned as an array: (num_samples, num_coils)
#
# ## Architecture Overview
#
# The simulation pipeline consists of three main stages:
# 1. **Magnetization simulation**: Compute mⱼ(t) using Bloch equations
# 2. **Phase encoding**: Apply spatial encoding (trajectory-specific)
# 3. **Signal integration**: Sum over voxels with coil sensitivities
#
# The signal integration uses multiple dispatch on `AbstractResource` types:
# - `CPU1()`: Serial execution, one time point at a time
# - `CPUThreads()`: Parallel execution over time points using Julia threads
# - `CUDALibs()`: GPU execution with one thread per time point
# - `CPUProcesses()`: Distributed execution, each worker processes its voxels
#
# ## For Users
#
# Most users should call:
#   `signal = simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)`
#
# If coil sensitivities are not available, a convenience method with uniform sensitivity is provided.
#
# For large datasets, use the partitioned version to process voxels in batches and avoid memory issues.
#
# ## For Developers
#
# When implementing a new trajectory type (subtype of `AbstractTrajectory`), you should define:
# 1. `to_sample_point(m, trajectory, readout, sample, xyz, p)` - Convert magnetization at echo time to sample point
# 2. `phase_encoding!(magnetization, trajectory, coordinates)` - Apply phase encoding (if needed)
# 3. `nsamples(trajectory)` - Total number of sample points
#
# See the Developer Guide in README.md for more details.
=========================================================================================#

#=========================================================================================
# PUBLIC API
=========================================================================================#

"""
    simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)

Simulate the MR signal at timepoint `t` from coil `i` as: `sᵢ(t) = ∑ⱼ cᵢⱼρⱼmⱼ(t)`, where
`cᵢⱼ` is the coil sensitivity of coil `i` at the position of voxel `j`, `ρⱼ` is the proton
density of voxel `j`, and `mⱼ(t)` is the (normalized) transverse magnetization in voxel `j`
obtained through Bloch simulations.

Note that it calls `simulate_magnetization` to compute the magnetization response at echo
times in all voxels, `phase_encoding!` to apply phase encoding (typically only relevant
for Cartesian trajectories), and `magnetization_to_signal` to compute the signal from the
(phase-encoded) magnetization at echo times.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or
  `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::SimulationParameters`: Array (typically a `StructArray`) containing
  [`AbstractTissueProperties`](@ref) for each voxel. **Must** include proton density fields
  (`ρˣ`, `ρʸ`). Ensure tissue properties (like `T₁`, `T₂`, `B₀`) use the units specified in
  the `AbstractTissueProperties` docstring (e.g., seconds for relaxation times, Hz for
  off-resonance).
- `trajectory::AbstractTrajectory`: Custom trajectory struct
- `coordinates::StructArray{<:Coordinates}`: Array with spatial coordinates for each voxel (in **cm**)
- `coil_sensitivities::AbstractMatrix`: Sensitivity of coil `j` in voxel `v` is given by
  `coil_sensitivities[v,j]`

# Returns
- `signal::AbstractArray{<:Complex}`: Simulated MR signal for the `sequence` and
  `trajectory`. The array is of size (# samples per readout, # readouts, # coils).
"""
function simulate_signal(
    resource::AbstractResource,
    sequence::BlochSimulator{T},
    parameters::StructArray{<:AbstractTissueProperties{N,T}},
    trajectory::AbstractTrajectory{T},
    coordinates::StructArray{<:Coordinates{T}},
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

#=========================================================================================
# CONVENIENCE METHODS
=========================================================================================#

"""
When coil sensitivities are not provided, use a single coil with sensitivity = 1 everywhere
"""
function simulate_signal(resource, sequence::BlochSimulator, parameters::AbstractArray{<:AbstractTissueProperties{N,T}}, trajectory, coordinates) where {N,T}

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

"""
    simulate_signal(sequence, partitioned_parameters::AbstractVector{<:SimulationParameters})

In situations where the number of voxels is too large to store the intermediate
`magnetization` array, the signal can be calculated in batches: the voxels are divided (by
the user) into partitions and the signal is calculated for each partition separately. The
final signal is the sum of the signals from all partitions.
"""
function simulate_signal(
    resource::AbstractResource,
    sequence::BlochSimulator,
    partitioned_parameters::AbstractVector{<:SimulationParameters},
    trajectory::AbstractTrajectory,
    partitioned_coordinates::AbstractVector{<:StructArray{<:Coordinates}},
    partitioned_coil_sensitivities::AbstractVector{<:AbstractMatrix{<:Complex}})

    # Ensure all partitioned arguments have the same number of partitions
    @assert length(partitioned_parameters) == length(partitioned_coordinates) == length(partitioned_coil_sensitivities)

    # Ensure that each partition has the same number of voxels
    @assert all(
        length(partitioned_parameters[i]) == size(partitioned_coil_sensitivities[i], 1) == length(partitioned_coordinates[i]) for i in 1:length(partitioned_parameters))

    # Define a helper function to calculate the signal for a single partition
    _simulate_signal(parameters, coordinates, coil_sensitivities) = begin
        simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)
    end

    # Loop over all partitions and sum the results
    signal = mapreduce(
        _simulate_signal,
        +,
        partitioned_parameters,
        partitioned_coordinates,
        partitioned_coil_sensitivities
    )

    return signal
end

#=========================================================================================
# MAGNETIZATION TO SIGNAL CONVERSION
#
# These functions convert the magnetization response (at echo times) to the actual MR signal
# by applying spatial encoding and integrating over all voxels with coil sensitivities.
=========================================================================================#

"""
    magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

Allocates memory for the signal and computes the signal for each coil separately using the
`_signal_per_coil!` function.

# Implementation details
The `_signal_per_coil!` function has different implementations depending on the
computational resources (i.e. the type of `resource`). The default implementations loop over
all time points and compute the volume integral of the transverse magnetization in each
voxel for each time point separately. This loop order is not necessarily optimal (and
performance may be) across all trajectories and computational resources. If a better
implementation is available, add new methods to this function for those specific
combinations of resources and trajectories.

The "voxels" are assumed to be distributed over the workers. Each worker computes performs a
volume integral over the voxels that it owns only (for all time points) using the CPU1()
code. The results are then summed up across all workers.

# Note
When using multiple CPU's, the "voxels" are distributed over the workers. Each worker
computes the signal for its own voxels in parallel and the results are summed up across all
workers.
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

    # Distributed execution: Each worker processes its local voxels, results are summed
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

#=========================================================================================
# RESOURCE-SPECIFIC IMPLEMENTATIONS
#
# These methods implement the parallel execution strategy for signal computation from
# magnetization. They integrate over voxels at each time point using different parallelization
# strategies.
=========================================================================================#

"""
    _signal_per_coil!(signal, resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

Compute the signal for a given coil by calculating a volume integral of the transverse
magnetization in each voxel for each time point separately (using the
`signal_at_time_point!` function). Each time point is computed in parallel for
multi-threaded CPU computation and on the GPU for CUDA computation.
"""
function _signal_per_coil!(signal, resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    error("This method for _signal_per_coil! should never be called. It's only there for the docstring.")
end

function _signal_per_coil!(signal, ::CPU1, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    # Serial execution: Loop over time points one at a time
    for time_point in 1:nsamples(trajectory)
        signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    end
end

function _signal_per_coil!(signal, ::CPUThreads, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    # Parallel execution using Julia threads: Each thread processes time points independently
    Threads.@threads for time_point in 1:nsamples(trajectory)
        signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    end
end

function _signal_per_coil!(signal, ::CUDALibs, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
    # GPU execution: Launch CUDA kernel with one thread per time point
    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    # Define kernel function to be run by each thread on GPU
    magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities) = begin

        # Calculate time point index from block and thread indices
        time_point = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        # Compute signal at this time point if within bounds
        if time_point <= nsamples(trajectory)
            signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
        end
        return nothing
    end

    # Launch kernel and synchronize
    CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
end

#=========================================================================================
# SIGNAL COMPUTATION AT INDIVIDUAL TIME POINTS
=========================================================================================#

"""
    signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

If `to_sample_point` has been defined for the provided trajectory, this (generic but not
optimized) function computes the signal at timepoint `time_point` for all receive coils. It
does so by computing the readout- and sample indices for the given `time_point`, reading in
the magnetization at echo time of the `r`-th readout, using `to_sample_point` to compute the
magnetization at the `s`-th sample index, and then it integrates over all voxels (while
scaling the magnetization with the proper coil sensitivity and proton density).

Better performance can likely be achieved by incorporating more trajectory-specific
information together with different loop orders.
"""
@inline function signal_at_time_point!(signal, time_point, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    # Compute readout and sample indices for time point t
    readout, sample = _get_readout_and_sample_idx(trajectory, time_point)

    # Each element of parameters contains tissue properties for a single voxel
    num_voxels = length(parameters)

    # Accumulator for signal at time index t
    s = zero(eltype(signal))

    # Integrate over all voxels: s(t) = ∑ⱼ cⱼρⱼmⱼ(t)
    for voxel = 1:num_voxels

        # Load parameters and spatial coordinates
        @inbounds p = parameters[voxel]
        # Load coil sensitivity for this coil in this voxel
        @inbounds c = coil_sensitivities[voxel]
        # Load magnetization in voxel at echo time of the r-th readout
        @inbounds m = magnetization[readout, voxel]
        # Load the spatial coordinates of the voxel (in cm)
        @inbounds xyz = coordinates[voxel]
        # Compute magnetization at s-th sample of r-th readout (apply trajectory-specific encoding)
        mₛ = to_sample_point(m, trajectory, readout, sample, xyz, p)
        # Add magnetization from this voxel, scaled with proton density and coil sensitivity
        ρ = complex(p.ρˣ, p.ρʸ)
        s += mₛ * (ρ * c)
    end

    # Store signal for this coil at time t
    @inbounds signal[time_point] = s

    return nothing
end

#=========================================================================================
# INTERNAL UTILITIES
#
# Helper functions for memory allocation.
=========================================================================================#

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
