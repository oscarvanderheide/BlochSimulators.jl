#=========================================================================================
# Magnetization Simulation
#
# This file implements the core magnetization simulation functionality for BlochSimulators.jl
#
# ## High-Level Workflow
#
# 1. User calls `simulate_magnetization(sequence, parameters)`
# 2. The function automatically detects the computational resource based on input types:
#    - Single voxel → CPU1 (single-threaded)
#    - CPU array → CPUThreads (multi-threaded, default)
#    - CuArray → CUDALibs (GPU)
#    - DArray → CPUProcesses (distributed across workers)
# 3. Memory is allocated for the output magnetization array
# 4. The simulation is dispatched to the appropriate parallel execution strategy
# 5. Results are returned as an array: (output_size(sequence)..., num_voxels)
#
# ## Architecture Overview
#
# The simulation uses multiple dispatch on `AbstractResource` types to select the execution
# strategy:
# - `CPU1()`: Serial execution, one voxel at a time
# - `CPUThreads()`: Parallel execution using Julia threads (`Threads.@threads`)
# - `CPUProcesses()`: Distributed execution across workers using `DArray`
# - `CUDALibs()`: GPU execution with CUDA kernels (WARPSIZE threads per voxel)
#
# Each strategy calls the sequence-specific `simulate_magnetization!` method that
# numerically integrates the Bloch equations for the given sequence and tissue properties
# using either the isochromat or the EPG formalism.
#
# ## For Users
#
# Most users should call the convenience function without specifying a resource:
#   `magnetization = simulate_magnetization(sequence, parameters)`
# Or, in case an NVIDIA GPU is available, one would typically do:
#   `magnetization = simulate_magnetization(gpu(f32(sequence,)) gpu(f32(parameters)))`
#
# The function will automatically select the best computational resource based on your input.
#
# ## For Developers
#
# When implementing a new sequence type (subtype of `BlochSimulator`), you must define:
# 1. `simulate_magnetization!(output, sequence, state, tissue_properties)` - Core simulation
# 2. `initialize_states(resource, sequence)` - Initialize state vectors
# 3. `output_size(sequence)` - Dimensions of output per voxel
# 4. `output_eltype(sequence)` - Element type of output (typically ComplexF32 or ComplexF64)
#
# See the Developer Guide in README.md for more details.
=========================================================================================#

#=========================================================================================
# PUBLIC API
=========================================================================================#

"""
    simulate_magnetization(resource, sequence, parameters)

Simulate the magnetization response (typically the transverse magnetization at echo times
without any spatial encoding gradients applied) for all combinations of tissue properties
contained in `parameters`.

This function can also be used to generate dictionaries for MR Fingerprinting purposes.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or
  `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::SimulationParameters`: Array (typically a `StructArray`) containing
  [`AbstractTissueProperties`](@ref) for each voxel. Ensure tissue properties (like `T₁`,
  `T₂`, `B₀`) use the units specified in the `AbstractTissueProperties` docstring (e.g.,
  seconds for relaxation times, Hz for off-resonance).

# Note
- If `resource == CUDALibs()`, the sequence and parameters must have been moved to the GPU
  using `gpu(sequence)` and `gpu(parameters)` prior to calling this function.
- If `resource == CPUProcesses()`, the parameters must be a `DArray` with the first
  dimension corresponding to the number of workers. The function will distribute the
  simulation across the workers in the first dimension of the `DArray`.

# Returns
- `magnetization::AbstractArray`: Array of size (output_size(sequence), length(parameters))
  containing the magnetization response of the sequence for all combinations of input tissue
  properties.
"""
function simulate_magnetization(
    resource::AbstractResource,
    sequence::BlochSimulator,
    parameters::AbstractVector{<:AbstractTissueProperties})

    # Sanity checks
    if hasD(first(parameters)) && !(sequence isa EPGSimulator)
        throw(ArgumentError("Diffusion is only supported for EPG-based sequences"))
    end

    # Allocate array to store magnetization for each voxel
    magnetization = _allocate_magnetization_array(resource, sequence, parameters)

    # Simulate magnetization for each voxel
    simulate_magnetization!(magnetization, resource, sequence, parameters)

    return magnetization
end

#=========================================================================================
# CONVENIENCE METHODS - Automatic Resource Selection
=========================================================================================#

"""
    simulate_magnetization(sequence, parameters)

Convenience function to simulate magnetization without specifying the computational
resource. The function automatically selects the appropriate resource based on the type of
the `sequence` and `parameters`. The fallback case is to use multi-threaded CPU
computations.

# Automatic Resource Selection Rules
- Single `AbstractTissueProperties` → `CPU1()` (single-threaded)
- `StructArray` on CPU → `CPUThreads()` (multi-threaded, default)
- `CuArray` → `CUDALibs()` (GPU)
- `DArray` → `CPUProcesses()` (distributed across workers)
"""
function simulate_magnetization(sequence::BlochSimulator, parameters::StructArray)

    sequence_on_gpu = _all_arrays_are_cuarrays(sequence)
    parameters_on_gpu = _all_arrays_are_cuarrays(parameters)

    if xor(sequence_on_gpu, parameters_on_gpu)
        throw(ArgumentError("Both sequence and parameters must be on the GPU or not on the GPU"))
    end

    if sequence_on_gpu && parameters_on_gpu
        return simulate_magnetization(CUDALibs(), sequence, parameters)
    elseif parameters isa DArray
        return simulate_magnetization(CPUProcesses(), sequence, parameters)
    else
        return simulate_magnetization(CPUThreads(), sequence, parameters)
    end
end

"""
If no `resource` is provided, the simulation is performed on the CPU in a multi-threaded
fashion by default.
"""
function simulate_magnetization(sequence, parameters)
    simulate_magnetization(CPUThreads(), sequence, parameters)
end

"""
However, if the `parameters` are a CuArray, the simulation is performed on the GPU.
"""
function simulate_magnetization(sequence, parameters::CuArray)
    simulate_magnetization(CUDALibs(), gpu(sequence), parameters)
end

"""
If the `parameters` are a DArray, the simulation is performed on the multiple workers.
"""
function simulate_magnetization(sequence, parameters::DArray)
    simulate_magnetization(CPUProcesses(), sequence, parameters)
end

"""
If the tissue properties for a single voxel are provided only, the simulation is performed
on the CPU in a single-threaded fashion.
"""
function simulate_magnetization(sequence, tissue_properties::AbstractTissueProperties)
    # Assemble "SimulationParameters"
    parameters = StructVector([tissue_properties])
    # Perform the simulation for this voxel on the CPU.
    simulate_magnetization(CPU1(), sequence, parameters)
end

#=========================================================================================
# RESOURCE-SPECIFIC IMPLEMENTATIONS
#
# These methods implement the parallel execution strategy for each computational resource.
# They are called by the public API after memory allocation and dispatch based on the
# `AbstractResource` type.
=========================================================================================#

"""
    simulate_magnetization!(magnetization, resource, sequence, parameters)

Simulate the magnetization response for all combinations of tissue properties contained in
`parameters` and stores the results in the pre-allocated `magnetization` array. The actual
implementation depends on the computational resource specified in `resource`.

This function is called by `simulate_magnetization` and is not considered part of the
public API.
"""
function simulate_magnetization!(magnetization, ::CPU1, sequence, parameters)
    # Serial execution: Loop over voxels one at a time
    vd = length(size(magnetization))  # Get voxel dimension (last dimension)
    for voxel ∈ eachindex(parameters)
        # Initialize state vector for this voxel
        state = initialize_states(CPU1(), sequence)
        # Simulate and store result in the voxel's slice of the output array
        simulate_magnetization!(selectdim(magnetization, vd, voxel), sequence, state, parameters[voxel])
    end
    return nothing
end

function simulate_magnetization!(magnetization, ::CPUThreads, sequence, parameters)
    # Parallel execution using Julia threads: Each thread processes voxels independently
    vd = length(size(magnetization))  # Get voxel dimension (last dimension)
    Threads.@threads for voxel ∈ eachindex(parameters)
        # Each thread initializes its own state vector (thread-safe)
        state = initialize_states(CPUThreads(), sequence)
        # Simulate and store result in the voxel's slice of the output array
        simulate_magnetization!(selectdim(magnetization, vd, voxel), sequence, state, parameters[voxel])
    end
    return nothing
end

function simulate_magnetization!(magnetization, ::CPUProcesses, sequence, parameters::DArray)
    # Distributed execution across workers: Each worker processes its local partition
    # The `:lp` selector accesses the local partition of the DArray on each worker
    @sync [@spawnat p simulate_magnetization!(magnetization[:lp], CPU1(), sequence, parameters[:lp]) for p in workers()]
    return nothing
end

function simulate_magnetization!(magnetization, ::CUDALibs, sequence, parameters)
    # GPU execution: Launch CUDA kernel with multiple threads per voxel
    # Each voxel gets assigned `WARPSIZE` threads for coalesced memory access
    # Since `THREADS_PER_BLOCK` is fixed, the number of required blocks is calculated
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels * WARPSIZE, THREADS_PER_BLOCK)

    # Define kernel function (executed on GPU)
    magnetization_kernel!(magnetization, sequence, parameters) = begin

        # Calculate voxel index from block and thread indices
        voxel = cld((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x, WARPSIZE)

        # Initialize state that gets updated during time integration
        states = initialize_states(CUDALibs(), sequence)

        # Do nothing if voxel index is out of bounds (extra threads in last block)
        if voxel > length(parameters)
            return nothing
        end

        # Run simulation for this voxel
        simulate_magnetization!(
            view(magnetization, :, voxel),
            sequence,
            states,
            @inbounds parameters[voxel]
        )
    end

    # Launch kernel and synchronize
    CUDA.@sync begin
        @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_kernel!(magnetization, sequence, parameters)
    end
    return nothing
end

#=========================================================================================
# INTERNAL UTILITIES
#
# Helper functions for memory allocation and array management.
=========================================================================================#

"""
    _allocate_magnetization_array(resource, sequence, parameters)

Allocate an array to store the output of the Bloch simulations (per voxel, echo times only)
to be performed with the `sequence`. For each `BlochSimulator`, methods should have been
added to `output_eltype` and `output_size` for this function to work properly.

This function is called by `simulate_magnetization` and is not considered part of the
public API.

# Returns
- `magnetization_array`: An array allocated on the specified `resource`, formatted to store
  the simulation results for each voxel across the specified echo times.
"""
function _allocate_magnetization_array(resource, sequence, parameters)

    _eltype = output_eltype(sequence)
    _size = (output_size(sequence)..., length(parameters))

    _allocate_array_on_resource(resource, _eltype, _size)
end

"""
    _allocate_array_on_resource(resource, _eltype, _size)

Allocate an array on the specified `resource` with the given element type `_eltype` and size
`_size`. If `resource` is `CPU1()` or `CPUThreads()`, the array is allocated on the CPU. If
`resource` is `CUDALibs()`, the array is allocated on the GPU. For `CPUProcesses()`, the
array is distributed in the "voxel"-dimension over multiple CPU workers.

This function is called by `_allocate_magnetization_array` and is not considered part of
the public API.
"""
function _allocate_array_on_resource(::Union{CPU1,CPUThreads}, _eltype, _size)
    return zeros(_eltype, _size...)
end

function _allocate_array_on_resource(::CUDALibs, _eltype, _size)
    return CUDA.zeros(_eltype, _size...)
end

function _allocate_array_on_resource(::CPUProcesses, _eltype, _size)
    distribution = ones(Int, length(_size) - 1)
    append!(distribution, nworkers())
    dzeros(_eltype, _size, workers(), distribution)
end
