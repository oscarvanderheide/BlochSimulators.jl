"""
    simulate_magnetization(resource, sequence, parameters)

Simulate the magnetization response (typically the transverse magnetization at echo times without any spatial encoding gradients applied) for all combinations of tissue properties contained in `parameters`.

This function can also be used to generate dictionaries for MR Fingerprinting purposes.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::SimulationParameters`: Array with different combinations of tissue properties for each voxel.

# Note
- If `resource == CUDALibs()`, the sequence and parameters must have been moved to the GPU using `gpu(sequence)` and `gpu(parameters)` prior to calling this function.
- If `resource == CPUProcesses()`, the parameters must be a `DArray` with the first dimension corresponding to the number of workers. The function will distribute the simulation across the workers in the first dimension of the `DArray`.

# Returns
- `magnetization::AbstractArray`: Array of size (output_size(sequence), length(parameters)) containing the magnetization response of the sequence for all combinations of input tissue properties.
"""
function simulate_magnetization(
    resource::AbstractResource,
    sequence::BlochSimulator,
    parameters::AbstractVector{<:AbstractTissueProperties})

    # Allocate array to store magnetization for each voxel
    magnetization = _allocate_magnetization_array(resource, sequence, parameters)

    # Simulate magnetization for each voxel
    simulate_magnetization!(magnetization, resource, sequence, parameters)

    return magnetization
end

"""
If no `resource` is provided, the simulation is performed on the CPU in a multi-threaded fashion by default.
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
If the tissue properties for a single voxel are provided only, the simulation is performed on the CPU in a single-threaded fashion.
"""
function simulate_magnetization(sequence, tissue_properties::AbstractTissueProperties)
    # Assemble "SimulationParameters" 
    parameters = StructVector([tissue_properties])
    # Perform simulation for this voxel on CPU
    simulate_magnetization(CPU1(), sequence, parameters)
end

"""
    function simulate_magnetization(sequence, parameters)

Convenience function to simulate magnetization without specifying the computational resource. The function automatically selects the appropriate resource based on the type of the `sequence` and `parameters`. The fallback case is to use multi-threaded CPU computations.
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
    simulate_magnetization!(magnetization, resource, sequence, parameters)

Simulate the magnetization response for all combinations of tissue properties contained in `parameters` and stores the results in the pre-allocated `magnetization` array. The actual implementation depends on the computational resource specified in `resource`.

This function is called by `simulate_magnetization` and is not intended considered part of te public API.
"""
function simulate_magnetization!(magnetization, ::CPU1, sequence, parameters)
    vd = length(size(magnetization))
    for voxel ∈ eachindex(parameters)
        state = initialize_states(CPU1(), sequence)
        simulate_magnetization!(selectdim(magnetization, vd, voxel), sequence, state, parameters[voxel])
    end
    return nothing
end

function simulate_magnetization!(magnetization, ::CPUThreads, sequence, parameters)
    vd = length(size(magnetization))
    Threads.@threads for voxel ∈ eachindex(parameters)
        state = initialize_states(CPUThreads(), sequence)
        simulate_magnetization!(selectdim(magnetization, vd, voxel), sequence, state, parameters[voxel])
    end
    return nothing
end

function simulate_magnetization!(magnetization, ::CPUProcesses, sequence, parameters::DArray)
    # Spawn tasks on each worker
    @sync [@spawnat p simulate_magnetization!(magnetization[:lp], CPU1(), sequence, parameters[:lp]) for p in workers()]
    return nothing
end

function simulate_magnetization!(magnetization, ::CUDALibs, sequence, parameters)
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels * WARPSIZE, THREADS_PER_BLOCK)

    # define kernel
    magnetization_kernel!(magnetization, sequence, parameters) = begin

        # get voxel index
        voxel = cld((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x, WARPSIZE)

        # # initialize state that gets updated during time integration
        states = initialize_states(CUDALibs(), sequence)

        # do nothing if voxel index is out of bounds
        if voxel > length(parameters)
            return nothing
        end

        # run simulation for voxel
        simulate_magnetization!(
            view(magnetization, :, voxel),
            sequence,
            states,
            @inbounds parameters[voxel]
        )
    end

    CUDA.@sync begin
        @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK magnetization_kernel!(magnetization, sequence, parameters)
    end
    return nothing
end

"""
    _allocate_magnetization_array(resource, sequence, parameters)

Allocate an array to store the output of the Bloch simulations (per voxel, echo times only)
to be performed with the `sequence`. For each `BlochSimulator`, methods should have been
added to `output_eltype` and `output_size` for this function to work properly.

This function is called by `simulate_magnetization` and is not intended considered part of te public API.

# Returns
- `magnetization_array`: An array allocated on the specified `resource`, formatted to store the simulation results for each voxel across the specified echo times.
"""
function _allocate_magnetization_array(resource, sequence, parameters)

    _eltype = output_eltype(sequence)
    _size = (output_size(sequence)..., length(parameters))

    _allocate_array_on_resource(resource, _eltype, _size)
end

"""
    _allocate_array_on_resource(resource, _eltype, _size)

Allocate an array on the specified `resource` with the given element type `_eltype` and size `_size`. If `resource` is `CPU1()` or `CPUThreads()`, the array is allocated on the CPU. If `resource` is `CUDALibs()`, the array is allocated on the GPU. For `CPUProcesses()`, the array is distributed in the "voxel"-dimension over multiple CPU workers.

This function is called by `_allocate_magnetization_array` and is not intended considered part of te public API.
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

