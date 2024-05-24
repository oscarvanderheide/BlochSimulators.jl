"""
    simulate_magnetization(resource, sequence, parameters)

Simulate the magnetization response (typically the transverse magnetization at echo times without any spatial encoding gradients applied)
for all combinations of tissue parameters contained in `parameters`.

This function can also be used to generate dictionaries for MR Fingerprinting purposes.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with different combinations of tissue parameters

# Returns
- `magnetization::AbstractArray`: Array of size (output_size(sequence), length(parameters)) containing the
    magnetization response of the sequence for all combinations of input tissue parameters.
"""
function simulate_magnetization(resource::AbstractResource, sequence, parameters)

    # Allocate array to store magnetization for each voxel
    magnetization = _allocate_magnetization_array(resource, sequence, parameters)

    # Simulate magnetization for each voxel
    simulate_magnetization!(magnetization, resource, sequence, parameters)

    return magnetization
end

"""
    simulate_magnetization!(magnetization, resource, sequence, parameters)

Simulate the magnetization at echo times (without any spatial encoding gradients applied)
for all combinations of tissue parameters contained in `parameters`. Stores the magnetization response (typically the transverse magnetization at echo times)
in the `magnetization` array.

- `magnetization::AbstractArray`: Pre-allocated output array to store simulation results.
- `resource::AbstractResource`: Computational resource (e.g., `CPU1()`, `CPUThreads()`, `CPUProcesses()`, `CUDALibs()`).
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with different combinations of tissue parameters
"""
function simulate_magnetization!(magnetization, ::CPU1, sequence, parameters)
    state = initialize_states(CPU1(), sequence)
    vd = length(size(magnetization))
    for voxel ∈ eachindex(parameters)
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

function simulate_magnetization!(magnetization, ::CUDALibs, sequence, parameters::CuArray)
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels, THREADS_PER_BLOCK)

    # define kernel
    magnetization_kernel!(magnetization, sequence, parameters) = begin

        # get voxel index
        voxel = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        # initialize state that gets updated during time integration
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

# Arguments
- `resource::AbstractResource`: The computational resource (e.g., CPU, GPU) to be used for the allocation.
- `sequence::BlochSimulator{T}`: The simulator, which defines the type of simulation to be performed.
- `parameters::AbstractVector{<:AbstractTissueParameters{N,T}}`: A vector with each element containing the tissue parameters for a voxel.

# Returns
- `magnetization_array`: An array allocated on the specified `resource`, formatted to store the simulation results for each voxel across the specified echo times.
"""
function _allocate_magnetization_array(
    resource::AbstractResource,
    sequence::BlochSimulator{T},
    parameters::AbstractVector{<:AbstractTissueParameters{N,T}}
) where {N,T}

    _eltype = output_eltype(sequence)
    _size = (output_size(sequence)..., length(parameters))

    _allocate_array_on_resource(resource, _eltype, _size)
end

"""
    _allocate_array_on_resource(resource::AbstractResource, _eltype, _size)

Allocate an array on the specified `resource` with the given element type `_eltype` and size `_size`. If `resource` is `CPU1()` or `CPUThreads()`, the array is allocated on the CPU. If `resource` is `CUDALibs()`, the array is allocated on the GPU. For `CPUProcesses()`, the array is distributed in the "voxel"-dimension over multiple CPU workers.
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

"""
    function simulate_magnetization(sequence, parameters)

Convenience function to simulate magnetization without specifying the computational resource. The function automatically selects the appropriate resource based on the type of the `parameters` argument.

- If the `parameters` are provided as a CuArray, the `sequence` is made GPU-compatible as well and the simulation is performed on the GPU.
- If the `parameters` are provided as a DArray, the simulation is performed on the multiple workers.
- If the `parameters` are provided as a regular array, the simulation is performed on the CPU in a multi-threaded fashion.
"""
function simulate_magnetization(sequence, parameters)
    simulate_magnetization(CPUThreads(), sequence, parameters)
end

function simulate_magnetization(sequence, parameters::CuArray)
    simulate_magnetization(CUDALibs(), gpu(sequence), parameters)
end

function simulate_magnetization(sequence, parameters::DArray)
    simulate_magnetization(CPUProcesses(), sequence, parameters)
end

