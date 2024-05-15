"""
    simulate_magnetization(resource, sequence, parameters)

Simulate the magnetization at echo times (without any spatial encoding gradients applied)
for all combinations of tissue parameters contained in `parameters`.

This function can also be used to generate dictionaries for MR Fingerprinting purposes.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with different combinations of tissue parameters

# Returns
- `output::AbstractArray`: Array of size (output_dimensions(sequence), length(parameters)) containing the
    magnetization at echo times for all combinations of input tissue parameters.
"""
function simulate_magnetization(resource, sequence, parameters) end

"""
    simulate_magnetization(::CPU1, sequence, parameters)

Perform simulations on a single CPU by looping over all entries of `parameters`
and performing Bloch simulations for each combination of tissue parameters.
"""
function simulate_magnetization(::CPU1, sequence, parameters)

    # intialize array to store magnetization for each voxel
    output = _allocate_magnetization_array(CPU1(), sequence, parameters)

    # initialize state that gets updated during time integration
    state = initialize_states(CPU1(), sequence)
    # voxel dimension of output array
    vd = length(size(output))
    # loop over voxels
    for voxel ∈ eachindex(parameters)
        # run simulation for voxel
        simulate_magnetization!(selectdim(output, vd, voxel), sequence, state, parameters[voxel])
    end

    return output
end

"""
    simulate_magnetization(::CPUThreads, sequence, parameters)

Perform simulations by looping over all entries of `parameters` in a
multi-threaded fashion. See the [Julia documentation](https://docs.julialang.org/en/v1/manual/multi-threading/)
for more details on how to launch Julia with multiple threads of execution.
"""
function simulate_magnetization(::CPUThreads, sequence, parameters)

    # intialize array to store magnetization for each voxel
    output = _allocate_magnetization_array(CPUThreads(), sequence, parameters)

    # voxel dimension of output array
    vd = length(size(output))
    # multi-threaded loop over voxels
    Threads.@threads for voxel ∈ eachindex(parameters)
        # initialize state that gets updated during time integration
        state = initialize_states(CPUThreads(), sequence)
        # run simulation for voxel
        simulate_magnetization!(selectdim(output, vd, voxel), sequence, state, parameters[voxel])
    end

    return output
end

"""
    simulate_magnetization(::CPUProcesses, sequence, dparameters::DArray)

Perform simulations using multiple, distributed CPUs. See the [Julia documentation](https://docs.julialang.org/en/v1/manual/distributed-computing/) and the [DistributedArrays](https://github.com/JuliaParallel/DistributedArrays.jl) package
for more details on how to use Julia with multiple workers.
"""
function simulate_magnetization(::CPUProcesses, sequence, dparameters::DArray)
    # DArrays are from the package DistributedArrays.
    # With [:lp] the local part of of such an array is used on a worker
    doutput = @sync [@spawnat p simulate_magnetization(CPU1(), sequence, dparameters[:lp]) for p in workers()]
    # On each worker, a part of the magnetization array is now computed.
    # Turn it into a single DArray with the syntax below
    return DArray(permutedims(doutput))
end

# If parameters are provided as a regular array instead of a DistributedArray, distribute them first
simulate_magnetization(resource::CPUProcesses, sequence, parameters) = simulate_magnetization(resource, sequence, distribute(parameters))

"""
    simulate_magnetization(::CUDALibs, sequence, parameters::CuArray)

Perform simulations on NVIDIA GPU hardware by making use of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package.
Each thread perform Bloch simulations for a single entry of the `parameters` array.
"""
function simulate_magnetization(::CUDALibs, sequence, parameters::CuArray)

    # intialize array to store magnetization for each voxel
    output = _allocate_magnetization_array(CUDALibs(), sequence, parameters)

    # compute nr of threadblocks to be used on GPU
    # threads per block hardcoded for now
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels, THREADS_PER_BLOCK)

    # define kernel function to be run by each thread on gpu
    magnetization_kernel!(output, sequence, parameters) = begin

        # get voxel index
        voxel = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        # initialize state that gets updated during time integration
        states = initialize_states(CUDALibs(), sequence)

        if voxel <= length(parameters)
            # run simulation for voxel
            simulate_magnetization!(view(output,:,voxel), sequence, states, parameters[voxel])
        end

        return nothing
    end

    # launch kernels
    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK magnetization_kernel!(output, sequence, parameters)
    end

    return output
end

# If parameters are not provided as CuArray, send them (and sequence struct) to gpu first
simulate_magnetization(resource::CUDALibs, sequence, parameters) = simulate_magnetization(resource, gpu(sequence), gpu(parameters))


"""
    _allocate_magnetization_array(resource, sequence, parameters)

Allocate an array to store the output of the Bloch simulations (per voxel, echo times only)
to be performed with the `sequence`. For each `BlochSimulator`, methods should have been
added to `output_eltype` and `output_dimensions` for this function to work properly.

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
    _size = (output_dimensions(sequence)..., length(parameters))

    _allocate_array_on_resource(resource, _eltype, _size)
end

"""
    _allocate_array_on_resource(::Union{CPU1,CPUThreads}, _eltype, _size)

Allocate a CPU array for use with a single CPU or multiple threads.
"""
function _allocate_array_on_resource(::Union{CPU1,CPUThreads}, _eltype, _size)
    return zeros(_eltype, _size...)
end
"""
    _allocate_array_on_resource(::CUDALibs, _eltype, _size)

Specialized allocation for CUDA-enabled devices.
"""
function _allocate_array_on_resource(::CUDALibs, _eltype, _size)
    return CUDA.zeros(_eltype, _size...)
end

"""
    _allocate_array_on_resource(::CPUProcesses, _eltype, _size)

Allocate a distributed array for use with multiple CPU processes. The array is distributed
in the "voxel" dimension (last dimension) over the workers.
"""
function _allocate_array_on_resource(::CPUProcesses, _eltype, _size)
    distribution = ones(Int, length(_size)-1)
    append!(distribution, nworkers())
    dzeros(_eltype, _size, workers(), distribution)
end