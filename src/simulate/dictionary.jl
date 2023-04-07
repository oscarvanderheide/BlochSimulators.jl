"""
    simulate(resource, sequence, parameters)

Simulate the magnetization at echo times (without any spatial encoding gradients applied) 
for all combinations of tissue parameters contained in `parameters`. 

Use this function to generate dictionaries for MR Fingerprinting purposes.

# Arguments
- `resource::ComputationalResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with different combinations of tissue parameters

# Returns
- `dictionary::AbstractArray`: Array of size (output_dimensions(sequence), length(parameters)) containing the
    magnetization at echo times for all combinations of input tissue parameters.
"""
function simulate(resource, sequence, parameters)

    # intialize array to store echos for each voxel
    output = _allocate_output(resource, sequence, parameters)

    # perform Bloch simulations on given hardware resource
    _simulate!(output, resource, sequence, parameters)

    return output
end

"""
    _allocate_output(resource, sequence::BlochSimulator, parameters)

Allocate an array to store the output of the Bloch simulations (per voxel, echo times only)
to be performed with the `sequence`. For each `BlochSimulator`, methods should have been
added to `output_eltype` and `output_dimensions` for this function to work properly.
"""
function _allocate_output(resource, sequence::BlochSimulator, parameters)

    type = output_eltype(sequence)
    dimensions = output_dimensions(sequence)

    if resource == CUDALibs()
        # allocate a CuArray of zeros on GPU
        output = CUDA.zeros(type, dimensions..., length(parameters))
    elseif resource == CPUProcesses()
        # allocate a DArray of zeros
        distribution = append!(fill(1, length(dimensions)), nworkers())
        output = dzeros(type, (dimensions..., length(parameters)), workers(), distribution)
    else
        # allocate an Array of zeros on the local CPU
        output = zeros(type, dimensions..., length(parameters))
    end

    return output
end

## 1. CPU code

"""
    _simulate!(output, ::CPU1, sequence, parameters)T

Run Bloch simulations in a serial fashion on a single worker.
"""
function _simulate!(output, ::CPU1, sequence, parameters)

    # initialize state that gets updated during time integration
    state = initialize_states(CPU1(), sequence)

    # voxel dimension of output array
    vd = length(size(output))

    # loop over voxels
    for voxel ∈ eachindex(parameters)
        simulate!(selectdim(output, vd, voxel), sequence, state, parameters[voxel])
    end

    return nothing
end

"""
    _simulate!(output, ::CPUThreads, sequence, parameters)

Run Bloch simulations in a multi-threaded fashion on a single worker.
"""
function _simulate!(output, ::CPUThreads, sequence, parameters)

    # initialize state that gets updated during time integration
    state = initialize_states(CPUThreads(), sequence)

    # voxel dimension of output array
    vd = length(size(output))

    # multi-threaded loop over voxels
    Threads.@threads for voxel ∈ eachindex(parameters)
        # simulate!(selectdim(output, vd, voxel), sequence, state, parameters[voxel])
        simulate!(view(output,:,voxel), sequence, state, parameters[voxel])
    end

    return nothing
end

function _simulate!(doutput::DArray, ::CPUProcesses, sequence, dparameters::DArray)
    # "doutput" and "dparamters" are DArrays from the package DistributedArrays.
    # With [:lp] the local part of of such an array is used on a worker
    @sync for p in workers()
        @async @spawnat p _simulate!(doutput[:lp], CPUThreads(), sequence, dparameters[:lp])
    end
    return nothing
end

# If parameters are provided as a regular array instead of a DistributedArray, distribute them first
_simulate!(doutput::DArray, resource::CPUProcesses, sequence, parameters) = _simulate!(doutput, resource, sequence, distribute(parameters))

const THREADS_PER_BLOCK = 32

"""
    _simulate!(output, ::CUDALibs, sequence, parameters)

Run Bloch simulations on a CUDA compatible GPU. Assumes output, sequence and parameters
are already transferred to the GPU with the `gpu` function.
"""
function _simulate!(output, ::CUDALibs, sequence, parameters)

    # threads per block hardcoded for now
    # compute nr of threadblocks to be used on GPU
    nr_voxels = length(parameters)
    nr_blocks = cld(nr_voxels, THREADS_PER_BLOCK)

    # launch kernels
    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK cuda_simulate_kernel!(output, sequence, parameters)
    end

    return nothing
end

"""
    cuda_simulate_kernel!(output, sequence, parameters)

Kernel function that gets launched by each thread on the GPU. Each thread performs Bloch simulations in a single voxel.
"""
function cuda_simulate_kernel!(output, sequence, parameters)

    # get voxel index
    voxel = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # initialize state that gets updated during time integration
    states = initialize_states(CUDALibs(), sequence)

    # perform actual simulations
    if voxel <= length(parameters)
        simulate!(view(output,:,voxel), sequence, states, parameters[voxel])
    end

    return nothing
end
