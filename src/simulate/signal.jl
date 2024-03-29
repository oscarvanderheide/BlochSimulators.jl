### Signal simulation code

"""
    simulate_signal(resource, sequence, parameters, trajectory, coil_sensitivities)

Simulate the MR signal at timepoint `t` from coil `i` as: `sᵢ(t) = ∑ⱼ cᵢⱼρⱼmⱼ(t)`,
where `cᵢⱼ`is the coil sensitivity of coil `i` at position of voxel `j`,
`ρⱼ` is the proton density of voxel `j` and `mⱼ(t)` the (normalized) transverse magnetization
in voxel `j` obtained through Bloch simulations.

# Arguments
- `resource::AbstractResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with tissue parameters for each voxel
- `trajectory::AbstractTrajectory`: Custom trajectory struct
- `coil_sensitivities::AbstractVector{<:SVector{ncoils}}`: Vector with `ncoils` coil sensitivities for each voxel

# Returns
- `signal::Vector{<:SVector{ncoils}}`: Simulated MR signal for the `sequence` and `trajectory`.
At each timepoint, the signal for each of the `ncoils` is stored.
"""
function simulate_signal(resource, sequence, parameters, trajectory, coil_sensitivities)

    @assert length(parameters) == size(coil_sensitivities,1)
    # check that proton density is part of parameters
    @assert :ρˣ ∈ fieldnames(eltype(parameters))
    @assert :ρʸ ∈ fieldnames(eltype(parameters))
    # check that spatial coordinates are part of parameters (only 2D for now)
    @assert :x ∈ fieldnames(eltype(parameters))
    @assert :y ∈ fieldnames(eltype(parameters))

    # compute magnetization at echo times in all voxels
    magnetizationtization = simulate_magnetization(resource, sequence, parameters)

    # apply phase encoding (typically only for Cartesian trajectories)
    phase_encoding!(magnetizationtization, trajectory, parameters)

    # compute signal from (phase-encoded) magnetization at echo times
    signal = magnetization_to_signal(resource, magnetizationtization, parameters, trajectory, coil_sensitivities)

    return signal
end

### NAIVE BUT GENERIC IMPLEMENTATION ###

"""
    magnetization_to_signal(resource, magnetizationtization, parameters, trajectory, coil_sensitivities)

Given the magnetization in all voxels (typically at echo times only), allocate memory for the signal
output on CPU, then loop over all time points `t` and use the (generic) `magnetization_to_signal!`
implementation to compute the signal for that time point.

This loop order is not necessarily optimal (and performance may be) across all trajectories and
computational resources. If a better implementation is available, add new methods to this
function for those specific combinations of resources and trajectories.
"""
function magnetization_to_signal(resource, magnetization, parameters, trajectory, coil_sensitivities)

    signal = _allocate_signal_output(resource, trajectory, coil_sensitivities)

    if resource == CPU1()
        for t in 1:nsamples(trajectory)
            magnetization_to_signal!(signal, t, magnetization, parameters, trajectory, coil_sensitivities)
        end
    elseif resource == CPUThreads()
        Threads.@threads for t in 1:nsamples(trajectory)
            # Different threads compute signals at different timepoints t
            magnetization_to_signal!(signal, t, magnetization, parameters, trajectory, coil_sensitivities)
        end
    elseif resource == CUDALibs()

        # compute nr of threadblocks to be used on GPU
        nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

        # define kernel function to be run by each thread on gpu
        magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coil_sensitivities) = begin

            t = (blockIdx().x - 1) * blockDim().x + threadIdx().x # global time point index

            if t <= nsamples(trajectory)
                magnetization_to_signal!(signal, t, magnetization, parameters, trajectory, coil_sensitivities)
            end
            return nothing
        end

        # launch kernels, threads per block hardcoded for now
        CUDA.@sync begin
            @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK magnetization_to_signal_kernel!(signal, magnetization, parameters, trajectory, coil_sensitivities)
        end
    end

    return signal
end

function magnetization_to_signal(::CPUProcesses, dmagnetizationtization::DArray, dparameters::DArray, trajectory, dcoil_sensitivities::DArray)

    # for some reason, assembling DArrays does not work with vectors but it does
    # with matrices
    vec_to_mat(x::AbstractVector) = reshape(x,length(x),1)
    # start computing local signal on each worker
    dsignal = @sync [@spawnat p vec_to_mat(magnetization_to_signal(CPU1(), localpart(dmagnetizationtization), localpart(dparameters), trajectory, localpart(dcoil_sensitivities))) for p in workers()]
    # assemble new DArray from the arrays on each worker
    dsignal = DArray(permutedims(dsignal))
    # sum results
    dsignal = reduce(+,dsignal,dims=2)
    # Don't convert to a local vector at this point
    return dsignal
end

"""
    magnetization_to_signal!(signal, t, magnetizationtization, parameters, trajectory, coil_sensitivities)

If `to_sample_point` has been defined for the provided trajectory, this (generic but not optimized)
function computes the signal at timepoint `t` for all receive coils. It does so by computing
the readout- and sample indices for the given timepoint `t`, reading in the magnetization at echo
time of the `r`-th readout, using `to_sample_point` to compute the magnetization at the `s`-th sample
index, and then it integrates over all voxels (while scaling the magnetization with the proper coil
sensitivity and proton density).

Better performance can likely be achieved by incorporating more trajectory-specific information
together with different loop orders.
"""

@inline function magnetization_to_signal!(signal, t, magnetization, parameters, trajectory, coil_sensitivities)

    # compute readout and sample indices for time point t
    readout, sample = _get_readout_and_sample_idx(trajectory, t)

    nv = length(parameters) # nr of voxels

    # accumulator for signal at time index t
    s = zero(eltype(signal)) # note that s is an SVector of length (# ncoils)

    for voxel = 1:nv

        # load parameters and spatial coordinates
        p = parameters[voxel]
        # load coil sensitivity for coil i in this voxel (SVector of length (# coils))
        c = coil_sensitivities[voxel]
        # load magnetization in voxel at echo time of the r-th readout
        m = magnetization[readout,voxel]
        # compute magnetization at s-th sample of r-th readout
        mₛ = to_sample_point(m, trajectory, readout, sample, p)
        # add magnetization from this voxel, scaled with proton density
        # and coil sensitivity, to signal accumulator s
        s += mₛ * (complex(p.ρˣ, p.ρʸ) * c)
    end

    # store signal for each coil at time t
    signal[t] = s
    return nothing
end

"""
    _allocate_signal_output(resource, trajectory::AbstractTrajectory, coil_sensitivities)

Allocate an array to store the output of the signal simulation (all readout points,
integrated over all voxels).
"""
function _allocate_signal_output(resource, trajectory::AbstractTrajectory, coil_sensitivities::AbstractVector{T}) where T<:SVector

    ns = nsamples(trajectory)

    if resource == CUDALibs()
        # allocate a CuArray of zeros on GPU
        output = CUDA.zeros(T, ns)
    elseif resource == CPUProcesses()
        # allocate a DArray of zeros
        nw = nworkers()
        output = dzeros(T, (ns, nw), workers(), (1, nw))
    elseif resource ∈ (CPU1(), CPUThreads())
        # allocate an Array of zeros on the local CPU
        output = zeros(T, ns)
    end

    return output
end

### Convenience functions

# When coil sensitivities are not provided, use a single coil with sensitivity = 1 everywhere
function simulate_signal(resource, sequence::BlochSimulator, parameters::AbstractArray{<:AbstractTissueParameters{N,T}}, trajectory) where {N,T}

    # use one coil with sensitivity 1 everywhere
    coil_sensitivities = ones(Complex{T},length(parameters)) .|> SVector

    # send to GPU if necessary
    if resource == CUDALibs()
        coil_sensitivities = gpu(coil_sensitivities)
    elseif resource == CPUProcesses()
        coil_sensitivities = distribute(coil_sensitivities)
    end

    # simulate signal and use only because there's only one coil anyway
    signal = simulate_signal(resource, sequence, parameters, trajectory, coil_sensitivities)

    return signal
end