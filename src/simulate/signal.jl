### Signal simulation code

"""
    simulate(resource, sequence, parameters, trajectory, coil_sensitivities)

Simulate the MR signal at timepoint `t` from coil `i` as: `sᵢ(t) = ∑ⱼ cᵢⱼρⱼmⱼ(t)`,
where `cᵢⱼ`is the coil sensitivity of coil `i` at position of voxel `j`,
`ρⱼ` is the proton density of voxel `j` and `mⱼ(t)` the (normalized) transverse magnetization
in voxel `j` obtained through Bloch simulations.

# Arguments
- `resource::ComputationalResource`: Either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()`
- `sequence::BlochSimulator`: Custom sequence struct
- `parameters::AbstractVector{<:AbstractTissueParameters}`: Vector with tissue parameters for each voxel
- `trajectory::AbstractTrajectory`: Custom trajectory struct
- `coil_sensitivities::AbstractVector{<:SVector{ncoils}}`: Vector with `ncoils` coil sensitivities for each voxel

# Returns
- `signal::Vector{<:SVector{ncoils}}`: Simulated MR signal for the `sequence` and `trajectory`. 
At each timepoint, the signal for each of the `ncoils` is stored.
"""
function simulate(resource, sequence, parameters, trajectory, coil_sensitivities)

    @assert length(parameters) == length(coil_sensitivities)
    @assert :ρˣ ∈ fieldnames(eltype(parameters))
    @assert :ρʸ ∈ fieldnames(eltype(parameters))
    @assert :x ∈ fieldnames(eltype(parameters))
    @assert :y ∈ fieldnames(eltype(parameters))

    # compute magnetization at echo times in all voxels
    echos = simulate(resource, sequence, parameters)

    # allocate output for signal
    signal = _allocate_output(resource, trajectory, coil_sensitivities)

    # expand readouts and perform volume integration
    _simulate!(signal, resource, echos, parameters, trajectory, coil_sensitivities)

    return signal
end

"""
    _allocate_output(resource, trajectory::AbstractTrajectory, coil_sensitivities)

Allocate an array to store the output of the Bloch simulations (all readout points,
integrated over all voxels) to be performed with the `sequence`.
"""
function _allocate_output(resource, trajectory::AbstractTrajectory, coil_sensitivities)

    type = eltype(coil_sensitivities)

    if resource == CUDALibs()
        # allocate a CuArray of zeros on GPU
        output = CUDA.zeros(type, nsamples(trajectory))
    elseif resource == CPUProcesses()
        # allocate a DArray of zeros
        output = dzeros(type, (nsamples(trajectory),nworkers()), workers(), (1,nworkers()))
    else
        # allocate an Array of zeros on the local CPU
        output = zeros(type, nsamples(trajectory))
    end

    return output
end


### SERIAL FUNCTIONS ###

"""
    _simulate(::CPU1, echos, parameters, trajectory, coil_sensitivities)

Perform signal simulations on CPU. Outer loop over the number of voxels,
inner loop over time.
"""
function _simulate!(signal, ::CPU1, echos, parameters, trajectory, coil_sensitivities)

    @inbounds for v ∈ eachindex(parameters)

        echosᵥ = selectdim(echos, ndims(echos), v) |> vec

        # loop over readouts/echos
        for readout in eachindex(echosᵥ)
            expand_readout_and_sample!(signal, readout, echosᵥ[readout],
                trajectory, parameters[v], coil_sensitivities[v])
        end
    end

    return reshape(signal,:,1) # reshape to matrix because DistributedArrays does not work with vectors
end

# function _simulate!(signal, ::CPU1, echos, parameters, trajectory, coil_sensitivities)

#     for t in 1:nsamples(trajectory)

#         r,s = _get_readout_and_sample_idx(trajectory, t)
#         nv = length(parameters) # nr of voxels

#         # accumulator for signal at time index t
#         S = zero(eltype(signal))

#         @inbounds for voxel = 1:nv

#             # load parameters and spatial coordinates
#             p = parameters[voxel]
#             C = coil_sensitivities[voxel]
#             # load magnetization in voxel at echo time of the r-th readout
#             m = echos[r,voxel]
#             # compute magnetization at s-th sample of r-th readout
#             mₛ = to_sample_point(m, trajectory, r, s, p)
#             # store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
#             S += (mₛ * complex(p.ρˣ, p.ρʸ)) * C
#         end

#         # # store signal at time t
#         @inbounds signal[t] = S
#     end
# end

# # # Given a DistributedArray of echos, compute Mv, Mᵀv or M

# function _simulate!(dsignal, ::CPUProcesses, dechos, dparameters, trajectory, dcoil_sensitivities)

#     dv = distribute(v)
#     # start computing local signal on each worker
#     dMv = [@spawnat p _simulate!(localpart(dsignal), CPU1(), localpart(dechos), localpart(dparameters), localpart(dcoordinates), trajectory, localpart(dcoil_sensitivities)) for p in workers()]
#     # sync
#     dMv = DArray(permutedims(dMv))
#     # sum results
#     dMv = reduce(+,dMv,dims=2) # How exactly does this work?
#     # Convert to a local vector
#     # dMv = vec(convert(Array{Complex{Float64},2}, dMv))
#     dMv = vec(convert(Array, dMv))

#     return dMv
# end

# # If parameters are provided as a regular array instead of a DistributedArray, distribute them first
# _simulate!(dsignal::DArray, resource::CPUProcesses, dechos, parameters, coil_sensitivities, trajectory) = _simulate!(doutput, resource, dechos, distribute(parameters), trajectory, distribute(coil_sensitivities))


"""
    _simulate!(signal, ::CUDALibs, sequence, parameters, trajectory, coil_sensitivities)

Run Bloch simulations on a CUDA compatible GPU. Assumes signal, sequence, parameters, trajectory
and coil_sensitivities are already transferred to the GPU with the `gpu` function. Each thread gets assigned
a time index and then it loops over the voxels.
"""
function _simulate!(signal, ::CUDALibs, echos, parameters, trajectory, coil_sensitivities)

    # threads per block hardcoded for now
    # compute nr of threadblocks to be used on GPU
    nr_blocks = cld(nsamples(trajectory), THREADS_PER_BLOCK)

    # launch kernels
    CUDA.@sync begin
        @cuda blocks=nr_blocks threads=THREADS_PER_BLOCK cuda_simulate_kernel!(signal, echos, parameters, trajectory, coil_sensitivities)
    end

    return nothing
end

"""
    cuda_simulate_kernel!(output, sequence, parameters, trajectory, coil_sensitivities)

Kernel function that gets launched by each thread on the GPU. Each thread performs Bloch simulations in a single voxel.
"""
function cuda_simulate_kernel!(signal, echos, parameters, trajectory, coil_sensitivities)

    t = global_id() # global time point index

    if t <= nsamples(trajectory)

        r,s = _get_readout_and_sample_idx(trajectory, t)
        nv = length(parameters) # nr of voxels

        # accumulator for signal at time index t
        S = zero(eltype(signal))

        @inbounds for voxel = 1:nv

            # load parameters and spatial coordinates
            p = parameters[voxel]
            C = coil_sensitivities[voxel]
            # load magnetization in voxel at echo time of the r-th readout
            m = echos[r,voxel]
            # compute magnetization at s-th sample of r-th readout
            mₛ = to_sample_point(m, trajectory, r, s, p)
            # store magnetization from this voxel, scaled with v (~ proton density) and C in accumulator
            S += (mₛ * complex(p.ρˣ, p.ρʸ)) * C
        end

        # # store signal at time t
        @inbounds signal[t] = S

    end

    nothing
end

# conventient function to simualte without coil sensitivities
function simulate(resource, sequence, parameters, trajectory)
    
    # use one coil with sensitivity 1 everywhere
    coil_sensitivities = SVector{1}.(complex.(ones(size(parameters))))
    
    # send to GPU if necessary
    if resource == CUDALibs()
        coil_sensitivities = gpu(coil_sensitivities)
    end
    
    # simulate signal and use only because there's only one coil anyway 
    signal = simulate(resource, sequence, parameters, trajectory, coil_sensitivities) .|> only

    return signal
end