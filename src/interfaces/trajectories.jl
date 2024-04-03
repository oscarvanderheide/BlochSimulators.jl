### Type definition

"""
    AbstractTrajectory{T}

The abstract type of which all gradient trajectories will be a subtype. The subtypes
should contain fields that can describe the full trajectory during a sequence. The type T refers to the precision of the floating point values within the trajectory struct.
"""
abstract type AbstractTrajectory{T} end

export AbstractTrajectory

### Functions that must be implemented for each trajectory.

"""
    phase_encoding!(magnetization, trajectory::AbstractTrajectory, parameters)

For each `::AbstractTrajectory`, a method should be added to this function if
it does any kind of phase encoding (so far Cartesian only).
"""
function phase_encoding!(magnetization, trajectory::AbstractTrajectory, parameters)
    println("No phase encoding for this trajectory")
    nothing
end

"""
    nreadouts(::AbstractTrajectory)

For each `::AbstractTrajectory`, a method should be added to this function that
specifies how many readouts the trajectory consists of.
"""
function nreadouts(::AbstractTrajectory)
    @warn "Must implement nreadouts"
end


"""
    nsamplesperreadout(::AbstractTrajectory, readout_idx)

For each `::AbstractTrajectory`, a method should be added to this function that
specifies how many samples in total are acquired during the trajectory.
"""
function nsamplesperreadout(::AbstractTrajectory, readout_idx)
    @warn "Must implement nsamplesperreadout"
end

"""
    to_sample_point(m, trajectory, readout_idx, sample_idx, parameters)

For each ::AbstractTrajectory, a method should be added to this function that,
given the magnetization `m` at the readout with index `readout_idx`, it computes the magnetization at
the readout point with index `sample_idx` (by applying spatial encoding gradients, T₂ decay,
B₀ rotation, etc...) based on the `trajectory` and `parameters`.

Arguments
- `m`:              Magnetization at the echo with index `readout_idx`.
- `trajectory`:     Trajectory struct containing fields that are used to compute the magnetization at other readout times,
                    including the effects of spatial encoding gradients.
- `readout_idx`:    Index that corresponds to the current readout.
- `sample_idx`:     Index for the desired sample during this readout.
- `parameters`:     Tissue parameters of current voxel, including spatial coordinates.

Output:
- mₛ:   Magnetization at sample with index `sample_idx`
"""
function to_sample_point(m, trajectory, readout_idx, sample_idx, parameters)
    @warn "Must implement to_sample_point"
end

### Derived functions

"""
    nsamples(trajectory::AbstractTrajectory)

Determines the total number of samples acquired with the trajectory.
Requires `nreadouts` and `nsamplesperreadout` to be implemented.
"""
function nsamples(trajectory::AbstractTrajectory)
    ns = 0
    for r in 1:nreadouts(trajectory)
        ns += nsamplesperreadout(trajectory, r)
    end
    return ns

    # for some strange reason the oneliner
    # sum(r -> nsamplesperreadout(trajectory,r), 1:nreadouts(trajectory))
    # does not work within CUDA kernels
end

"""
    _get_readout_and_sample_idx(trajectory, t)

Given time index `t`, compute the associated readout and sample indices `(r,s)`.
For trajectories where each readout has the same length, this is equivalent to
`r,s = fld1(t,ns), mod1(t,ns)` with ns being the (constant) number of samples per readout.
"""
@inline function _get_readout_and_sample_idx(trajectory, t)
    # determine readout and sample point associated with current thread
    found_r = false

    r = 1 # index of readout associated with time index t
    readout_start = 1

    while !found_r # && r <= nreadouts(trajectory)
        readout_end = readout_start + nsamplesperreadout(trajectory, r) - 1

        if readout_start <= t <= readout_end
            found_r = true
        else
            r += 1
            readout_start = readout_end + 1
        end
    end

    s = t - readout_start + 1 # index of sample within readout
    return r, s
end

export nsamples, nreadouts, nsamplesperreadout