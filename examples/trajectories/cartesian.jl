### Type definition

"""
    SpokesTrajectory{T} <: AbstractTrajectory{T}

Typical Cartesian and radial trajectories have a lot in common: a readout
can be described by a starting point in k-space and a Δk per sample point.
To avoid code repetition, both type of trajectories are made a subtype of
SpokesTrajectory such that some methods that would be the same for both
trajectories otherwise are written for SpokesTrajectory instead.
"""
abstract type SpokesTrajectory{T} <: AbstractTrajectory{T} end # Cartesian & Radial

export SpokesTrajectory

"""
    CartesianTrajectory{T,I,U,V} <: SpokesTrajectory{T}

Struct that is used to implement a typical Cartesian gradient trajectory.
The trajectory is described in a compact fashion by only storing
the starting position in k-space (`k_start_readout`) for each readout
as well as the step in k-space per readout point `Δk_adc`.

Note that CartesianTrajectory and RadialTrajectory are essentially the same in when
using when using this compact description. A SpokesTrajectory struct is therefore
defined as a supertype of both and methods are defined for SpokesTrajectory instead
to avoid code repetition.

The type parameters are intentionally left vague. The `J`, for example, may be an
integer for sequences where each readout has the same number of samples, but for
sequences with different numbers of samples per readout it may be a vector of integers.

# Fields
- `nreadouts::I`: The total number of readouts for this trajectory
- `nsamplesperreadout::I`: The total number of samples per readout
- `Δt::T`: Time between sample points
- `k_start_readout::U`: Starting position in k-space for each readout
- `Δk_adc::U`: k-space step Δkₓ per sample point (same for all readouts)
- `py::V`: Phase encoding index for each readout
"""
struct CartesianTrajectory{T<:Real,I<:Integer,U<:AbstractVector,V<:AbstractVector} <: SpokesTrajectory{T}
    nreadouts::I
    nsamplesperreadout::I
    Δt::T # time between sample points
    k_start_readout::U # starting position in k-space for each readout
    Δk_adc::T # Δk during ADC for each readout
    py::V # phase encoding order, not needed but nice to store
end

@functor CartesianTrajectory
@adapt_structure CartesianTrajectory

export CartesianTrajectory

### Interface requirements

@inline nreadouts(t::SpokesTrajectory) = t.nreadouts
@inline nsamplesperreadout(t::SpokesTrajectory, readout) = t.nsamplesperreadout

function phase_encoding!(magnetization, trajectory::CartesianTrajectory, parameters)
    y  = map(p->p.y, parameters) |> vec
    kʸ = imag.(trajectory.k_start_readout)
    @. echos *= exp(im * kʸ * y')
    return nothing
end

# perhaps do this with metaprogramming instead (iteratate over all subtypes of AbstractTrajectory)
function phase_encoding!(magnetization::DArray, trajectory::CartesianTrajectory, parameters::DArray)

    @sync for p in workers()
        @async begin
            @spawnat p phase_encoding!(localpart(magnetization), trajectory, localpart(parameters))
        end
    end

    return nothing
end



@inline function to_sample_point(mₑ, trajectory::CartesianTrajectory, readout_idx, sample_idx, p)

    # Note that m has already been phase-encoded

    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    x = p.x
    # There are ns samples per readout, echo time is assumed to occur
    # at index (ns÷2)+1. Now compute sample index relative to the echo time
    s = sample_idx - ((ns ÷ 2) + 1)
    # Apply readout gradient, T₂ decay and B₀ rotation
    E₂ = exp(-Δt * s * R₂)
    θ = Δkₓ * x
    hasB₀(p) && (θ += π * p.B₀ * Δt * 2)
    E₂eⁱᶿ = E₂ * exp(im * s * θ)
    mₛ = E₂eⁱᶿ * mₑ

    return mₛ
end

### Utility functions (not used in signal simulations)

# Convenience constructor to quickly generate Cartesian trajectory
# with nr readouts and ns samples per readout
CartesianTrajectory(nr, ns) = CartesianTrajectory(nr, ns, 10^-5, complex.(rand(nr)), rand(), rand(Int, nr))

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::CartesianTrajectory, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.k_start_readout[idx], tr.Δk_adc, tr.py[idx])

# Nicer printing in REPL
Base.show(io::IO, tr::CartesianTrajectory) = begin
    println("")
    println(io, "Cartesian trajectory")
    println(io, "nreadouts:          ", tr.nreadouts)
    println(io, "nsamplesperreadout: ", tr.nsamplesperreadout)
    println(io, "Δt:                 ", tr.Δt)
    println(io, "k_start_readout:    ", typeof(tr.k_start_readout))
    println(io, "Δk_adc:             ", tr.Δk_adc)
    println(io, "py:                 ", typeof(tr.py))
end

"""
    kspace_coordinates(tr::CartesianTrajectory)

Return matrix (nrsamplesperreadout, nrreadouts) with kspace coordinates for the trajectory. Needed for nuFFT reconstructions.
"""
function kspace_coordinates(tr::CartesianTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout
    k = [tr.k_start_readout[r] + (s - 1) * tr.Δk_adc for s in 1:ns, r in 1:nr]

    return k
end

"""
    sampling_mask(tr::CartesianTrajectory)

For undersampled Cartesian trajectories, the gradient trajectory can also be described by a sampling mask.
"""
function sampling_mask(tr::CartesianTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout

    min_py = minimum(tr.py)

    sampling_mask = [CartesianIndices((1:ns, tr.py[r]-min_py+1:tr.py[r]-min_py+1)) for r in 1:nr]

    return sampling_mask
end
