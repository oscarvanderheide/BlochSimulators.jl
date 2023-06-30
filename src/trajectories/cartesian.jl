### Type definition

"""
    SpokesTrajectory <: AbstractTrajectory

Typical Cartesian and radial trajectories have a lot in common: a readout
can be described by a starting point in k-space and a Δk per sample point.
To avoid code repetition, both type of trajectories are made a subtype of
SpokesTrajectory s.t. and some methods (that would be the same for both
trajectories otherwise) are written for SpokesTrajectory instead.
"""
abstract type SpokesTrajectory <: AbstractTrajectory end # Cartesian & Radial

export SpokesTrajectory

"""
    CartesianTrajectory{T,I,J,U,V} <: SpokesTrajectory

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
- `nsamplesperreadout::J`: The total number of samples per readout
- `Δt::T`: Time between sample points
- `k_start_readout::U`: Starting position in k-space for each readout
- `Δk_adc::U`: k-space step Δk between each readout
- `py::V`: Phase encoding index for each readout
"""
struct CartesianTrajectory{T,I,J,U,V} <: SpokesTrajectory
    nreadouts::I
    nsamplesperreadout::J
    Δt::T # time between sample points
    k_start_readout::U # starting position in k-space for each readout
    Δk_adc::U # Δk during ADC for each readout
    py::V # phase encoding order, not needed but nice to store
end

@functor CartesianTrajectory
@adapt_structure CartesianTrajectory

export CartesianTrajectory

### Interface requirements

@inline nreadouts(t::SpokesTrajectory) = t.nreadouts
@inline nsamplesperreadout(t::SpokesTrajectory, readout) = t.nsamplesperreadout

function expand_readout_and_sample!(output, readout_idx, m, trajectory::SpokesTrajectory, p, C)

    # get index associated with first sample of current readout
    sample_idx = 1 + sum(r -> nsamplesperreadout(trajectory,r), 1:readout_idx-1, init=0)
    # time between sample points within readout
    Δt = trajectory.Δt
    # starting point in k-space for current readout
    k⁰ = trajectory.k_start_readout[readout_idx]
    # k-space step per sample point for current readout
    Δk = trajectory.Δk_adc[readout_idx]
    # nr of samples for current readout
    ns = trajectory.nsamplesperreadout
    # linear scaling factors (coil sensitivity, proton density)
    ρC = complex(p.ρˣ, p.ρʸ) * C
    # decay per Δt of readout
    R₂ = inv(p.T₂)
    E₂ = exp(-Δt*R₂)
    # compute rotation per Δt of readout due to gradients and B₀
    θ = (Δk.re * p.x + Δk.im * p.y) + (hasB₀(p) ? (2π * p.B₀ * Δt) : 0)
    # combine decay and rotation in complex multiplicator
    E₂eⁱᶿ = E₂ * exp(im*θ)
    # go back in time to the start of the readout
    m = rewind(m, R₂, (ns/2)*Δt, p)
    # apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
    m = prephaser(m,k⁰.re, k⁰.im, p.x, p.y)

    for s in 1:ns # loop over samples per readout
        # scale magnetization with proton density and coil sensitivities and sample
        output[sample_idx+s-1] += m * ρC
        # rotate and decay to go to next sample point
        m  = m * E₂eⁱᶿ
    end

    return nothing
end

# given magnetization at echo time,
# undo T2 decay and B0 phase that happened between start readout and echo time
@inline function rewind(m,R₂,Δt,p::AbstractTissueParameters)
    # m is magnetization at echo time
    # undo T2 decay and B0 phase that happened between start readout and echo time
    arg = Δt*R₂
    hasB₀(p) && (arg -= im*Δt*2*π*p.B₀)
    return m * exp(arg)
end

# apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
@inline function prephaser(m,kˣ,kʸ,x,y)
    return m * exp(im * (kˣ * x + kʸ * y))
end

@inline function to_sample_point(m, trajectory::SpokesTrajectory, readout_idx, sample_idx, p)

    # read in constants
    R₂ = inv(p.T₂)
    ns = trajectory.nsamplesperreadout
    Δt = trajectory.Δt
    k⁰ = trajectory.k_start_readout[readout_idx]
    Δk = trajectory.Δk_adc[readout_idx]
    x,y = p.x, p.y

    # go back in time to the start of the readout by undoing T₂ decay and B₀ rotation
    m = rewind(m, R₂, (ns÷2)*Δt, p)
    # apply gradient prephaser (i.e. phase encoding + readout prephaser for Cartesian)
    m = prephaser(m, k⁰.re, k⁰.im, x, y)
    # apply readout gradient, T₂ decay and B₀ rotation
    E₂ = exp(-Δt*(sample_idx-1)*R₂)
    θ = Δk.re * x + Δk.im * y
    hasB₀(p) && (θ += π*p.B₀*Δt*2)
    θ *= (sample_idx-1)
    E₂eⁱᶿ = E₂ * exp(im*θ)
    mₛ = E₂eⁱᶿ * m

    return mₛ
end

### Utility functions (not used in signal simulations)

# Convenience constructor to quickly generate Cartesian trajectory
# with nr readouts and ns samples per readout
CartesianTrajectory(nr,ns) = CartesianTrajectory(nr,ns,10^-5,complex.(rand(nr)), complex.(rand(nr)),rand(Int,nr))

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::CartesianTrajectory, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.k_start_readout[idx], tr.Δk_adc[idx], tr.py[idx])

# Nicer printing in REPL
Base.show(io::IO, tr::CartesianTrajectory) = begin
    println("")
    println(io, "Cartesian trajectory")
    println(io, "nreadouts:          ", tr.nreadouts)
    println(io, "nsamplesperreadout: ", tr.nsamplesperreadout)
    println(io, "Δt:                 ", tr.Δt)
    println(io, "k_start_readout:    ", typeof(tr.k_start_readout))
    println(io, "Δk_adc:             ", typeof(tr.Δk_adc))
    println(io, "py:                 ", typeof(tr.py))
end

"""
    kspace_coordinates(tr::SpokesTrajectory)

Return matrix (nrsamplesperreadout, nrreadouts) with kspace coordinates for the trajectory. Needed for nuFFT reconstructions.
"""
function kspace_coordinates(tr::SpokesTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout
    k = [tr.k_start_readout[r] + (s-1)*tr.Δk_adc[r] for s in 1:ns, r in 1:nr]

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

    sampling_mask = [ CartesianIndices((1:ns, tr.py[r] - min_py + 1:tr.py[r] - min_py + 1)) for r in 1:nr]

    return sampling_mask
end
