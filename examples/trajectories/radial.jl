### Type definition

"""
    RadialTrajectory{T,I,U,V} <: SpokesTrajectory{T}

Struct that is used to implement a typical radial gradient trajectory.
The trajectory can is described in a compact fashion by only storing
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
- `Δk_adc::U`: k-space step Δk between each readout
- `φ::V`: Radial angle for each readout
"""
struct RadialTrajectory{T<:Real,I<:Integer,U,V} <: SpokesTrajectory{T}
    nreadouts::I
    nsamplesperreadout::I
    Δt::T # time between sample points
    k_start_readout::U # starting position in k-space for each readout
    Δk_adc::U # Δk during ADC for each readout
    φ::V # radial angles, not needed but nice to store
end

@functor RadialTrajectory
@adapt_structure RadialTrajectory

export RadialTrajectory

### Interface requirements

@inline function to_sample_point(mₑ, trajectory::RadialTrajectory, readout_idx, sample_idx, p)

    # Note that m has already been phase-encoded

    # Read in constants
    R₂ = inv(p.T₂)
    ns = nsamplesperreadout(trajectory, readout_idx)
    Δt = trajectory.Δt
    @inbounds k⁰ = trajectory.k_start_readout[readout_idx]
    @inbounds Δk = trajectory.Δk_adc[readout_idx]
    x, y = p.x, p.y

    # There are ns samples per readout, echo time is assumed to occur
    # at index (ns÷2)+1. Now compute sample index relative to the echo time
    s = sample_idx - ((ns ÷ 2) + 1)

    # Apply readout gradient, T₂ decay and B₀ rotation
    E₂ = exp(-Δt * s * R₂)
    θ = Δk.re * x + Δk.im * y
    hasB₀(p) && (θ += π * p.B₀ * Δt * 2)
    E₂eⁱᶿ = E₂ * exp(im * s * θ)
    mₛ = E₂eⁱᶿ * mₑ

    return mₛ
end

### Utility functions (not used in signal simulations)

# Convenience constructor to quickly generate Cartesian trajectory
# with nr readouts and ns samples per readout
RadialTrajectory(nr, ns) = RadialTrajectory(nr, ns, 10^-5, complex.(rand(nr)), complex.(rand(nr)), rand(nr))

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::RadialTrajectory, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.k_start_readout[idx], tr.Δk_adc[idx], tr.φ[idx])

# Nicer printing in REPL
Base.show(io::IO, tr::RadialTrajectory) = begin
    println("")
    println(io, "Radial trajectory")
    println(io, "nreadouts:          ", tr.nreadouts)
    println(io, "nsamplesperreadout: ", tr.nsamplesperreadout)
    println(io, "Δt:                 ", tr.Δt)
    println(io, "k_start_readout:    ", typeof(tr.k_start_readout))
    println(io, "Δk_adc:             ", typeof(tr.Δk_adc))
    println(io, "φ:                  ", typeof(tr.φ))
end

"""
    add_gradient_delay!(tr::RadialTrajectory, S)

Apply gradient delay to radial trajectory in in-place fashion. The delay is described by the 2x2 matrix S and is assumed to influence
the start of the readout only, not the readout direction.
"""
function add_gradient_delay!(tr::RadialTrajectory, S)

    for r = 1:tr.nreadouts
        # normal vector in (units of Δk_adc)
        n = SVector{2}(tr.Δk_adc[r].re, tr.Δk_adc[r].im)
        # compute delay for spoke in this direction using S matrix
        delay = S * n
        # adjust starting position in k-space of spoke
        tr.k_start_readout[r] += complex(delay...)
    end
    return tr
end

add_gradient_delay(tr::RadialTrajectory, S) = add_gradient_delay!(deepcopy(tr), S)

"""
    kspace_coordinates(tr::RadialTrajectory)

Return matrix (nrsamplesperreadout, nrreadouts) with kspace coordinates for the trajectory. Needed for nuFFT reconstructions.
"""
function kspace_coordinates(tr::RadialTrajectory)

    nr = tr.nreadouts
    ns = tr.nsamplesperreadout
    k = [tr.k_start_readout[r] + (s - 1) * tr.Δk_adc[r] for s in 1:ns, r in 1:nr]

    return k
end