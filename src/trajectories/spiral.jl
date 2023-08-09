# NEEDS TESTING

### Type definition

"""
    SpiralTrajectory{T,I,U,V} <: AbstractTrajectory

NOT TESTED

# Fields
- `nreadouts::I`: The total number of readouts for this trajectory
- `nsamplesperreadout::I`: The total number of samples per readout. Assumed to be constant
- `Δt::T`: Time between sample points
- `Δk⁰::U`: Δk's for all samples during the first spiral readout
- `φ::U`: Angles with which the spiral is rotated compared to the first spiral


"""
struct SpiralTrajectory{T,I,U,V} <: AbstractTrajectory
    nreadouts::I
    nsamplesperreadout::I
    Δt::T # time between sample points
    Δk⁰::U # Δk during ADC for the first spiral
    φ::V # angles with which the spiral is rotated compared to the first spiral
end

@functor SpiralTrajectory
@adapt_structure SpiralTrajectory

### Interface requirements

### Utility functions (not used in signal simulations)

# Add method to getindex to reduce sequence length with convenient syntax (e.g. trajectory[idx] where idx is a range like 1:nr_of_readouts)
Base.getindex(tr::SpiralTrajectory, idx) = typeof(tr)(length(idx), tr.nsamplesperreadout, tr.Δt, tr.Δk⁰[idx], tr.φ[idx])

# Nicer printing in REPL
Base.show(io::IO, tr::SpiralTrajectory) = begin
    println("")
    println(io, "Spiral trajectory")
    println(io, "nreadouts:          ", tr.nreadouts)
    println(io, "nsamplesperreadout: ", tr.nsamplesperreadout)
    println(io, "Δt:                 ", tr.Δt)
    println(io, "Δk⁰:                ", typeof(tr.Δk⁰))
    println(io, "φ:                  ", typeof(tr.φ))
end

