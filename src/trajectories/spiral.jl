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

# # given magnetization at echo times, simulate the whole readouts
# @inline function echotops_to_full_readouts!(buffer,echos,voxel,trajectory::SpiralTrajectory,x,y,T₂,B₀)

#     Δt = trajectory.Δt
#     ns = trajectory.nsamplesperreadout # nr of samples per readout
#     nr = trajectory.nreadouts # nr of readouts
#     φ   = @views trajectory.φ
#     Δk⁰ = @views trajectory.Δk⁰

#     # decay per Δt of readout
#     R₂ = inv(T₂)
#     E₂ = exp(-Δt*R₂)
#     # B₀ rotation for Δt of readout
#     Δt2πB₀ = Δt*2π*B₀

#     E₂eⁱᶿ = similar(Δk⁰)

#     @inbounds for r = 1:nr

#         # for the first spiral, the Δk's are stored in Δk⁰
#         # for the other spirals, we need to rotate them with angle φ
#         eⁱᵠ = exp(im*φ[r])

#         # load magnetization in voxel at echo time of the r-th readout
#         m = echos[r,voxel]

#         @simd for s = 1:ns
#             buffer[s,r] = m
#             # rotate and decay to go to next sample point:
#             Δk = eⁱᵠ * Δk⁰[s]
#             # compute rotation per Δt of readout due to gradients and B₀
#             θ = (Δk.re * x + Δk.im * y) + Δt2πB₀
#             # combine decay and rotation in complex multiplicator
#             E₂eⁱᶿ = E₂ * cis(im*θ) # cis(θ) is slightly faster than exp(im*θ)
#             m  = m * E₂eⁱᶿ
#         end # loop over samples per readout
#     end # loop over readouts
# end

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

