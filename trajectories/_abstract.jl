"""
    SpokesTrajectory{T} <: AbstractTrajectory{T}

Typical Cartesian and radial trajectories have a lot in common: a readout
can be described by a starting point in k-space and a Δk per sample point.
To avoid code repetition, both type of trajectories are made a subtype of
SpokesTrajectory such that some methods that would be the same for both
trajectories otherwise are written for SpokesTrajectory instead.
"""
abstract type SpokesTrajectory{T} <: AbstractTrajectory{T} end

abstract type CartesianTrajectory{T} <: SpokesTrajectory{T} end
abstract type RadialTrajectory{T} <: SpokesTrajectory{T} end

export SpokesTrajectory, CartesianTrajectory, RadialTrajectory

# Interface requirements that are common for subtypes of SpokesTrajectory
@inline nreadouts(t::SpokesTrajectory) = t.nreadouts
@inline nsamplesperreadout(t::SpokesTrajectory) = t.nsamplesperreadout
@inline nsamplesperreadout(t::SpokesTrajectory, readout) = t.nsamplesperreadout
