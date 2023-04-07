export BlochSimulator, IsochromatSimulator, EPGSimulator

### Type definitions

"""
    BlochSimulator{T}

The abstract type of which all sequence simulators will be a subtype. The parameter `T`
should be a number type (e.g. `Float64`, `Float32`) and the tissueparameters that are used as
input to the simulator should have the same number type. By convention, a BlochSimulator
will be used to simulate magnetization at echo times only without taking into account
spatial encoding gradients (i.e. readout or phase encoding gradients). To simulate the
magnetization at other readout times, including phase from spatial encoding gradients,
an `AbstractTrajectory` will be needed as well.

To make a simulator for a particular pulse sequence:
1. Make a struct that's a subtype of either `IsochromatSimulator` or `EPGSimulator`.
    The struct will hold parameters that are necessary for performing the simulations.
2. Add a method to `simulate!` that implements the pulse sequence. For both performance and GPU compatibility,
    make sure that simulate! does not do any heap allocations. Examples for `pSSFP` and `FISP`
    sequences are found in `src/sequences`.
3. Add methods to `output_eltype` and `output_size` that are used to allocate an output array within the simulate function.

4. [Optional] Add a method to show for nicer printing of the sequence in the REPL
5. [Optional] Add a method to getindex to easily reduce the length of the sequence
6. [Optional] Add a constructor for the struct that takes in data from Matlab or
    something else and assembles the struct

**IMPORTANT**

The `simulate!` functions (which dispatch on the provided sequence) are assumed to be type-stable and non-allocating
Should be possible to achieve when using functions from `operators/epg.jl`` and `operators/isochromat.jl` and a properly
parametrized sequence struct.
"""
abstract type BlochSimulator{T} end

# In this package we distinguish two different kind of Bloch simulators:
# 1. Simulators based on spin isochromat model (IsochromatSimulator)
# 2. Simulators based on extended phase graph model (EPGSimulator)

"""
    IsochromatSimulator{T} <: BlochSimulator{T}

Abstract type of which all sequence simulators that are based on the isochromat
model will be a subtype. The parameter `T` should be a number type (e.g. `Float64`, `Float32`)
and the tissueparameters that are used as input to the simulator should have the same number type.
"""
abstract type IsochromatSimulator{T} <: BlochSimulator{T} end

"""
    EPGSimulator{T,Ns} <: BlochSimulator{T}

Abstract type of which all sequence simulators that are based on the EPG
model will be a subtype. The parameter `T` should be a number type (e.g. `Float64`, `Float32`)
and the tissueparameters that are used as input to the simulator should have the same number type.
The parameter `Ns` corresponds to the maximum order of configuration states that are tracked
in the simulations.
"""
abstract type EPGSimulator{T,Ns} <: BlochSimulator{T} end

# The type T of the simulator is supposed to be the same as the type of the input parameters
# (e.g. T₁,T₂, etc) and is useful for dispatching in some functions. For the EPG model we need to
# keep track of a certain number of configuration states Ns. This number Ns is made part of the type so
# that it is known at compile time (which is needed because we will be working with StaticArrays)

# Building blocks for sequence simulators based on isochromat model are contained in src/operators/isochromat.jl
# Building blocks for sequence simulators based on EPG model are contained in src/operators/epg.jl
# These "building blocks" are things like rotation, decay, regrowth.

### Functions that must be implemented for each BlochSimulator

"""
    output_eltype(::BlochSimulator)

For each `<:BlochSimulator`, a method should be added to this function that
specifies the output type of the simulation. For MR signal simulation, this
is typically a complex number representing the transverse magnetization. For
other types of simulations, one may want to retrieve the x,y and z components of
an isochromat as output (implemented as a `FieldVector` perhaps) or state configuration
matrices `Ω`.
"""
function output_eltype(::BlochSimulator)
    @warn "Must implement output_eltype" 
end

"""
    output_dimensions(::BlochSimulator)

For each `<:BlochSimulator`, a method should be added to this function that
specifies the output size of the simulation for a *single* `::AbstractTissueParameters`.
"""
function output_dimensions(::BlochSimulator)
    @warn "Must implement output_dimensions"
end

"""
    simulate!(echos, sequence::BlochSimulator, state, p::AbstractTissueParameters) end

For each `<:BlochSimulator`, a method should be added to this function that
implements the actual pulse sequence using information contained in the sequence struct
together with the operators from `src/operators/{isochromat,epg}.jl`. For performance reasons
as well as GPU compatibility it is important that the implementation is type-stable and
non-allocating.

# Arguments
- `echos`: Pre-allocated array with `size(echos) = output_dimensions(sequence)` and
    `eltype(echos) = output_eltype(sequence)` to store the output of the simulation.
- `sequence`: Sequence struct containing fields that are used to implement the actual pulse
    sequence.
- `state`: Either an `Isochromat` or `EPGStates`, depending on which model is used.
- `p`: Custom struct (`<:AbstractTissueParameters`) containing input parameters
    to the simulation (e.g. `T₁T₂`)

"""
function simulate!(echos, sequence::BlochSimulator, state, p::AbstractTissueParameters) 
    @warn "Must implement simulate!"
end

### Optional functions

# Little bit nicer printing in REPL
Base.show(io::IO, sequence::BlochSimulator) = print(io, dump(sequence));