module BlochSimulators

# Load external packages
using ComputationalResources
using CUDA
using Distributed
using DistributedArrays
using Functors
using InteractiveUtils # needed for subtypes function
using LinearAlgebra
using OffsetArrays
using StaticArrays
using StructArrays
using Unitful
using Unitless

import Adapt: adapt, adapt_storage, @adapt_structure # to allow custom Structs of non-isbits type to be used on gpu
import Functors: @functor, functor, fmap, isleaf

# To perform simulations within a voxel we need tissue properties as inputs.
# Supported combinations of tissue properties are defined in tissueparameters.jl
include("interfaces/tissueproperties.jl")



export @parameters, AbstractTissueProperties, hasB₁, hasB₀
export T1T2, T1T2B1, T1T2B0, T1T2B1B0
export T1T2PDxPDy, T1T2B1PDxPDy, T1T2B0PDxPDy, T1T2B1B0PDxPDy
export SimulationParameters

# Informal interface for sequence implementations. By convention,
# a sequence::BlochSimulator is used to simulate magnetization at
# echo times only.
include("interfaces/sequences.jl")

export BlochSimulator, IsochromatSimulator, EPGSimulator

# Operator functions for isochromat model and EPG model
include("operators/isochromat.jl")
include("operators/epg.jl")
include("operators/utils.jl")

export Isochromat, ConfigurationStates

# Currently included example sequences:

# Isochromat simulator that is generic in the sense that it accepts
# arrays of RF and gradient waveforms similar to the Bloch simulator from
# Brian Hargreaves (http://mrsrl.stanford.edu/~brian/blochsim/)
include("../sequences/generic2d.jl") # with summation over slice direction
include("../sequences/generic3d.jl")

# An isochromat-based pSSFP sequence with variable flip angle train
include("../sequences/pssfp2d.jl")
include("../sequences/pssfp3d.jl")

# An EPG-based gradient-spoiled (FISP) sequence with variable flip angle train
include("../sequences/fisp2d.jl")
include("../sequences/fisp3d.jl")

include("../sequences/adiabatic.jl")

# Informal interface for trajectory implementations. By convention,
# a `sequence::BlochSimulator` is used to simulate magnetization at echo times
# only and a trajectory::AbstractTrajectory is used to simulate
# the magnetization at other readout times.
include("interfaces/trajectories.jl")

# To simulate the effects of a gradient trajectory, the spatial coordinates
# of the voxels must be known. The coordinates are stored in a `Coordinates` struct
# with x,y, and z fields (with alias `xyz`).
include("interfaces/coordinates.jl")

export Coordinates, make_coordinates, @coordinates

# Currently included example trajectories:
include("../trajectories/cartesian.jl")
include("../trajectories/radial.jl")

# Utility functions (gpu, f32, f64) to send structs to gpu
# and change their precision
include("utils/gpu.jl")
include("utils/precision.jl")

# This packages supports different types of computation:
# 1. Single CPU computation
# 2. Multi-threaded CPU computation (when Julia is started with -t <nthreads>)
# 2. Multi-process CPU computation (when workers are added with addprocs)
# 3. CUDA GPU computation
# ComputationalResources is used to dispatch on the different computational resources.

# Hard-coded nr of threads per block on GPU
const THREADS_PER_BLOCK = 32

# Main function to call a sequence simulator with a set of input parameters are defined in simulate.jl
include("simulate/magnetization.jl")
include("simulate/signal.jl")

export simulate_magnetization, simulate_signal, magnetization_to_signal, phase_encoding!

end # module
