module BlochSimulators

    # Load external packages
    using LinearAlgebra
    using StaticArrays
    using ComputationalResources
    using OffsetArrays
    using Distributed
    using DistributedArrays
    using CUDA
    import Adapt: adapt, adapt_storage, @adapt_structure # to allow custom Structs of non-isbits type to be used on gpu
    using Functors
    import Functors: @functor, functor, fmap, isleaf
    using InteractiveUtils # needed for subtypes function
    using Unitful, Unitless

    # To perform simulations we need tissue parameters as inputs. Supported combinations of tissue parameters
    # are defined tissueparameters.jl
    include("parameters/tissueparameters.jl")

    # Informal interface for sequence implementations. By convention,
    # a sequence::BlochSimulator is used to simulate magnetization at
    # echo times only.
    include("sequences/_interface.jl")

    # Operator functions for isochromat model and EPG model
    include("operators/isochromat.jl")
    include("operators/epg.jl")
    include("operators/utils.jl")

    # Currently included example sequences:

    # Isochromat simulator that is generic in the sense that it accepts
    # arrays of RF and gradient waveforms similar to the Bloch simulator from
    # Brian Hargreaves (http://mrsrl.stanford.edu/~brian/blochsim/)
    include("sequences/generic2d.jl") # with summation over slice direction
    include("sequences/generic3d.jl")

    # An isochromat-based pSSFP sequence with variable flip angle train
    include("sequences/pssfp.jl")

    # An EPG-based gradient-spoiled (FISP) sequence with variable flip angle train
    include("sequences/fisp.jl")

    # Informal interface for trajectory implementations. By convention,
    # a sequence::BlochSimulator is used to simulate magnetization at echo times
    # only and a trajectory::AbstractTrajectory is used to simulate
    # the magnetization at other readout times.
    include("trajectories/_interface.jl")

    # Currently included example trajectories:
    include("trajectories/cartesian.jl")
    include("trajectories/radial.jl")

    # Utility functions (gpu, f32, f64) to send structs to gpu
    # and change their precision
    include("utils/gpu.jl")
    include("utils/precision.jl")

    # # This packages supports different types of computation:
    # # 1. Single CPU computation
    # # 2. Multi-threaded CPU computation (when Julia is started with -t <nthreads>)
    # # 2. Multi-process CPU computation (when workers are added with addprocs)
    # # 3. CUDA GPU computation
    # # ComputationalResources is used to dispatch on the different computational resources.

    # # Main function to call a sequence simulator with a set of input parameters are defined in simulate.jl
    include("simulate/dictionary.jl")
    include("simulate/signal.jl")

    export simulate

end # module
