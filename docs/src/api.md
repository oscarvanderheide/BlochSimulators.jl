```@meta
CurrentModule = BlochSimulators
```

# API

## Isochromat Operators

```@docs
Isochromat
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["operators/isochromat.jl"]
```

## EPG Operators
```@docs
EPGStates
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["operators/epg.jl"]
```

## Tissue Parameters

```@docs
AbstractTissueParameters
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["parameters/tissueparameters.jl"]
```

## Sequences

#### Abstract Types
```@docs
BlochSimulator
IsochromatSimulator
EPGSimulator
```

#### Interface
```@autodocs
Modules = [BlochSimulators]
Pages   = ["sequences/_interface.jl"]
```

#### Examples

```@autodocs
Modules = [BlochSimulators]
Pages   = ["sequences/fisp.jl", "sequences/generic2d.jl", "sequences/generic3d.jl", "sequences/pssfp.jl"]
```

## Trajectories

#### Abstract types
```@docs
AbstractTrajectory
SpokesTrajectory
```

#### Interface
```@autodocs
Modules = [BlochSimulators]
Pages   = ["trajectories/_interface.jl"]
```

#### Examples
```@autodocs
Modules = [BlochSimulators]
Pages   = ["trajectories/cartesian.jl", "trajectories/radial.jl", "trajectories/spiral.jl"]
```

## Dictionary Simulation
```@docs
simulate(resource, sequence, parameters)
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["simulate/dictionary.jl"]
```

## Signal Simulation
```@docs
simulate(resource, sequence, parameters, trajectory, coil_sensitivities)
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["simulate/signal.jl"]
```

## Utility Functions
```@autodocs
Modules = [BlochSimulators]
Pages   = ["utils/precision.jl", "utils/gpu.jl"]
```

## Index
```@index
```
