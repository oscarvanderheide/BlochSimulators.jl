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
ConfigurationStates
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["operators/epg.jl"]
```

## Tissue Parameters

```@docs
AbstractTissueProperties
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["interfaces/tissueproperties.jl"]
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
Pages   = ["interfaces/sequences.jl"]
```

#### Examples

```@autodocs
Modules = [BlochSimulators]
Pages   = ["sequences/fisp2d.jl", "sequences/fisp3d.jl", "sequences/generic2d.jl", "sequences/generic3d.jl", "sequences/pssfp2d.jl", "sequences/pssfp3d.jl", "sequences/adiabatic.jl"]
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
Pages   = ["interfaces/trajectories.jl", "trajectories/_abstract.jl"]
```

#### Examples
```@autodocs
Modules = [BlochSimulators]
Pages   = ["trajectories/cartesian.jl", "trajectories/radial.jl", "trajectories/spiral.jl"]
```

## Dictionary Simulation
```@docs
simulate_magnetization(resource, sequence, parameters)
```

```@autodocs
Modules = [BlochSimulators]
Pages   = ["simulate/magnetization.jl"]
```

## Signal Simulation
```@docs
simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)
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
