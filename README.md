[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://oscarvanderheide.github.io/BlochSimulators.jl/dev)

# BlochSimulators

BlochSimulators is a Julia package for performing Bloch simulations within the context of Magnetic Resonance Imaging. It provides efficient implementations of both the Isochromat Summation model and the Extended Phase Graph (EPG) model to simulate MR signals resulting from custom pulse sequences and k-space trajectories. Simulations can be deployed on various computational resources, with strong support for **CUDA-compatible GPU acceleration** to achieve high runtime performance. The package is well-suited for simulating dictionaries for [MR Fingerprinting](https://doi.org/10.1038/nature11971) or performing forward model evaluations for [MR-STAT](https://doi.org/10.1016/j.mri.2017.10.015).
For detailed information, please refer to the [**Documentation**](https://oscarvanderheide.github.io/BlochSimulators.jl/dev).


#### Installation

BlochSimulators.jl is registered in the General Julia registry. To install the package, press `]` in the Julia REPL to enter package mode, followed by either

`pkg> add BlochSimulators` (if you want to use the package as-is)

or

`pkg> dev BlochSimulators` (if you want to make modificatios to the source code).
#### Basic Usage

Here's a minimal example simulating the magnetization for a single voxel using the EPG model with a FISP sequence:

```julia
using BlochSimulators

# Define tissue properties (T1=1.0s, T2=0.1s)
parameters = @parameters T1=1.0 T2=0.1

# Use a predefined FISP sequence (replace with your actual sequence definition)
# Note: You might need to define or load a sequence struct first.
# This is a placeholder example.
struct MyFISPSequence <: EPGSimulator end # Placeholder
sequence = MyFISPSequence() # Placeholder

# Simulate magnetization
magnetization = simulate_magnetization(CPU1(), sequence, parameters)

println("Final magnetization (Mx, My, Mz): ", magnetization)
```

Please see the `examples` directory and the [documentation](https://oscarvanderheide.github.io/BlochSimulators.jl/dev) for more complete examples, including sequence definitions and signal simulation.


#### Examples

The `examples` folder contains several example sequence structs as well as k-space trajectory structs. Users of BlochSimulators.jl are encouraged to modify these examples or assemble their own structs with custom sequence/k-space trajectory implementations. Example Julia code on how to use a custom sequence to simulate an MR Fingerprinting dictionary is available [here](./docs/build/dictionary.html). Another example in which both a sequence and k-space trajectory struct are used to simulate MR signal is available [here](./docs/build/signal.html).

#### Citation

See `CITATION.bib` for details on how to cite this work.
