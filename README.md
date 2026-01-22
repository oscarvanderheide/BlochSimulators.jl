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

---

## Developer Guide

This section provides an overview of the package architecture and design decisions to help contributors understand how the code is organized and how to extend it.

### Package Architecture Overview

BlochSimulators is built around a **separation of concerns** design philosophy that cleanly divides the simulation workflow into distinct, composable components:

1. **Tissue Properties** → What are we simulating? (T₁, T₂, B₁, B₀, etc.)
2. **Sequences** → How do we manipulate magnetization? (RF pulses, gradients, timing)
3. **Operators** → Low-level physics engines (Isochromat vs EPG models)
4. **Trajectories** → How do we encode spatial information? (k-space sampling)
5. **Computational Resources** → Where do we run? (CPU single/multi-threaded, GPU, distributed)

This modular design allows researchers to mix and match components (e.g., different sequences with different trajectories) without reimplementing core physics.

### Core Design Patterns

#### 1. Type-Driven Dispatch and Compile-Time Optimization

The package heavily uses Julia's type system to achieve high performance:

- **Parametric types** encode critical information at compile time. For example, `EPGSimulator{T,Ns}` includes both the precision `T` (Float32/Float64) and the number of configuration states `Ns` as type parameters.
- **Ns as a type parameter** is essential: `Ns` indicates the size of the EPG configuration state matrix Ω (3×Ns). This state matrix is accessed frequently and is the core of EPG simulations, so we cannot have it heap-allocated. Having its size known at compile time enables stack allocation (via `StaticArrays`) and is critical for GPU kernel compilation—the GPU compiler needs to know array sizes to generate efficient code. This is why we dispatch on `Ns` as a type parameter rather than a field value.
- **Val{Ns}** is used in struct fields when Ns needs to be stored but also accessible at compile time for dispatch.

#### 2. Informal Interfaces via Abstract Types

Rather than formal traits (like in Rust), the package uses abstract types to define informal interfaces:

- **`BlochSimulator{T}`** → Subtypes must implement:
  - `output_eltype(sequence)` → What type of output (Complex, Real, Isochromat)? If you just want transverse magnetization, combine the x and y components as a complex number (x + iy). For some sequences where transverse magnetization always aligns with one axis, you can store as reals (reducing memory and computation). If you need full x,y,z components at each timepoint, use `Isochromat` as the output type.
  - `output_size(sequence)` → How many time points?
  - `simulate_magnetization!(magnetization, sequence, state, parameters)` → The actual physics implementation
  
- **`AbstractTissueProperties{N,T}`** → Encapsulates voxel parameters (T₁, T₂, B₁, etc.)
  - Subtypes of `FieldVector` from StaticArrays for performance
  - Named fields improve code readability while maintaining vector operations
  - The `@parameters` macro provides a convenient constructor

- **`AbstractTrajectory{T}`** → Subtypes must implement:
  - `nreadouts(trajectory)`, `nsamplesperreadout(trajectory, idx)`
  - `to_sample_point(m, trajectory, readout_idx, sample_idx, parameters)` → Propagate magnetization with spatial encoding

**Why informal interfaces?** Julia does not have formal interfaces (yet), but interfaces provide significant benefits for code organization and extensibility. By defining informal interfaces through abstract types and their required methods, developers implementing new sequences or trajectories know exactly what methods to define by looking at the abstract type's documentation.

#### 3. Resource-Based Dispatch for Heterogeneous Computing

The package uses `ComputationalResources.jl` to abstract computational backends:

- `CPU1()` → Single-threaded execution
- `CPUThreads()` → Multi-threaded (respects `JULIA_NUM_THREADS`)
- `CPUProcesses()` → Distributed computing (works with `DArray`s)
- `CUDALibs()` → GPU execution

Key design decisions:
- The `gpu(x)` function (using `Functors.jl`) recursively moves structs to GPU, converting Arrays to CuArrays
- Sequences/trajectories must be marked with `@functor` to enable GPU transfer
- On GPU, simulations use a fixed `THREADS_PER_BLOCK` with `WARPSIZE` (hardcoded to 32 for CUDA devices) threads per voxel
- **Type stability and zero-allocation** in kernel code is critical for GPU performance

#### 4. The Two-Level Simulation API

The package exposes two main entry points that differ in their level of abstraction:

**Level 1: `simulate_magnetization(resource, sequence, parameters)`**
- Simulates magnetization at **echo times only**, without spatial encoding
- Returns array of size `(output_size(sequence), length(parameters))`
- Used for MR Fingerprinting dictionaries where k-space trajectories aren't needed
- Each voxel's simulation is independent → embarrassingly parallel

**Level 2: `simulate_signal(resource, sequence, parameters, trajectory, coordinates, coil_sensitivities)`**
- Full signal simulation including spatial encoding and multi-coil reception
- First calls `simulate_magnetization` for all voxels
- Then propagates magnetization to all k-space sample points via `trajectory`
- Returns array of size `(# samples per readout, # readouts, # coils)`
- Used for realistic MRI signal simulations

This two-level design avoids code duplication and allows users to choose the appropriate abstraction level.

### Code Organization

```
src/
├── BlochSimulators.jl           # Main module file, includes all components
├── interfaces/                   # Abstract types and informal interfaces
│   ├── tissueproperties.jl      # T₁T₂, T₁T₂B₁, etc. + @parameters macro
│   ├── sequences.jl             # BlochSimulator, IsochromatSimulator, EPGSimulator
│   ├── trajectories.jl          # AbstractTrajectory interface
│   └── coordinates.jl           # Spatial coordinate handling
├── operators/                    # Low-level physics implementations
│   ├── isochromat.jl            # Rodrigues rotation, decay/regrowth for isochromats
│   ├── epg.jl                   # EPG state manipulation (shift, RF, decay, etc.)
│   └── utils.jl                 # Shared utilities
├── simulate/                     # High-level simulation orchestration
│   ├── magnetization.jl         # simulate_magnetization + resource dispatch
│   └── signal.jl                # simulate_signal + trajectory integration
└── utils/                        # Cross-cutting concerns
    ├── gpu.jl                   # gpu() function, Functors integration
    └── precision.jl             # f32/f64 precision conversion

sequences/                        # Example sequence implementations
├── fisp2d.jl, fisp3d.jl         # Gradient-spoiled FISP (EPG-based)
├── pssfp2d.jl, pssfp3d.jl       # pSSFP (EPG-based)
├── generic2d.jl, generic3d.jl   # Generic isochromat simulators
└── adiabatic.jl                 # Adiabatic inversion sequences

trajectories/                     # Example trajectory implementations
├── cartesian.jl, radial.jl, spiral.jl
└── _abstract.jl                 # Shared trajectory utilities
```

**Key principle**: Example sequences and trajectories live in top-level `sequences/` and `trajectories/` directories (not in `src/`). This makes it clear they're templates for users to copy and modify, not core library code.

### Adding a New Sequence

To implement a custom pulse sequence (e.g., for a novel contrast mechanism):

1. **Choose your model**: Subtype either `IsochromatSimulator{T}` or `EPGSimulator{T,Ns}`
   - Isochromat model: More general, handles arbitrary gradients, but computationally expensive
   - EPG model: Fast and efficient for gradient-spoiled sequences, which would otherwise require simulating many isochromats per voxel. Somewhat less flexible than the isochromat model.
   
2. **Create a struct** with all sequence parameters:
   ```julia
   struct MySequence{T, Ns} <: EPGSimulator{T, Ns}
       RF_train::Vector{Complex{T}}  # Flip angles and phases
       TR::T                          # Repetition time
       TE::T                          # Echo time
       max_state::Val{Ns}             # Maximum EPG state order
   end
   ```
   
3. **Mark it as a functor** (required for GPU support and precision conversion):
   ```julia
   @functor MySequence
   @adapt_structure MySequence
   ```

4. **Implement required methods**:
   ```julia
   output_size(seq::MySequence) = length(seq.RF_train)
   output_eltype(seq::MySequence) = Complex{Float64}
   
   function simulate_magnetization!(magnetization, sequence::MySequence, Ω, p::AbstractTissueProperties)
       # Use operators from operators/epg.jl or operators/isochromat.jl
       # Example: excite!(Ω, flip_angle), decay!(Ω, E₁, E₂), shift!(Ω)
       # Store results in magnetization array
   end
   ```

5. **Important constraints**:
   - `simulate_magnetization!` must be **type-stable** and **non-allocating**
   - Pre-compute expensive quantities (like decay constants) once per voxel
   - Use `@inbounds` for performance-critical loops after verifying correctness
   - Look at [fisp2d.jl](sequences/fisp2d.jl) or [pssfp2d.jl](sequences/pssfp2d.jl) as templates

### Adding a New Trajectory

To implement a custom k-space trajectory (e.g., for non-Cartesian imaging):

1. **Create a struct** subtyping `AbstractTrajectory{T}`:
   ```julia
   struct MyTrajectory{T} <: AbstractTrajectory{T}
       kx::Matrix{T}  # k-space coordinates (readout × time)
       ky::Matrix{T}
       Δt::Vector{T}  # Time between samples
       # ... other fields
   end
   ```

2. **Implement required methods**:
   ```julia
   nreadouts(traj::MyTrajectory) = size(traj.kx, 1)
   nsamplesperreadout(traj::MyTrajectory, idx) = size(traj.kx, 2)
   
   function to_sample_point(m, traj::MyTrajectory, readout_idx, sample_idx, params)
       # Apply gradient-induced phase: exp(i * 2π * k⋅r)
       # Apply T₂ decay: m *= exp(-Δt/T₂)
       # Apply B₀ rotation: m *= exp(i * 2π * B₀ * Δt)
       # Return modified magnetization
   end
   ```

3. **[Optional] Implement phase encoding**: For Cartesian-like trajectories, implement `phase_encoding!(magnetization, trajectory, coordinates)` to apply phase encoding before readout.

### Performance Considerations

1. **Type stability is paramount**: Use `@code_warntype` to check your `simulate_magnetization!` implementation. Any red warnings will kill performance.

2. **GPU-specific considerations**:
   - Each voxel gets multiple threads (WARPSIZE) for coalesced memory access
   - Avoid dynamic allocations in kernel code (use pre-allocated buffers)
   - StaticArrays are your friend for small fixed-size arrays

3. **Memory layout**: StructArrays (array-of-structs → struct-of-arrays) provide better cache locality and SIMD opportunities. The package automatically uses them for tissue parameters.

4. **Precision**: Use `f32(sequence)` to convert to Float32 for significant speedup on consumer GPUs (which have limited Float64 performance).

### Testing and Validation

- Unit tests in `test/runtests.jl` verify core operators and interfaces
- Benchmark scripts in `benchmarks/` compare performance across backends
- When adding a new sequence, validate against known analytical solutions or published results when possible
- Use `docs/examples/` to create reproducible examples for your sequence

### Common Pitfalls

1. **Forgetting `@functor` and `@adapt_structure`**: Your sequence won't transfer to GPU properly
2. **Non-constant global references**: Will prevent GPU kernel compilation
3. **Not respecting the type parameter `T`**: Mixing Float32 and Float64 breaks type stability
4. **Allocating inside `simulate_magnetization!`**: Use pre-allocated views or StaticArrays
5. **For EPG: `Ns` must be a multiple of 32**: On GPU, each thread block handles multiple configuration states, with each thread getting `Ns/WARPSIZE` states to work on. Since `WARPSIZE` is hardcoded to 32 for CUDA devices, `Ns` must be a multiple of 32. Note that even if non-multiples were allowed, simulation time for (say) Ns=45 would be identical to Ns=64 due to thread block quantization—you'd pay for 64 states anyway.

### Where to Get Help

- Check the [documentation](https://oscarvanderheide.github.io/BlochSimulators.jl/dev) for API reference
- Look at existing sequence implementations in `sequences/` for patterns
- Read operator implementations in `src/operators/` to understand available building blocks
- Open an issue on GitHub for bugs or feature requests
