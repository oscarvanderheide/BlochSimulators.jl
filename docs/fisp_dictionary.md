In this example we demonstrate how to generate an MR Fingerprinting
dictionary using a FISP type sequence

````julia
using BlochSimulators
using ComputationalResources
````

First, construct a FISP sequence struct (see `src/sequences/fisp.jl`)

````julia
nTR = 1000; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect; # flip angle train
TR,TE,TI = 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay
max_state = 25; # maximum number of configuration states to keep track of

sequence = FISP2D(RF_train, TR, TE, max_state, TI);
````

Next, set the desired input parameters

````julia
T₁ = 0.500:0.010:2.0; # T₁ range
T₂ = 0.025:0.005:0.5; # T₂ range

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁
````

Now we can perform the simulations on different hardware resources
Note that the first time a function is called in a Julia session,
a precompilation procedure starts and the runtime for subsequent function
calls are significantly faster

To perform simulations on a single CPU in single-threaded mode

````julia
@time dictionary = simulate(CPU1(), sequence, parameters);
@time dictionary = simulate(CPU1(), sequence, parameters);
````

To use multiple threads, Julia must be started with the `--threads=auto`
flag (or some integer instead of `auto`)

````julia
@time dictionary = simulate(CPUThreads(), sequence, parameters);
@time dictionary = simulate(CPUThreads(), sequence, parameters);
````

For distributed CPU mode, use the Distribute packages (ships with Julia)
to add workers

````julia
using Distributed
addprocs(8, exeflags="--project=.")
@everywhere using BlochSimulators
````

Then the dictionary is simulated in a distributed fashion with the following syntax

````julia
@time dictionary = simulate(CPUProcesses(), sequence, parameters);
@time dictionary = simulate(CPUProcesses(), sequence, parameters);
````

To perform simulations on GPU, we first convert the sequence and parameters
to single precision and then send them to the gpu

````julia
cu_sequence = sequence |> f32 |> gpu;
cu_parameters = parameters |> f32 |> gpu;

@time dictionary = simulate(CUDALibs(), cu_sequence, cu_parameters);
@time dictionary = simulate(CUDALibs(), cu_sequence, cu_parameters);
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

