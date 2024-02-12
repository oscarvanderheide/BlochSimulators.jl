# Script used to generate figure with Isochromat model benchmark on CPU hardware:
# The Hargreaves' simulator takes vectors of RF, GR and Δt values as input
# and at each (user-chosen) timestep, performs a rotation followed by decay and regrowth.
# In BlochSimulators, the Generic2D and Generic3D simulators operate in a similar fashion.
# The pSSFP2D sequence is used to simulate signal for a transient-state gradient-balanced sequence with fixed TR and TE
# but varying flip angles. We first setup a pSSFP2D simulator and then use a function from utilities.jl
# to generate the corresponding RF, GR and Δt vectors that can then be used with Generic2D/3D or
# Hargreaves simulator.

## Bash commands to download, modify and compile Hargreaves' Bloch simulator:

# download source code from website:
# mkdir -p hargreaves
# wget http://mrsrl.stanford.edu/~brian/blochsim/bloch.c -P hargreaves/

# replace #include "mex.h" with #include stdlib.h:
# sed -i 's/#include "mex.h"/#include <stdlib.h>/' hargreaves/bloch.c

# remove the mex function (lines 513 and further)
# sed -i '513,$d' hargreaves/bloch.c

# compile into shared library (linux, may need to be adjusted on windows/mac)
# gcc -O2 -shared hargreaves/bloch.c -o hargreaves/bloch.so -lm

## Load packages

    using Pkg
    Pkg.activate("benchmarks")

    using Revise, BenchmarkTools, StaticArrays, ComputationalResources, MAT, Test, JLD2
    using PGFPlotsX, Colors

    using BlochSimulators

## Wrapper function that calls Hargreaves' blochsimfz from shared library

    function hargreaves((b1real, b1imag, xgrad, ygrad, zgrad, tsteps, t1, t2, dfreq, dxpos, dypos, dzpos, sample));

        ntime = length(tsteps)
        nfreq = length(dfreq)
        npos = length(dxpos)

        @assert ntime == length(sample)

        # initialize arrays to store magnetization
        mx = zeros(ntime, nfreq, npos);
        my = zeros(ntime, nfreq, npos);
        mz = zeros(ntime, nfreq, npos);

        # set initial conditions (x,y,z) = (0,0,1) for each isochromat
        mz[:,:,:] .= 1

        # store all timepoints
        mode = 2

        # call compiled c version
        ccall((:blochsimfz, "hargreaves/bloch.so"), Int32, (
            Ptr{Float64}, # b1real
            Ptr{Float64}, # b1imag
            Ptr{Float64}, # xgrad
            Ptr{Float64}, # ygrad
            Ptr{Float64}, # zgrad
            Ptr{Float64}, # tsteps
            Int32,  # ntime
            Float64, # t1
            Float64, # t2
            Ptr{Float64}, # dfreq
            Int32, # nfreq
            Ptr{Float64}, # dxpos
            Ptr{Float64}, # dypos
            Ptr{Float64}, # dzpos
            Int32, # npos
            Ptr{Float64}, # mx
            Ptr{Float64}, # my
            Ptr{Float64}, # mz
            Int32 # mode
            ),
        b1real, b1imag, xgrad, ygrad, zgrad, tsteps, ntime, t1, t2, dfreq, nfreq, dxpos, dypos, dzpos, npos, mx, my, mz, mode)

        # return magnetization at specified sample times only
        if sum(sample) != ntime

            mx = mx[sample,:,:]
            my = my[sample,:,:]
            mz = mz[sample,:,:]

        end

        return mx, my, mz
    end

    # Utility functions to assemble simulators and quickly switch between several different sequence simulator formats
    includet("utilities.jl")

## Assemble sequence simulators

    nTR, nRF = 1120, 25

    single_double(sequence) = (
        f64 = sequence,
        f32 = f32(sequence)
    );

    pssfp2d     = assemble_pssfp2d(nTR, nRF, 1)     |> single_double;
    pssfp3d     = assemble_pssfp3d(nTR, nRF)        |> single_double;
    generic2d   = pssfp2d_to_generic2d(pssfp2d.f64) |> single_double;
    generic3d   = pssfp3d_to_generic3d(pssfp3d.f64) |> single_double;
    

## Run for random parameter combination input to compile and compare output

    N = 1

    parameters  = [BlochSimulators.T₁T₂xyz(rand(1.0:0.01:5.0), rand(0.04:0.001:1.0), rand(), rand(), only(pssfp2d.f64.z)) for _ = 1:N]

    hargreaves_input = generic3d_to_hargreaves(generic3d.f64, parameters);

    output_hargreaves       = hargreaves(hargreaves_input)
    output_generic3d_f64    = simulate_magnetization(CPU1(), generic3d.f64, parameters);
    output_generic3d_f32    = simulate_magnetization(CPU1(), generic3d.f32, f32(parameters));
    output_generic2d_f64    = simulate_magnetization(CPU1(), generic2d.f64, parameters);
    output_generic2d_f32    = simulate_magnetization(CPU1(), generic2d.f32, f32(parameters));

    output_pssfp3d_f64      = simulate_magnetization(CPU1(), pssfp3d.f64, parameters);
    output_pssfp3d_f32      = simulate_magnetization(CPU1(), pssfp3d.f32, f32(parameters));
    output_pssfp2d_f64      = simulate_magnetization(CPU1(), pssfp2d.f64, parameters);
    output_pssfp2d_f32      = simulate_magnetization(CPU1(), pssfp2d.f32, f32(parameters));

    @info "Test whether pSSFP2d (with just one z=0 location) and Generic2D (with same z=0 location) give the same output"
    @test output_pssfp2d_f64 ≈ [complex(m.x,m.y) for m in output_generic2d_f64]

    @info "Test whether Generic2D (with just one z=0 location) and Generic3D give the same output"
    @test output_generic3d_f64 ≈ output_generic2d_f64

    @info "Test whether Generic3D and Hargreaves' simulator give the same output"
    output_hargreaves = map(Isochromat, output_hargreaves...);
    output_hargreaves = reshape(output_hargreaves, nTR, N);
    @test output_hargreaves ≈ output_generic3d_f64

## Benchmark using many voxels

    TIMINGS = (
        hargreaves_f64 = [],
        generic3d_f64 = [],
        generic3d_f32 = [],
        pssfp3d_f64 = [],
        pssfp3d_f32 = []
    )

    N_range = 10^3:10^3:10*10^3

    for N in N_range

        println(N)
        parameters_xyz_f64  = [BlochSimulators.T₁T₂xyz(rand(1.0:0.01:5.0), rand(0.04:0.001:1.0), rand(), rand(), rand()) for i in 1:N]
        parameters_xyz_f32  = f32(parameters_xyz_f64)
        hargreaves_input = generic3d_to_hargreaves(generic3d.f64, parameters_xyz_f64);

        push!(TIMINGS.hargreaves_f64, @elapsed hargreaves(hargreaves_input))
        push!(TIMINGS.generic3d_f64, @elapsed simulate_magnetization(CPU1(), generic3d.f64, parameters_xyz_f64))
        push!(TIMINGS.generic3d_f32, @elapsed simulate_magnetization(CPU1(), generic3d.f32, parameters_xyz_f32))
        push!(TIMINGS.pssfp3d_f64, @elapsed simulate_magnetization(CPU1(), pssfp3d.f64, parameters_xyz_f64))
        push!(TIMINGS.pssfp3d_f32, @elapsed simulate_magnetization(CPU1(), pssfp3d.f32, parameters_xyz_f32))

    end

## make pgf plot

# @load "data/TIMINGS_1_isochromat_cpu.jld2" TIMINGS

p = @pgf Axis(
        {
            xlabel = "Number of simulations",
            ylabel = "Runtime (s)",
            title = "{\\textbf{Isochromat model,} \\\\ \\textbf{CPU simulations (single-threaded)}}",
            grid = "both",
            # ymode = "log",
            # xmode = "log",
            minor_grid_style="{dotted,gray!50}",
            major_grid_style="{gray!50}",
            legend_pos="outer north east",
            label_style="{font=\\Large}",
            title_style="{align=center, font=\\LARGE}",
            legend_cell_align="{left}",
            xtick="{2000,4000,6000,8000,10000}",
            xticklabels="{2000,4000,6000,8000,10000}",
            scaled_ticks="false",
            legend_style="{at={(0.5,-0.25)},anchor=north}"
            # tick_label_style="{font=tiny}"
            # legend_style="{align=left}"
            # ymax = "100000",
            # xmin = "15"
        },
        PlotInc(
            {color =  colors[5], mark  = "o", style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.hargreaves_f64))))
            ),
        LegendEntry("Hargreaves' C Simulator (double precision)"),
        PlotInc(
            {color =  colors[1], mark  = "o", style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.generic3d_f64))))
            ),
        LegendEntry("BlochSimulators.jl Generic3D (double precision)"),
        PlotInc(
            {color =  colors[1], mark  = "triangle*", mark_options={fill=colors[1]}, style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.generic3d_f32))))
            ),
        LegendEntry("BlochSimulators.jl Generic3D (single precision)"),
        PlotInc(
            {color =  colors[2], mark  = "o", style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.pssfp3d_f64))))
            ),
        LegendEntry("BlochSimulators.jl pSSFP3D (double precision)"),
        PlotInc(
            {color =  colors[2], mark = "triangle*", mark_options={fill=colors[2]}, style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.pssfp3d_f32))))
            ),
        LegendEntry("BlochSimulators.jl pSSFP3D (single precision)")
        )


    # @save "data/TIMINGS_1_isochromat_cpu.jld2" TIMINGS
    # pgfsave("figures/1_isochromat_cpu.pdf", p, dpi=300)

    p
