# Script used to generate figure with EPG model benchmark on GPU hardware:
# We use the FISP2D simulator which can be used to simulate the magnetization
# response for a transient-state gradient-spoiled sequence with fixed TR and TE but varying flip
# angles. We compare this against SnapMRF and RNN-EPG.

## SnapMRF: Bash commands to download and install

# Download and install:
# mkdir -p snapmrf
# git clone git@github.com:dongwang881107/snapMRF.git --depth 1 --branch=master snapmrf
# (cd snapmrf/src && make)

# Modify snapmrf/data/MRF001.csv s.t. the TR extensions become 0
# awk -F, 'NR>1 { $3 = "0.0" } { print }' OFS=, snapmrf/data/MRF001.csv > snapmrf/data/MRF002.csv

# Instructions to get timings for SnapMRF for 10_000 to 350_000 simulations:
# rm timings_snapmrf.txt
# touch timings_snapmrf.txt
# for i in {1100..4500..100}
# do
#     snapmrf/src/mrf --T1 1001:1:$i --T2 101:1:200 --B1 1.01:0.2:1.10 -w 25 snapmrf/data/MRF002.csv | tail -n 2 | grep s | cut -d ' ' -f4 >> timings_snapmrf.txt
# done
# the above command runs snapmrf for several dictionary sizes and stores the timings in timings_snapmrf.txt
# retrieve those timings in Julia:
# parse.(Float64, readlines("timings_snapmrf.txt"))
# remember to multiply by nsubslices because SnapMRF does not support slice profile correction out of the box

## RNN-EPG:

# Follow instructions on https://gitlab.com/HannaLiu/rnn_epg to download
# the model to some location, say `benchmarks/RNN`. Copy the file 3_epg_rnn.py to the `benchmarks/RNN/Code`
# (cp benchmarks/3_epg_rnn.py benchmarks/RNN/Code/)
# Make sure a python environment in which tensorflow is installed is activated.

# Instructions to get timings for EPG-RNN for 10_000 to 100_000 simulations:
#
# We "abuse" line 68 of 3_epg_rnn.py which sets the nr of B1 steps to modify the
# number of simulations. For each different B1 value, 10000 simulations are performed.
# open a separate terminal
# conda activate tf-gpu
# rm timings_rnnepg.txt
# touch timings_rnnepg.txt
# cd benchmarks/RNN
# for i in {1..35}
# do
#   python Code/3_epg_rnn.py $i
# done

## Load packages

    using Pkg
    Pkg.activate("benchmarks")

    using Revise, BenchmarkTools, StaticArrays, ComputationalResources
    using MAT, Test, RawArray, DelimitedFiles, LinearAlgebra, JLD2
    using PGFPlotsX, Colors

    using BlochSimulators

    # Utility functions to assemble simulators and quickly switch between several different sequence simulator formats
    includet("utilities.jl")

## Setup simulator

    # load flip angles from snapmrf/data/MRF002.csv (the entries are multiplication factors of the base flip angle of 60 degrees)
    factors = (readdlm("snapmrf/data/MRF002.csv",',', header=true) |> first)[:,1]
    nTR = length(factors)
    RF_train = complex.(60 * factors)
    sliceprofiles = complex.(ones(nTR,1)) # no slice profile correction in SnapMRF
    TR = 0.016  # [s] base TR from SnapMRF (see snapmrf/src/functions.cu)
    TE = 0.0035 # [s] base TE from SnapMRF (see snapmrf/src/functions.cu)
    TI = 0.040  # [s] base TI from SnapMRF (see snapmrf/src/functions.cu)
    max_state = 45

    fisp = FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI);

## Compare output

    # Run SnapMRF
    run(`snapmrf/src/mrf -w 45 snapmrf/data/MRF002.csv -a output.ra -p parameters.ra`)
    # Load output echos
    m_snapmrf = RawArray.raread("output.ra")
    # Load parameters for which the echos were generated
    parameters_snapmrf = RawArray.raread("parameters.ra") |> permutedims;
    T1 = parameters_snapmrf[:,1] * 10^-3;
    T2 = parameters_snapmrf[:,2] * 10^-3;
    # Convert to BlochSimulators' format
    parameters = map(T₁T₂,T1,T2);
    # Simulate with BlochSimulators
    m_blochsimulators = simulate_magnetization(CUDALibs(), gpu(f32(fisp)), gpu(f32(parameters)));
    # SnapMRF automatically normalizes columns of dictionary, do the same to be able to compare
    normalize!.(eachcol(m_blochsimulators))
    # From GPU to CPU
    m_blochsimulators = collect(m_blochsimulators)
    # Compare outputs
    @assert m_snapmrf ≈ m_blochsimulators

## Make sequence with complex and real flip angles to reduce computations and memory requirements

    nTR = 5 * 224
    nsubslices = 35
    max_state = 25
    RF_train = complex.(range(0,90,length=nTR))
    sliceprofiles = complex.(rand(nTR,nsubslices))

    fisp = FISP2D(RF_train, sliceprofiles, TR, TE, max_state, TI);

    fisp_real = BlochSimulators.FISP2D( abs.(RF_train), abs.(sliceprofiles), TR, TE, max_state, TI );

    fisp = gpu(f32(fisp));
    fisp_real = gpu(f32(fisp_real));

    TIMINGS = (
        blochsimulators = [],
        blochsimulators_real = [],
        blochsimulators_real_derivatives = [],
        snapmrf = parse.(Float64, readlines("timings_snapmrf.txt")) * nsubslices, # for max_state = 25
        rnn = parse.(Float64, readlines("timings_rnnepg.txt"))
    )

## Compile

    N = 10000
    println(N)
    T₁_range = 1.0:0.01:5.0
    T₂_range = 0.01:0.001:1.0
    parameters = [BlochSimulators.T₁T₂xyz{Float64}(rand(T₁_range), rand(T₂_range),rand(), rand(), rand()) for i in 1:N]
    parameters = gpu(f32(parameters))

    @elapsed simulate_magnetization(CUDALibs(), fisp, parameters)
    @elapsed simulate_magnetization(CUDALibs(), fisp_real, parameters)

## benchmark

    N_range = 10^4:10^4:350_000

    for N in N_range

        println(N)
        parameters = [BlochSimulators.T₁T₂xyz{Float64}(rand(T₁_range), rand(T₂_range),rand(), rand(), rand()) for i in 1:N]
        parameters = gpu(f32(parameters))

        push!(TIMINGS.blochsimulators,      @elapsed simulate_magnetization(CUDALibs(), fisp, parameters))
        push!(TIMINGS.blochsimulators_real, @elapsed simulate_magnetization(CUDALibs(), fisp_real, parameters))

    end

## also benchmark computation of partial derivatives

    function simulate_derivatives(resource, sequence, parameters, Δ = 10^-4 )

        Δ = Float32(Δ)

        # simulate magnetization
        m = simulate_magnetization(resource, sequence, parameters)

        parameters = collect(parameters)

        # derivatives w.r.t. T₁
        Δpars = map(p -> T₁T₂(p.T₁+Δ, p.T₂), parameters) |> gpu
        Δm = simulate_magnetization(resource, sequence, Δpars)
        ∂m∂T₁ = @. (Δm - m)/Δ

        # derivatives w.r.t. T₂
        Δpars = map(p -> T₁T₂(p.T₁, p.T₂+Δ), parameters) |> gpu
        Δm = simulate_magnetization(resource, sequence, Δpars)
        ∂m∂T₂ = @. (Δm - m)/Δ

        return m, ∂m∂T₁, ∂m∂T₂
    end

    N = 10000
    parameters = [BlochSimulators.T₁T₂xyz{Float64}(rand(T₁_range), rand(T₂_range),rand(), rand(), rand()) for i in 1:N]
    parameters = gpu(f32(parameters))

    BlochSimulators.CUDA.@time dm = simulate_derivatives(CUDALibs(), fisp_real, parameters);

    # BlochSimulators.CUDA.@time dm = simulate_magnetization(CUDALibs(), fisp, parameters);

    N_range = 10^4:10^4:350_000

    for N in N_range

        println(N)
        parameters = [BlochSimulators.T₁T₂xyz{Float64}(rand(T₁_range), rand(T₂_range),rand(), rand(), rand()) for i in 1:N]
        parameters = gpu(f32(parameters))

        push!(TIMINGS.blochsimulators_real_derivatives, @elapsed simulate_derivatives(CUDALibs(), fisp_real, parameters))

    end

## make pgf plot
p = @pgf Axis(
    {
        xlabel = "Number of simulations",
        ylabel = "Runtime (s)",
        grid = "both",
        ymode = "log",
        xmode = "log",
        minor_grid_style="{dotted,gray!50}",
        major_grid_style="{gray!50}",
        legend_pos="outer north east",
        label_style="{font=\\Large}",
        title_style="{align = center, font=\\LARGE}",
        # title = "{\\textbf{EPG model}, \\ single precision GPU simulations}",
        title = "{\\\\ \\textbf{EPG model,}\\\\\\textbf{single precision GPU simulations}}",
        legend_style="{at={(0.5,-0.25)},anchor=north}",
        legend_cell_align="{left}",
        xtick="{10000,100000,350000}",
        xticklabels="{\$10^4\$,\$10^5\$,\$3.5\\times10^5\$}",
        # ytick="{0,1,2,3,4,5,6,7}",
        # yticklabels="{0,1,2,3,4,5,6,7}",
        xmin="0"
        # scaled_ticks="false"
        # tick_label_style="{font=tiny}"
        # legend_style="{align=left}"
        # ymax = "100000",
        # xmin = "0"

    },
    PlotInc(
        {color =  colors[5], mark  = "o", style = { thick }},
        Coordinates( collect(zip((N_range),(TIMINGS.snapmrf))))
        ),
    LegendEntry("SnapMRF (complex)"),
    PlotInc(
        {color =  colors[2], mark  = "o", style = { thick }},
        Coordinates( collect(zip((N_range),(TIMINGS.blochsimulators))))
        ),
    LegendEntry("BlochSimulators.jl FISP2D (complex)"),
    PlotInc(
        {color =  colors[2], mark  = "triangle*", mark_options={fill=colors[2]}, style = { thick }},
        Coordinates( collect(zip((N_range),(TIMINGS.blochsimulators_real))))
        ),
    LegendEntry("BlochSimulators.jl FISP2D (real)"),

    PlotInc(
        {color =  colors[1], mark = "o",  style = { thick }},
        Coordinates( collect(zip((N_range),(TIMINGS.rnn))))
        ),
    LegendEntry("RNN-EPG (real, including derivatives)"),
    PlotInc(
        {color =  colors[2], mark  = "x", mark_options={fill=colors[2]}, style = { thick }},
        Coordinates( collect(zip((N_range),(TIMINGS.blochsimulators_real_derivatives))))
        ),
    LegendEntry("BlochSimulators.jl FISP2D (real, including derivatives)")


    )
    #     legend(("SnapMRF (complex, single precision)", "BlochSimulators.jl FISP (complex, single precision)", "BlochSimulators.jl FISP (real, single precision)", "RNN-EPG (real, single precision)"), fontsize=16)

    # SAVEFIGS && @save "figures/TIMINGS_3_epg_gpu.jld2" TIMINGS
    # SAVEFIGS && pgfsave("figures/3_epg_gpu.pdf", p, dpi=300)

    p