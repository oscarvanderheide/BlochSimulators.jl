# Script used to generate figure with isochromat model benchmark on GPU hardware:
# We use the same Generic3D and pSSFP2D simulators from 1_isochromat_cpu.jl
# but now we run the simulations on GPU hardware instead.


## Load packages

    using Revise, BenchmarkTools, StaticArrays, ComputationalResources, MAT, JLD2, Test
    using PGFPlotsX, Colors

    using BlochSimulators

    includet("utilities.jl")

## Assemble sequence simulators

    nTR, nRF = 1120, 25

    single_double_gpu(sequence) = (
        cpu_f64 = sequence,
        cpu_f32 = f32(sequence),
        gpu_f64 = gpu(sequence),
        gpu_f32 = gpu(f32(sequence))
    );

    pssfp3d     = assemble_pssfp3d(nTR, nRF)              |> single_double_gpu;
    generic3d   = pssfp3d_to_generic3d(pssfp3d.cpu_f64)   |> single_double_gpu;

## Run for 1 voxel to compile and compare output

    # Compile and benchmark against CPU output
    N = 100

    parameters_cpu_f64 = [BlochSimulators.T₁T₂xyz{Float64}(rand(1.0:0.01:5.0), rand(0.01:0.001:1.0),rand(), rand(), rand()) for i in 1:N]
    parameters_cpu_f32 = f32(parameters_cpu_f64)
    parameters_gpu_f64 = gpu(parameters_cpu_f64)
    parameters_gpu_f32 = gpu(f32(parameters_cpu_f64))

    d1_gpu = simulate_magnetization(CUDALibs(), pssfp3d.gpu_f64,   parameters_gpu_f64)
    d2_gpu = simulate_magnetization(CUDALibs(), generic3d.gpu_f64, parameters_gpu_f64)
    d3_gpu = simulate_magnetization(CUDALibs(), pssfp3d.gpu_f32,   parameters_gpu_f32)
    d4_gpu = simulate_magnetization(CUDALibs(), generic3d.gpu_f32, parameters_gpu_f32)

    d1_cpu = simulate_magnetization(CUDALibs(), pssfp3d.cpu_f64,   parameters_cpu_f64)
    d2_cpu = simulate_magnetization(CUDALibs(), generic3d.cpu_f64, parameters_cpu_f64)
    d3_cpu = simulate_magnetization(CUDALibs(), pssfp3d.cpu_f32,   parameters_cpu_f32)
    d4_cpu = simulate_magnetization(CUDALibs(), generic3d.cpu_f32, parameters_cpu_f32)

    # Test whether CPU and GPU give the same output
    @test Array(d1_gpu) == d1_cpu
    @test Array(d2_gpu) == d2_cpu
    @test Array(d3_gpu) == d3_cpu
    @test Array(d4_gpu) == d4_cpu

## Benchmark

    TIMINGS = (
        pssfp_gpu_f64 = [],
        pssfp_gpu_f32 = [],
        generic3d_gpu_f64 = [],
        generic3d_gpu_f32 = []
    )

    N_range = 10_000:10^4:350_000 # different nr of voxels

    for N in N_range

        println(N)
        parameters_cpu_f64 = [BlochSimulators.T₁T₂xyz{Float64}(rand(1.0:0.01:5.0), rand(0.01:0.001:1.0),rand(), rand(), rand()) for i in 1:N]
        parameters_cpu_f32 = f32(parameters_cpu_f64)
        parameters_gpu_f64 = gpu(parameters_cpu_f64)
        parameters_gpu_f32 = gpu(f32(parameters_cpu_f64))

        push!(TIMINGS.pssfp_gpu_f64,     @elapsed simulate_magnetization(CUDALibs(), pssfp3d.gpu_f64,        parameters_gpu_f64))
        push!(TIMINGS.pssfp_gpu_f32,     @elapsed simulate_magnetization(CUDALibs(), pssfp3d.gpu_f32,        parameters_gpu_f32))
        push!(TIMINGS.generic3d_gpu_f32, @elapsed simulate_magnetization(CUDALibs(), generic3d.gpu_f32,    parameters_gpu_f32))
        push!(TIMINGS.generic3d_gpu_f64, @elapsed simulate_magnetization(CUDALibs(), generic3d.gpu_f64,    parameters_gpu_f64))

    end

## Make pgf plot
    p = @pgf Axis(
        {
            xlabel = "Number of simulations",
            ylabel = "Runtime (s)",
            title = "{\\textbf{Isochromat model, GPU simulations}}",
            grid = "both",
            ymode = "log",
            xmode = "log",
            minor_grid_style="{dotted,gray!50}",
            major_grid_style="{gray!50}",
            legend_pos="outer north east",
            label_style="{font=\\Large}",
            title_style="{align=center, font=\\LARGE}",
            legend_cell_align="{left}",
            xtick="{10000,100000,350000}",
            xticklabels="{\$10^4\$,\$10^5\$,\$3.5\\times10^5\$}",
            # ytick="{0,1,2,3,4,5,6,7}",
            # yticklabels="{0,1,2,3,4,5,6,7}",
            xmin="0",
            legend_style="{at={(0.5,-0.25)},anchor=north}",
            scaled_ticks="false"
            # tick_label_style="{font=tiny}"
            # legend_style="{align=left}"
            # ymax = "100000",
            # xmin = "15"

        },
        PlotInc(
            {color =  colors[1], mark  = "o", style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.generic3d_gpu_f64))))
            ),
        LegendEntry("BlochSimulators.jl Generic3D (double precision)"),
        PlotInc(
            {color =  colors[1], mark  = "triangle*", mark_options={fill=colors[1]}, style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.generic3d_gpu_f32))))
            ),
        LegendEntry("BlochSimulators.jl Generic3D (single precision)"),
        PlotInc(
            {color =  colors[2], mark  = "o", style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.pssfp_gpu_f64))))
            ),
        LegendEntry("BlochSimulators.jl pSSFP3D (double precision)"),
        PlotInc(
            {color =  colors[2], mark = "triangle*", mark_options={fill=colors[2]}, style = { thick }},
            Coordinates( collect(zip((N_range),(TIMINGS.pssfp_gpu_f32))))
            ),
        LegendEntry("BlochSimulators.jl pSSFP3D (single precision)")
        )

    p
    # @save "data/TIMINGS_2_isochromat_gpu.jld2" TIMINGS
    # pgfsave("figures/2_isochromat_gpu.pdf", p, dpi=300)