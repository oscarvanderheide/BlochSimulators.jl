# Script used to generate figure signal simulation benchmark GPU hardware:
# We use the pSSFP2D simulator together with a radial gradient trajectory
# to simulate the MR signal for a numerical phantom on GPU hardware. We
# compare this against KomaMRI.jl. Care was taken to make sure the simulators
# are setup in such a way that the output is the same.

## Load packages


using Pkg
Pkg.activate("benchmarks")

using Revise, BenchmarkTools, StaticArrays, ComputationalResources, MAT, Test, JLD2
using PGFPlotsX, Colors

import BlochSimulators
import KomaMRI
import KomaMRI.KomaMRICore

import BlochSimulators: f32, gpu

includet("utilities.jl")

## Setup for benchmark

# Generic parameters

nTR = 500; # KomaMRI crashes for some nr of voxels when nTR = 1000 or 1120
nsamplesperreadout = 224;

# KomaMRI.jl setup

# modified version of radial_base from KomaMRI.jl
my_radial_base(FOV, nsamplesperreadout, sys) = begin
    Δt = sys.ADC_Δt
    Gmax = sys.Gmax
    Δx = FOV / (nsamplesperreadout - 1)
    Ta = Δt * (nsamplesperreadout - 1)
    Ga = 1 / (KomaMRI.γ * Δt * FOV)
    ζ = Ga / sys.Smax
    Ga ≥ sys.Gmax ? error("Ga=$(Ga*1e3) mT/m exceeds Gmax=$(Gmax*1e3) mT/m, increase Δt to at least Δt_min="
                          * string(round(1 / (γ * Gmax * FOV), digits=2)) * " us.") : 0
    #Radial base
    rad = KomaMRI.Sequence([KomaMRI.Grad(Ga, Ta, ζ)]) #Gx
    # add logic to make sure exact echo time is sampled
    dur_rad = rad.DUR[1]
    adc_delay = dur_rad / 2 - ((nsamplesperreadout / 2)) * Δt
    rad.ADC = [KomaMRI.ADC(nsamplesperreadout, Ta, adc_delay)]
    # Acq/Recon parameters
    Nspokes = ceil(Int64, π / 2 * nsamplesperreadout) #Nyquist in the radial direction
    Δθ = π / Nspokes
    # println("## Radial parameters ##")
    # println("FOVr = $(round(FOV*1e2,digits=2)) cm; Δr = $(round(Δx*1e3,digits=2)) mm")
    # println("Nspokes = $Nspokes, to satisfy the Nyquist criterion")
    PHASE = KomaMRI.Sequence([KomaMRI.Grad(-Ga / 2, Ta, ζ)])
    seq = PHASE + rad + PHASE
    #Saving parameters
    seq.DEF = Dict("Nx" => nsamplesperreadout, "Ny" => Nspokes, "Nz" => 1, "Δθ" => Δθ, "Name" => "radial", "FOV" => [FOV, FOV, 0])

    return seq
end

# Flip angle train (with 0-π phase cycling)
rf_angles = LinRange(1.0, 50.0, nTR) |> collect # degrees
# rf_angles .= 90
rf_phases = repeat([0.0, 180.0], nTR ÷ 2) # degrees
# rf_phases .= 0
TR = 0.015
A = deg2rad.(rf_angles) .* exp.(1im * deg2rad.(rf_phases))
# Load MR system
sys = KomaMRI.Scanner()
# Setup excitation pulse
TB1 = 1e-3
EX = KomaMRI.PulseDesigner.RF_hard(1 / (2π * KomaMRI.γ * TB1), TB1, sys) #α = γ B1 T
# Setup radial readout
FOV = 23e-2
RAD = my_radial_base(FOV, nsamplesperreadout, sys)
# Setup delay
DELAY = KomaMRI.Delay(0.5 * (TR - KomaMRI.dur(EX) - KomaMRI.dur(RAD)))
# MRF with rotated spokes
φ, Nφ = (√5 + 1) / 2, 7;
Δθ = π / (φ + Nφ - 1) # Tiny golden angle with Nφ = 7
# Δθ = 0
seq = (-A[1] / 2) * EX + KomaMRI.Delay(0.5 * (TR - KomaMRI.dur(EX))) + sum([A[n] * EX + DELAY + KomaMRICore.rotz((n - 1) * Δθ) * RAD + DELAY for n = 1:nTR])
seq.DEF["Nz"] = nTR # So each TR is reconstructed independently by MRIReco.jl?

sim_params = Dict{String,Any}(
    "return_type" => "mat",
    "gpu_device" => 0,
    "Δt_rf" => 1,
    "Δt" => 1,
    "precision" => "f32"
)

#Simulation parameter unpacking, and setting defaults if key is not defined
sim_params = KomaMRICore.default_sim_params(sim_params)

# # EX = KomaMRI.PulseDesigner.RF_sinc(1/(2π*KomaMRI.γ*TB1),TB1,sys) #α = γ B1 T
seqd = KomaMRICore.discretize(EX; sim_params)

## BlochSimulators.jl setup

# Assemble pSSFP sequence simulator. For this benchmark, we don't take into slice profile so we just
# set z = 0 and use a "block" excitation waveform like KomaMRI without slice select gradients.
# However, KomaMRI discretizes the bloch waveform in two steps so we do the same here,
# otherwise the relaxation effects during the RF excitation are different.

# Assemble pSSFP2D sequence struct to simulate with isochromat model
fa = rad2deg.(A) |> vec
z = SVector{1}(0.0)
nRF = 2
γΔtRF = @SVector fill(deg2rad(1.0) / nRF, nRF) # note the size nRF here
γΔtGRz = (ex=0.0, inv=0.0, pr=0.0)
Δt = (ex=EX.RF[1].T / nRF, inv=10.0^8, pr=(TR - EX.RF[1].T) / 2) # inv set to high value to effectively have no inversion prepulse

pssfp = BlochSimulators.pSSFP2D(rf_angles .* exp.(1im * deg2rad.(rf_phases)), TR, γΔtRF, Δt, γΔtGRz, z)
pssfp = gpu(f32(pssfp))

# Assemble golden angle radial trajectory
Δk = [(2π / (FOV * 100)) .* exp(im * r * Δθ) for r in 0:nTR-1]
k₀ = -(nsamplesperreadout ÷ 2) .* Δk
φ = mod.(rad2deg.(r * Δθ for r in 0:nTR-1), 360)

@assert π / maximum(abs.(k₀)) * nsamplesperreadout ≈ 100 * FOV

Δt_adc = sys.ADC_Δt

trajectory = BlochSimulators.RadialTrajectory2D(nTR, nsamplesperreadout, Δt_adc, k₀, Δk, φ)
trajectory = gpu(f32(trajectory))

## Compile and check whether output is similar

# Prepare tissue properties/phantom

N = 1000 # nr of voxels
x, y = rand(N), rand(N) # [cm]
T₁ = 0.5 .+ 2 * rand(N) # [s]
T₂ = 0.1 * T₁ .+ rand(N) # [s]
ρˣ, ρʸ = 1 .+ rand(N), zeros(N) # [a.u.]

# Simulate with BlochSimulators

parameters = gpu(f32(map(BlochSimulators.T₁T₂ρˣρʸxy, T₁, T₂, ρˣ, ρʸ, x, y)))
coil_sensitivities = gpu(f32(fill(SVector(1.0 + 0.0im), N))) # must be ones, coil_sensitivities currently not supported in KomaMRI

@time s_blochsimulators = BlochSimulators.simulate_signal(CUDALibs(), pssfp, parameters, trajectory, coil_sensitivities);

# extract signal from the only receive coil and collect on host CPU
s_blochsimulators = only.(s_blochsimulators) |> collect

# signal is complex conjugate wrt signal from komamri (probably a minus sign somewhere)
s_blochsimulators = conj(s_blochsimulators)


# Simulate with KomaMRI

_x = x / 100
_y = y / 100
_z = zeros(N)

obj = KomaMRI.Phantom{Float32}(x=_x, y=_y, z=_z, T1=T₁, T2=T₂, ρ=ρˣ);

s_komamri = KomaMRI.simulate(obj, seq, sys, sim_params=sim_params) |> vec;

# Check that outputs match

# In KomaMRI, the readout gradient strength is just slightly different at the start of the ADC
# compared to the other values during the ADC (when looking at seqd values).
# Not sure if this is intended behaviour or something I did wrong.
# When comparing the outputs I ignore these values.

remove_first_sample_readouts(x) = vec(reshape(x, nsamplesperreadout, nTR)[2:end, :])

s_blochsimulators = remove_first_sample_readouts(s_blochsimulators)
s_komamri = remove_first_sample_readouts(s_komamri)

@test s_komamri ≈ s_blochsimulators

# println("Maximum relative error: $(maximum(abs.(s_komamri - s_blochsimulators) ./ abs.(s_komamri)) * 100) [%]")
# println("Maximum absolute error: $(maximum(abs.(s_komamri - s_blochsimulators))) [a.u]")

## Benchmark

TIMINGS = (komamri=[], blochsimulators=[])

N_range = 10_000:10_000:350_000
# N_range = [10000,350000]

for N in N_range

    # Prepare tissue properties/phantom
    x, y = rand(N), rand(N)
    T₁, T₂ = ones(N), ones(N) / 10
    ρˣ, ρʸ = rand(N), zeros(N)

    # KomaMRI
    obj = KomaMRI.Phantom{Float32}(x=x / 100, y=y / 100, z=zeros(N), T1=T₁, T2=T₂, ρ=ρˣ)

    sim_params["return_type"] = "raw" # needed to extract simulation time

    komamri_output = KomaMRI.simulate(obj, seq, sys; sim_params)

    t = komamri_output.params["userParameters"]["sim_time_sec"]
    push!(TIMINGS.komamri, t)

    # BlochSimulators
    parameters = gpu(f32(map(BlochSimulators.T₁T₂ρˣρʸxy, T₁, T₂, ρˣ, ρʸ, x, y)))
    coil_sensitivities = gpu(f32(fill(SVector(1.0 + 0.0im), N)))

    t = @elapsed begin
        @time s_blochsimulators = BlochSimulators.simulate_signal(CUDALibs(), pssfp, parameters, trajectory, coil_sensitivities)
    end

    push!(TIMINGS.blochsimulators, t)

    # Compare output

    sim_params["return_type"] = "mat" # needed to extract signal

    s_komamri = KomaMRI.simulate(obj, seq, sys; sim_params) |> vec

    s_blochsimulators = only.(s_blochsimulators) |> collect |> conj

    s_blochsimulators = remove_first_sample_readouts(s_blochsimulators)
    s_komamri = remove_first_sample_readouts(s_komamri)

    @test s_blochsimulators ≈ s_komamri
end

## PGFPlot

p = @pgf Axis(
    {
        xlabel = "Size of phantom (voxels)",
        ylabel = "Runtime (s)",
        title = "{\\\\ \\textbf{Forward model evaluations on GPU}\\\\ (gradient-balanced transient-state sequence)\\\\ with golden angle radial readouts)}",
        grid = "both",
        ymode = "log",
        xmode = "log",
        minor_grid_style = "{dotted,gray!50}",
        major_grid_style = "{gray!50}",
        legend_pos = "outer north east",
        label_style = "{font=\\Large}",
        title_style = "{align=center, font=\\Large}",
        xtick = "{10000,100000,350000}",
        xticklabels = "{\$10^4\$,\$10^5\$,\$3.5\\times10^5\$}",
        xmin = "0",
        ymax = "100",
        legend_style = "{at={(0.5,-0.25)},anchor=north}",
        legend_cell_align = "{left}"
    },
    PlotInc(
        {color = colors[2], mark = "o", style = {thick}},
        Coordinates(collect(zip((N_range), (TIMINGS.blochsimulators))))
    ),
    LegendEntry("BlochSimulators.jl (pSSFP2D + RadialTrajectory2D)"),
    PlotInc(
        {color = colors[1], mark = "o", style = {thick}},
        Coordinates(collect(zip((N_range), (TIMINGS.komamri))))
    ),
    LegendEntry("KomaMRI.jl")
)

# SAVEFIGS && @save "figures/TIMINGS_4_forward_gpu.jld2" TIMINGS
# SAVEFIGS && pgfsave("figures/4_forward_gpu.pdf", p, dpi=300)

p

## Repeat benchmark but now with sinc pulse and different z-locations

# Generic parameters

nTR = 500; # KomaMRI crashes for some nr of voxels when nTR = 1000 or 1120
nsamplesperreadout = 224;

nRF = 40
nz = 15

# KomaMRI.jl setup
# Flip angle train (with 0-π phase cycling)
rf_angles = LinRange(1.0, 50.0, nTR) |> collect # degrees
# rf_angles .= 90
rf_phases = repeat([0.0, 180.0], nTR ÷ 2) # degrees
# rf_phases .= 0
TR = 0.015
A = deg2rad.(rf_angles) .* exp.(1im * deg2rad.(rf_phases))
# Load MR system
sys = KomaMRI.Scanner()
# Setup excitation pulse
TB1 = 1e-3
EX = KomaMRI.PulseDesigner.RF_sinc(1 / (2π * KomaMRI.γ * TB1), TB1, sys) #α = γ B1 T
# Setup radial readout
FOV = 23e-2
RAD = my_radial_base(FOV, nsamplesperreadout, sys)
# Setup delay
DELAY = KomaMRI.Delay(0.5 * (TR - KomaMRI.dur(EX) - KomaMRI.dur(RAD)))
# MRF with rotated spokes
φ, Nφ = (√5 + 1) / 2, 7;
Δθ = π / (φ + Nφ - 1) # Tiny golden angle with Nφ = 7
# Δθ = 0
seq = (-A[1] / 2) * EX + KomaMRI.Delay(0.5 * (TR - KomaMRI.dur(EX))) + sum([A[n] * EX + DELAY + KomaMRICore.rotz((n - 1) * Δθ) * RAD + DELAY for n = 1:nTR])
seq.DEF["Nz"] = nTR # So each TR is reconstructed independently by MRIReco.jl?

sim_params = Dict{String,Any}(
    "return_type" => "mat",
    "gpu_device" => 0,
    "Δt_rf" => EX.RF.dur[1] / nRF,
    "Δt" => 1,
    "precision" => "f32"
)

#Simulation parameter unpacking, and setting defaults if key is not defined
sim_params = KomaMRICore.default_sim_params(sim_params)

# # EX = KomaMRI.PulseDesigner.RF_sinc(1/(2π*KomaMRI.γ*TB1),TB1,sys) #α = γ B1 T
seqd = KomaMRICore.discretize(EX; sim_params)

# BlochSimulators.jl setup

# Assemble pSSFP sequence simulator. For this benchmark, we don't take into slice profile so we just
# set z = 0 and use a "block" excitation waveform like KomaMRI without slice select gradients.
# However, KomaMRI discretizes the bloch waveform in two steps so we do the same here,
# otherwise the relaxation effects during the RF excitation are different.

# Assemble pSSFP2D sequence struct to simulate with isochromat model
fa = rad2deg.(A) |> vec

# z locations for slice profile 
z = range(start=-0.5, stop=0.5, length=nz)
γΔtRF = SVector{nRF}((2π) * KomaMRI.γ * real.(seqd.B1[3:end-3]) .* EX.RF.dur[1] / nRF .|> deg2rad)
γΔtGRz = (ex=0.0, inv=0.0, pr=0.0)
Δt = (ex=EX.RF[1].T / nRF, inv=10.0^8, pr=(TR - EX.RF[1].T) / 2) # inv set to high value to effectively have no inversion prepulse

pssfp = BlochSimulators.pSSFP2D(rf_angles .* exp.(1im * deg2rad.(rf_phases)), TR, γΔtRF, Δt, γΔtGRz, SVector{nz}(z))
pssfp = gpu(f32(pssfp))

# Assemble golden angle radial trajectory
Δk = [(2π / (FOV * 100)) .* exp(im * r * Δθ) for r in 0:nTR-1]
k₀ = -(nsamplesperreadout ÷ 2) .* Δk
φ = mod.(rad2deg.(r * Δθ for r in 0:nTR-1), 360)

@assert π / maximum(abs.(k₀)) * nsamplesperreadout ≈ 100 * FOV

Δt_adc = sys.ADC_Δt

trajectory = BlochSimulators.RadialTrajectory2D(nTR, nsamplesperreadout, Δt_adc, k₀, Δk, φ)
trajectory = gpu(f32(trajectory))

## Compile and check whether output is similar

# Prepare tissue properties/phantom

N = 100 # nr of voxels
x, y = rand(N), rand(N) # [cm]
T₁ = 0.5 .+ 2 * rand(N) # [s]
T₂ = 0.1 * T₁ .+ rand(N) # [s]
ρˣ, ρʸ = 1 .+ rand(N), zeros(N) # [a.u.]

# Simulate with BlochSimulators

parameters = gpu(f32(map(BlochSimulators.T₁T₂ρˣρʸxy, T₁, T₂, ρˣ, ρʸ, x, y)))
coil_sensitivities = gpu(f32(fill(SVector(1.0 + 0.0im), N))) # must be ones, coil_sensitivities currently not supported in KomaMRI

@time s_blochsimulators = BlochSimulators.simulate_signal(CUDALibs(), pssfp, parameters, trajectory, coil_sensitivities);

# extract signal from the only receive coil and collect on host CPU
s_blochsimulators = only.(s_blochsimulators) |> collect

# signal is complex conjugate wrt signal from komamri (probably a minus sign somewhere)
s_blochsimulators = conj(s_blochsimulators)

# Simulate with KomaMRI

_z = repeat([z...], N) / 100
_x = repeat(x, inner=nz) / 100
_y = repeat(y, inner=nz) / 100
_T₁ = repeat(T₁, inner=nz)
_T₂ = repeat(T₂, inner=nz)
_ρˣ = repeat(ρˣ, inner=nz)

obj = KomaMRI.Phantom{Float32}(x=_x, y=_y, z=_z, T1=_T₁, T2=_T₂, ρ=_ρˣ);

sim_params["return_type"] = "mat"

s_komamri = KomaMRI.simulate(obj, seq, sys, sim_params=sim_params) |> vec;

# Time 

N = 100000 # nr of voxels
x, y = rand(N), rand(N) # [cm]
T₁ = 0.5 .+ 2 * rand(N) # [s]
T₂ = 0.1 * T₁ .+ rand(N) # [s]
ρˣ, ρʸ = 1 .+ rand(N), zeros(N) # [a.u.]

# KomaMRI
_z = repeat([z...], N) / 100
_x = repeat(x, inner=nz) / 100
_y = repeat(y, inner=nz) / 100
_T₁ = repeat(T₁, inner=nz)
_T₂ = repeat(T₂, inner=nz)
_ρˣ = repeat(ρˣ, inner=nz)

obj = KomaMRI.Phantom{Float32}(x=_x, y=_y, z=_z, T1=_T₁, T2=_T₂, ρ=_ρˣ);

sim_params["return_type"] = "raw" # needed to extract simulation time

komamri_output = KomaMRI.simulate(obj, seq, sys; sim_params);

t_komamri = komamri_output.params["userParameters"]["sim_time_sec"]

# BlochSimulators
parameters = gpu(f32(map(BlochSimulators.T₁T₂ρˣρʸxy, T₁, T₂, ρˣ, ρʸ, x, y)))
coil_sensitivities = gpu(f32(fill(SVector(1.0 + 0.0im), N)))

t_blochsimulators = @elapsed begin
    @time s_blochsimulators = BlochSimulators.simulate_signal(CUDALibs(), pssfp, parameters, trajectory, coil_sensitivities)
end

println("KomaMRI: $t_komamri [s]\nBlochSimulators: $t_blochsimulators [s]")