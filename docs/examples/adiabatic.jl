## Adiabatic inversion example

using Pkg; Pkg.activate("docs")

# In this example we demonstrate how to simulate an adiabatic inversion 
# pulse using the AdiabaticInversion implementation in BlochSimulators.jl

using BlochSimulators
using StructArrays, ComputationalResources
using PythonPlot

# using KomaMRI

# Parameters taken from Bernstein - Handbook of MRI Pulse Sequences, p. 196
γ = 267.52218744e6
T = 8e-3 # s
t = LinRange(-T/2,T/2,1000)
Δt = first(diff(t))
TΔf = 10 # ?    
β = 800 # rad/s
A₀ = 14e-6 # T
μ = 4.9 # 

# Amplitude modulation
A = @. A₀ * sech(β*t)
γΔtA = γ * Δt * A

# Frequency modulation
Δω = @. -μ*β*tanh(β*t)
Δf = Δω / 2π

# Assemble "sequence" 
sequence = BlochSimulators.AdiabaticInversion(γΔtA, Δω, Δt)

# Set parameters
B₀ = -2000:2000
parameters = T₁T₂B₀.(1.0,.1, B₀)

# Perform simulations
@time m = simulate_magnetization(CPU1(), sequence, parameters) |> StructArray |> vec

# Plot results
figure();
plot(B₀,m.z); 
plot(B₀,abs.(complex.(m.x,m.y))); 
legend(["mz", "mxy"]); 
ylim([-1,1])
title("Adiabatic inversion for different off-resonance values")
