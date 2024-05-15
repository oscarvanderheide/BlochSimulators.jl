using Test
using BlochSimulators

# load general packages
using CUDA
using LinearAlgebra
using StaticArrays
using ComputationalResources
using Distributed
using DistributedArrays

# test some individual functions in BlochSimulators
@testset "Test operator functions for isochromat model" begin

    # create single spin isochromat
    m = BlochSimulators.Isochromat(rand(3)...)

    # check initial conditions
    m = BlochSimulators.initial_conditions(m)
    @test m == BlochSimulators.Isochromat(0.0, 0.0, 1.0)

    # test inversion
    # "adiabatic inversion"
    @test BlochSimulators.invert(m) == BlochSimulators.Isochromat(0.0, 0.0, -1.0)

    # "
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    @test BlochSimulators.invert(m, p) == BlochSimulators.Isochromat(0.0, 0.0, -1.0)

    # failed 90 degree inversion
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.5)
    @test abs(BlochSimulators.invert(m, p).z) < eps()

    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    @test BlochSimulators.invert(m, p).z == 1.0

    # test decay
    m = BlochSimulators.Isochromat(1.0, 1.0, -1.0)
    Δt = 1e7
    E₁ = exp(-Δt / 0.8)
    E₂ = exp(-Δt / 0.05)
    # really long Δt so everything should be zero after decay
    m = BlochSimulators.decay(m, E₁, E₂)
    @test all(iszero.(m))

    # test regrowth

    m = BlochSimulators.regrowth(m, E₁)
    # really long Δt so z component should be 1
    @test m == BlochSimulators.initial_conditions(m)

    # rotate

    z = 0.0
    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 0.0)
    γΔtGRz = 0.0
    Δt = 0.0

    m = BlochSimulators.Isochromat(0.0, 0.0, 1.0)

    # rotate over x-axis
    γΔtRF = π / 2 + 0.0im
    mx = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mx.y == -1.0 && abs(mx.z) < eps()

    # rotate over y-axis
    γΔtRF = 0.0 + im * π / 2
    my = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test my.x == -1.0 && abs(my.z) < eps()

    # rotate over z-axis
    γΔtRF = 0.0 + 0.0 * im
    γΔtGRz = π / 2
    z = 1.0
    m = BlochSimulators.Isochromat(1.0, 0.0, 0.0)
    mz = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mz.y == -1.0 && abs(mz.x) < eps()

    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 1.0)
    γΔtGRz = 0.0
    Δt = 0.5
    mz = BlochSimulators.rotate(m, γΔtRF, γΔtGRz, z, Δt, p)
    @test mz.x == -1.0 && abs(mz.y) < eps()

    # also test the rotate function without RF argument

    γΔtGRz = π / 2
    z = 1.0
    m = BlochSimulators.Isochromat(1.0, 0.0, 0.0)
    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 0.0)
    mz = BlochSimulators.rotate(m, γΔtGRz, z, Δt, p)
    @test mz.y == -1.0 && abs(mz.x) < eps()

    p = BlochSimulators.T₁T₂B₁B₀(1.0, 0.1, 1.0, 1.0)
    γΔtGRz = 0.0
    Δt = 0.5
    mz = BlochSimulators.rotate(m, γΔtGRz, z, Δt, p)
    @test mz.x == -1.0 && abs(mz.y) < eps()

end

# test some individual functions in BlochSimulators
@testset "Test operator functions for EPG model" begin

    # create single spin isochromat
    Ω = @MMatrix zeros(ComplexF64, 3, 20)

    # check initial conditions
    BlochSimulators.initial_conditions!(Ω)
    @test Ω[3, 1] == 1.0 + 0.0im
    @test sum(Ω) == 1.0 + 0.0im

    # test inversion

    # "adiabatic inversion"
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω)
    @test Ω[3, 1] == -1

    # "non-adiabatic inversion"
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test Ω[3, 1] == -1

    # failed 90 degree inversion
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.5)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test abs(Ω[3, 1]) < eps()

    # 0 B1 so 0 effect
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.invert!(Ω, p)
    @test Ω[3, 1] == 1

    # check "adiabatic inversion" for higher order states as well
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.Z(Ω) .= 0.1
    BlochSimulators.invert!(Ω)
    @test all(Ω[3, :] .== complex(-0.1))

    # test decay
    Ω = @MMatrix rand(ComplexF64, 3, 20)
    Δt = 1e7
    E₁ = exp(-Δt / 0.8)
    E₂ = exp(-Δt / 0.05)
    # really long Δt so everything should be zero after decay
    BlochSimulators.decay!(Ω, E₁, E₂)
    @test all(iszero.(Ω))

    # test regrowth

    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.decay!(Ω, E₁, E₂)
    BlochSimulators.regrowth!(Ω, E₁)
    # really long Δt so z component should be 1
    @test Ω[3, 1] == 1.0

    # rotate

    RF = complex(90.0)
    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 1.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, RF, p)
    @test Ω[1, 1] == -im && Ω[2, 1] == conj(Ω[1, 1])

    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, complex(45.0), p)
    @test Ω[1, 1] ≈ -(√2 / 2) * im && Ω[2, 1] == conj(Ω[1, 1])

    p = BlochSimulators.T₁T₂B₁(1.0, 0.1, 0.0)
    BlochSimulators.initial_conditions!(Ω)
    BlochSimulators.excite!(Ω, complex(90.0), p)
    @test Ω[1, 1] ≈ 0 && Ω[2, 1] == conj(Ω[1, 1])

    # dephasing

    # dont act on Z states
    Ω = @MMatrix rand(ComplexF64, 3, 20)
    Ω2 = copy(Ω)
    BlochSimulators.dephasing!(Ω)
    @test Ω[3, :] == Ω2[3, :]

    # check "left boundary"
    BlochSimulators.initial_conditions!(Ω)
    Ω[1, 1] = 1
    Ω[2, 1] = 2
    Ω[2, 2] = 3im
    BlochSimulators.dephasing!(Ω)
    @test Ω[1, 2] == 1 && Ω[2, 1] == 3im && Ω[1, 1] == conj(Ω[2, 1])

    # check "right boundary"
    Ω .= reshape(1:60, 3, 20)
    Ω[2, 1] = 1
    Ω[2, 2] = 5im
    Ωpre = copy(Ω)
    BlochSimulators.dephasing!(Ω)
    @test Ω[1, end] == Ωpre[1, end-1]
    @test Ω[2, end-1] == Ωpre[2, end]
    @test Ω[3, end] == Ωpre[3, end]

    # spoil

    # check that transverse states are nulled
    Ω = @MMatrix rand(ComplexF64, 3, 20)
    BlochSimulators.spoil!(Ω)
    @test all(Ω[1, :] .== complex(0.0))
    @test all(Ω[2, :] .== complex(0.0))

end

@testset "Test functions to change precision" begin

    # f32 and f64 should recursively go through nested structures and convert floating point numbers only

    # first, test some complicated but random nested structure
    x = [1, 2.0, 3.0f0, [4.0, 5.0], 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

    @test f64(x) == x
    @test f32(x) == [1, 2.0f0, 3.0f0, [4.0f0, 5.0f0], 6.0f0im, (7.0f0, 8.0f0im), (a=9.0f0, b=10.0f0im)]

    # test FISP sequence struct
    nTR = 10
    s = FISP2D(nTR)

    @test f64(s) == s

    @test f32(s).RF_train == ComplexF32.(s.RF_train)
    @test f32(s).sliceprofiles == ComplexF32.(s.sliceprofiles)
    @test f32(s).TR == Float32(s.TR)
    @test f32(s).TE == Float32(s.TE)
    @test f32(s).TI == Float32(s.TI)
    @test f32(s).max_state == s.max_state

    # test Cartesian trajectory struct
    t = CartesianTrajectory(nTR, 100)

    @test f64(t) == t

    @test f32(t).nreadouts == t.nreadouts
    @test f32(t).nsamplesperreadout == t.nsamplesperreadout
    @test f32(t).Δt == Float32(t.Δt)
    @test f32(t).k_start_readout == ComplexF32.(t.k_start_readout)
    @test f32(t).Δk_adc == ComplexF32.(t.Δk_adc)
    @test f32(t).py == t.py

    # test AbstractTissueParameters
    p = T₁T₂B₁B₀(1.0, 2.0, 3.0, 4.0)
    f32(p) == T₁T₂B₁B₀(1.0f0, 2.0f0, 3.0f0, 4.0f0)
    f64(f32(p)) == p

end

@testset "Test functions to move to gpu" begin

    # first, test some complicated but random nested structure
    x = [1, 2.0, 3.0f0, [4.0, 5.0], 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

    @test gpu(x) == [1, 2.0, 3.0f0, CuArray([4.0, 5.0]), 6.0im, (7.0, 8.0im), (a=9.0, b=10.0im)]

    # test FISP sequence struct
    nTR = 10
    s = FISP2D(nTR)

    @test gpu(s).RF_train == CuArray(s.RF_train)
    @test gpu(s).sliceprofiles == CuArray(s.sliceprofiles)
    @test gpu(s).TR == s.TR
    @test gpu(s).TE == s.TE
    @test gpu(s).TI == s.TI
    @test gpu(s).max_state == s.max_state

    # test Cartesian trajectory struct
    t = CartesianTrajectory(nTR, 100)

    @test gpu(t).nreadouts == t.nreadouts
    @test gpu(t).nsamplesperreadout == t.nsamplesperreadout
    @test gpu(t).Δt == t.Δt
    @test gpu(t).k_start_readout == CuArray(t.k_start_readout)
    @test gpu(t).Δk_adc == t.Δk_adc
    @test gpu(t).py == CuArray(t.py)
    @test gpu(t).readout_oversampling == t.readout_oversampling

    # test AbstractTissueParameters
    p = T₁T₂B₁B₀(1.0, 2.0, 3.0, 4.0)
    @test gpu(p) == p
    @test gpu([p]) == CuArray([p])

end

@testset "Test dictionary generation on different computational resources" begin

    # Simulate dictionary with CPU1()
    nTR = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂(rand(), rand()) for _ = 1:1000]

    magnetization_cpu1 = simulate_magnetization(CPU1(), sequence, parameters)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    magnetization_cputhreads = simulate_magnetization(CPUThreads(), sequence, parameters)
    @test magnetization_cpu1 ≈ magnetization_cputhreads

    # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # and check if outcome is the same
    if workers() == [1]
        addprocs(2, exeflags="--project=.")
        @everywhere using BlochSimulators, ComputationalResources
    end

    magnetization_cpuprocesses = simulate_magnetization(CPUProcesses(), sequence, parameters)

    @test magnetization_cpu1 ≈ convert(Array, magnetization_cpuprocesses)

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        magnetization_cudalibs = simulate_magnetization(CUDALibs(), gpu(sequence), gpu(parameters)) |> collect
        @test magnetization_cpu1 ≈ magnetization_cudalibs
    end

end

@testset "Test for SpokesTrajectory" begin

    # Simulate magnetization at echo times for a single voxel with coordinates (0,0,0)
    nTR = 100
    sequence = FISP2D(nTR)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, 1.0, 0.0)]
    coordinates = [Coordinates(0.0, 0.0, 0.0)]

    d = simulate_magnetization(CPU1(), sequence, parameters)

    # Use some trajectory to simulate signal
    nr, ns = 100, 40
    trajectory = CartesianTrajectory(nr, ns)

    s = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates)

    # Because this voxel has x = y = 0, a gradient trajectory
    # should not influence it and for one voxel there's no
    # volume integral either
    @test d ≈ vec(s[(ns÷2)+1,:])

    # If we now simulate with a voxel with x and y non-zero, then
    # at echo time the magnetization should be the same in abs
    s2 = simulate_signal(CPU1(), sequence, [T₁T₂ρˣρʸ(1.0, 0.1, 1.0, 0.0)], trajectory, coordinates)

    @test abs.(d) ≈ abs.(vec(s2[(ns÷2)+1,:]))

end

@testset "Tests for CartesianTrajectory" begin

    # constants
    nr = 100
    ns = 128
    Δt = 4e-6 # s
    fovx = 10.0 # cm
    fovy = 10.0 # cm
    Δkˣ = 2π / fovx
    Δkʸ = 2π / fovy
    py = collect(-50:49)
    k0 = [(-ns / 2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nr]
    os = 1

    # assemble Cartesian trajectory
    cartesian = CartesianTrajectory(nr, ns, Δt, k0, Δkˣ, py, os)

    # test whether getindex method to reduce sequence length works
    @test cartesian[1:50].k_start_readout == CartesianTrajectory(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).k_start_readout
    @test cartesian[1:50].Δk_adc == CartesianTrajectory(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).Δk_adc
    @test cartesian[1:50].py == CartesianTrajectory(50, ns, Δt, k0[1:50], Δkˣ, py[1:50], os).py

    @test BlochSimulators.sampling_mask(cartesian)[10] == CartesianIndices((1:ns, 10:10))

    @test BlochSimulators.kspace_coordinates(cartesian)[:, 1] == k0[1] .+ (collect(0:ns-1) .* Δkˣ)
end

@testset "Tests for RadialTrajectory" begin

    # assemble radial trajectory
    nr = 100
    ns = 128
    Δt = 4e-6 # s
    os = 2  # factor two oversampling
    fovx = 10.0 # cm
    fovy = 10.0 # cm
    φ = π / ((√5 + 1) / 2) # golden angle of ~111 degrees
    Δkˣ = 2π / (os * fovx)
    φ = collect(φ .* (0:nr-1))
    k0 = -(ns / 2) * Δkˣ + 0.0im
    k0 = collect(@. exp(im * φ) * k0)
    Δk = Δkˣ + 0.0im
    Δk = collect(@. exp(im * φ) * Δk)
    os = 1

    radial = RadialTrajectory(nr, ns, Δt, k0, Δk, φ, 1)

    # test whether getindex method to reduce sequence length works
    @test radial[1:50].k_start_readout == RadialTrajectory(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).k_start_readout
    @test radial[1:50].Δk_adc == RadialTrajectory(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).Δk_adc
    @test radial[1:50].φ == RadialTrajectory(50, ns, Δt, k0[1:50], Δk[1:50], φ[1:50], os).φ

    # test gradient delay for radial
    S = rand(2, 2)
    radial_delay = deepcopy(radial)
    BlochSimulators.add_gradient_delay!(radial_delay, S)
    @test all(radial_delay.k_start_readout .== BlochSimulators.add_gradient_delay(radial, S).k_start_readout)

end

@testset "Signal simulation sanity checks" begin

    # if magnetization .= 1, x and y .= 0, coil_sensitivities .= 1, T₂ .= Inf,
    # then signal should simply be the nr of voxels at all time points
    nv = 100 # voxels
    nr = 100 # readouts
    ns = 10  # samples per readout
    nc = 1 # number of coils

    magnetization = complex.(ones(nr, nv))
    parameters = fill(T₁T₂ρˣρʸ(Inf, Inf, 1.0, 0.0), nv)
    coordinates = fill(Coordinates(0.0, 0.0, 0.0), nv)
    trajectory = RadialTrajectory(nr, ns)
    coil_sensitivities = complex(ones(nv, nc))
    resource = CPU1()

    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)
  
    @test signal == fill(nv, ns * nr, nc)

    # if proton density is 0, then signal should be 0

    parameters = fill(T₁T₂ρˣρʸ(rand(), rand(), 0.0, 0.0), nv)

    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    @test signal == zeros(ns * nr, nc)

    # if coil sensitivities are 0 everywhere, then signal should be 0

    parameters = fill(T₁T₂ρˣρʸ(rand(4)...), nv)
    nc = 4
    coil_sensitivities = complex(zeros(nv, nc))
    signal = magnetization_to_signal(resource, magnetization, parameters, trajectory, coordinates, coil_sensitivities)

    @test signal == zeros(ns * nr, nc)
    
end

@testset "Test Cartesian signal simulations on different computational resources" begin

    # Simulate signal with CPU1() as reference
    nTR = 1000
    nvoxels = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, rand(2)...) for _ = 1:nvoxels]

    trajectory = CartesianTrajectory(nTR, 100)
    nc = 2
    coil_sensitivities = rand(ComplexF64, nvoxels, nc)
    coordinates = [Coordinates(rand(3)...) for _ = 1:nvoxels]
    signal_cpu1 = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates, coil_sensitivities)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    signal_cputhreads = simulate_signal(CPUThreads(), sequence, parameters, trajectory, coordinates, coil_sensitivities)
    @test signal_cpu1 ≈ signal_cputhreads

    # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # and check if outcome is the same
    if workers() == [1]
        addprocs(2, exeflags="--project=.")
        @everywhere using BlochSimulators, ComputationalResources, DistributedArrays
    end

    signal_cpuprocesses = simulate_signal(CPUProcesses(), sequence, distribute(parameters), trajectory, distribute(coordinates), distribute(coil_sensitivities))
    @test signal_cpu1 ≈ signal_cpuprocesses

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        signal_cudalibs = simulate_signal(CUDALibs(), gpu(sequence), gpu(parameters), gpu(trajectory), gpu(coordinates), gpu(coil_sensitivities))
        @test signal_cpu1 ≈ convert(Array, signal_cudalibs)
    end

end

@testset "Test radial signal simulations on different computational resources" begin

    # Simulate signal with CPU1() as reference
    nTR = 1000
    nvoxels = 1000
    sequence = FISP2D(nTR)
    sequence.sliceprofiles[:, :] .= rand(ComplexF64, nTR, 3)
    parameters = [T₁T₂ρˣρʸ(1.0, 0.1, rand(2)...) for _ = 1:nvoxels]

    trajectory = RadialTrajectory(nTR, 100)
    nc = 2
    coil_sensitivities = rand(ComplexF64, nvoxels, nc)
    coordinates = [Coordinates(rand(3)...) for _ = 1:nvoxels]
    signal_cpu1 = simulate_signal(CPU1(), sequence, parameters, trajectory, coordinates, coil_sensitivities)

    # Now simulate with CPUThreads() (multi-threaded CPU) and check if outcome is the same
    signal_cputhreads = simulate_signal(CPUThreads(), sequence, parameters, trajectory, coordinates, coil_sensitivities)
    @test signal_cpu1 ≈ signal_cputhreads

    # Now add workers and simulate with CPUProcesses() (distributed CPU)
    # and check if outcome is the same
    if workers() == [1]
        addprocs(2, exeflags="--project=.")
        @everywhere using BlochSimulators, ComputationalResources, DistributedArrays
    end

    signal_cpuprocesses = simulate_signal(CPUProcesses(), sequence, distribute(parameters), trajectory, distribute(coordinates), distribute(coil_sensitivities))
    @test signal_cpu1 ≈ signal_cpuprocesses

    if CUDA.functional()
        # Simulate with CUDALibs() (GPU) and check if outcome is the same
        signal_cudalibs = simulate_signal(CUDALibs(), gpu(sequence), gpu(parameters), gpu(trajectory), gpu(coordinates), gpu(coil_sensitivities))
        @test signal_cpu1 ≈ convert(Array, signal_cudalibs)
    end

end