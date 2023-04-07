using BlochSimulators
using ComputationalResources
using StaticArrays

nr = 500
ns = 224
nv = 350000
sequence = FISP(nr);
trajectory = CartesianTrajectory(nr,ns);
resource = CPU1()
parameters = [T₁T₂ρˣρʸxy(2.0,0.2,1.0,0.0,0.1,0.1) for i = 1:nv]
coil_sensitivities = [SVector{1,ComplexF64}(ones(1)...) for i = 1:nv]

trajectory = BlochSimulators.CartesianTrajectory(
    nr,
    ns,
    1e-6,
    ones(ComplexF64, nr),
    ones(ComplexF64, nr),
    rand(nr)
)

# matrix with slice profile correction factors
nz = 3
spc =
max_state = 5

sequence = BlochSimulators.FISP(complex.(90 * ones(nr)), complex.(ones(nr, nz)), 0.008,0.004, 5, 0.01)

sequence_d, parameters_d, trajectory_d, coil_sensitivities_d = gpu(f32(sequence)), gpu(f32(parameters)), gpu(f32(trajectory)), gpu(f32(coil_sensitivities))


@time e1 = BlochSimulators.simulate(resource, sequence, parameters);
@time s1 = BlochSimulators.simulate(CPU1(), sequence, parameters, trajectory, coil_sensitivities);
# @time e2 = BlochSimulators.simulate(CUDALibs(), gpu(sequence), gpu(parameters));

@time e2 = BlochSimulators.simulate(CUDALibs(), sequence_d, parameters_d);
@time s2 = BlochSimulators.simulate(CUDALibs(),sequence_d, parameters_d, trajectory_d, coil_sensitivities_d);


