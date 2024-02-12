# custom colors used in plotting
colors = [(0.0,0.447,0.741); (0.850,0.325,0.098); (0.929,0.694,0.125); (0.494,0.184,0.556); (0.466,0.674,0.188)]
colors = [RGB(x...) for x in colors]

## Convert pSSFP (from BlochSimulators.jl) to Generic2D format (from BlochSimulators.jl)
function pssfp2d_to_generic2d(pssfp::BlochSimulators.pSSFP2D)

    γ = 26753.0

    RF = ComplexF64[]
    GRx = Float64[]
    GRy = Float64[]
    GRz = Float64[]
    Δt = Float64[]
    ADC = Bool[]

    # excitation
    RF_excitation  = Array(pssfp.γΔtRF ./ (γ * pssfp.Δt.ex))
    GRx_excitation = zero(RF_excitation)
    GRy_excitation = zero(RF_excitation)
    GRz_excitation = fill(pssfp.γΔtGRz.ex / (γ * pssfp.Δt.ex), length(RF_excitation))
    Δt_excitation = fill(pssfp.Δt.ex, length(RF_excitation))
    ADC_excitation = zeros(Bool, length(RF_excitation))

    # free precession
    RF_precession = zeros(2)
    Δt_precession = fill(pssfp.Δt.pr, 2)
    GRx_precession = zeros(2)
    GRy_precession = zeros(2)
    GRz_precession = fill(pssfp.γΔtGRz.pr / (γ * pssfp.Δt.pr), 2)
    ADC_precession = [true; false]

    # START ASSEMBLING SEQUENCE

    # inversion done using initial guess m = [0,0,-1]
    append!(Δt, pssfp.Δt.inv)
    append!(RF, 0.0 + 0.0im)
    append!(GRx, 0.0)
    append!(GRy, 0.0)
    append!(GRz, 0.0)
    append!(ADC, false)

    # apply "alpha over two" pulse
    α₀ = -pssfp.RF_train[1]/2
    append!(RF, α₀*RF_excitation)
    append!(Δt, Δt_excitation)
    append!(GRx, GRx_excitation)
    append!(GRy, GRy_excitation)
    append!(GRz, GRz_excitation)
    append!(ADC, ADC_excitation)

    # Free precession part: slice select pre and rephaser and precession until RF
    append!(RF, RF_precession[1])
    append!(Δt, Δt_precession[1])
    append!(GRx, 0)
    append!(GRy, 0)
    append!(GRz, 2*GRz_precession[1])
    append!(ADC, false)

    # ADC = zeros(Bool, length(RF))

    for r in 1:length(pssfp.RF_train)

        # RF excitation
        α = pssfp.RF_train[r]
        append!(RF, α*RF_excitation)
        append!(Δt, Δt_excitation)
        append!(GRx, GRx_excitation)
        append!(GRy, GRy_excitation)
        append!(GRz, GRz_excitation)
        append!(ADC, ADC_excitation)
        # Remainder of the TR for which RF is off
        append!(RF, RF_precession)
        append!(Δt, Δt_precession)
        append!(ADC, ADC_precession)
        append!(GRx, GRx_precession)
        append!(GRy, GRy_precession)
        append!(GRz, GRz_precession)

    end

    GR = permutedims([GRx ;; GRy ;; GRz])
    samples = BitVector(ADC)

    return BlochSimulators.Generic2D(RF, GR, samples, Δt, collect(pssfp.z))
end

## Convert Generic2D format (from BlochSimulators.jl) to Hargreaves' format
function generic2d_to_hargreaves(generic::BlochSimulators.Generic2D, parameters::AbstractArray{<:BlochSimulators.T₁T₂xy})

    b1real = -real.(generic.RF)
    b1imag = imag.(generic.RF)
    xgrad = generic.GR[1,:]
    ygrad = generic.GR[2,:]
    zgrad = generic.GR[3,:]
    tsteps = generic.Δt
    t1 = parameters[1].T₁
    t2 = parameters[1].T₂
    dfreq = [zero(t1)]
    dxpos = [p.x for p in parameters]
    dypos = [p.y for p in parameters]
    dzpos = zero(dxpos)
    sample = generic.sample

    return b1real, b1imag, xgrad, ygrad, zgrad, tsteps, t1, t2, dfreq, dxpos, dypos, dzpos, sample
end

## Convert Generic2D format (from BlochSimulators.jl) to Generic3D format (from BlochSimulators.jl)

function generic2d_to_generic3d(generic::BlochSimulators.Generic2D)
    return BlochSimulators.Generic3D(generic.RF, generic.GR, generic.sample, generic.Δt)
end

## Convert Generic3D format (from BlochSimulators.jl) to Hargreaves' format
function generic3d_to_hargreaves(generic::BlochSimulators.Generic3D, parameters::AbstractArray{<:BlochSimulators.T₁T₂xyz})

    b1real = -real.(generic.RF)
    b1imag = imag.(generic.RF)
    xgrad = generic.GR[1,:]
    ygrad = generic.GR[2,:]
    zgrad = generic.GR[3,:]
    tsteps = generic.Δt
    t1 = parameters[1].T₁
    t2 = parameters[1].T₂
    dfreq = [zero(t1)]
    dxpos = [p.x for p in parameters]
    dypos = [p.y for p in parameters]
    dzpos = [p.z for p in parameters]
    sample = generic.sample

    return b1real, b1imag, xgrad, ygrad, zgrad, tsteps, t1, t2, dfreq, dxpos, dypos, dzpos, sample
end

## Convert Generic3D format (from BlochSimulators.jl) to Hargreaves' format
function generic3d_to_komamri(generic::BlochSimulators.Generic3D, parameters::AbstractArray{<:BlochSimulators.T₁T₂xyz})

    Gx = generic.GR[1,:]
    Gy = generic.GR[2,:]
    Gz = generic.GR[3,:]
    B1 = generic.RF
    Δf = zero(Gx)
    ADCflag = generic.sample

    Δt = generic.Δt[1:end-1]
    t = cumsum(Δt)

    t = [0.0; t]

    x = [p.x for p in parameters]
    y = [p.y for p in parameters]
    z = [p.z for p in parameters]
    T₁ = [p.T₁ for p in parameters]
    T₂ = [p.T₂ for p in parameters]

    seqd = KomaMRI.DiscreteSequence(Gx, Gy, Gz, complex.(B1), Δf, ADCflag, t, Δt)
    obj = KomaMRI.Phantom{Float64}(x=x, y=y, z=z, T1=T₁, T2=T₂)

    return seqd, obj
end

# load RF train used in in-vivo experiment

# invivo = MAT.matread("figures/invivo/ovdh_RF15_Slice11_Maxit15_kSpaces5/MRSTAT_Input.mat");
# const invivo_RF_train = complex.(invivo["Runtime"]["RF_angles_ex"]);
# const invivo_py = invivo["Runtime"]["py_factor"] .|> Int


# function to assemble BlochSimulators.pSSFP sequence simulator
function assemble_pssfp2d(nTR, nRF, nz)

    γ = 26753.0;
    # RF_train = invivo_RF_train[1:nTR]; # flip angles in degrees (and complex for phase)
    RF_train = LinRange(1,90,nTR) |> collect; # flip angles in degrees (and complex for phase)
    @. RF_train[2:2:end] *= -1 # 0-π phase cycling
    RF_train = complex.(RF_train)

    TR = 0.008; # s

    if nz == 1
        z = SVector(0.0)
    else
        z = SVector( vec(collect(LinRange(-1.0,1.0,nz)))... );
    end

    # pssfp specific parameters
    RFexdur = 0.001;
    Δt = (ex=0.001/nRF, inv = 1e8, pr = (TR - RFexdur)/2); # effectively no inversion prepusle
    γΔtGRz = (ex=0.002/nRF, inv = 0.00, pr = -0.01);
    γΔtRF = (pi/180) * (1/nRF) * SVector(ones(nRF)...); # RF waveform normalized to flip angle of 1 degree


    return BlochSimulators.pSSFP2D(RF_train,TR, γΔtRF, Δt, γΔtGRz, z);
end

# function to assemble BlochSimulators.pSSFP sequence simulator
function assemble_pssfp3d(nTR, nRF)

    γ = 26753.0;
    # RF_train = invivo_RF_train[1:nTR]; # flip angles in degrees (and complex for phase)
    RF_train = LinRange(1,90,nTR) |> collect; # flip angles in degrees (and complex for phase)
    @. RF_train[2:2:end] *= -1 # 0-π phase cycling
    RF_train = complex.(RF_train)

    TR = 0.008; # s

    # pssfp specific parameters
    RFexdur = 0.001;
    Δt = (ex=0.001/nRF, inv = 1e8, pr = (TR - RFexdur)/2); # effectively no inversion prepusle
    γΔtRF = (pi/180) * (1/nRF) * SVector(ones(nRF)...); # RF waveform normalized to flip angle of 1 degree

    return BlochSimulators.pSSFP3D(RF_train, TR, γΔtRF, Δt);
end

function pssfp3d_to_generic3d(pssfp::BlochSimulators.pSSFP3D)

    γ = 26753.0

    RF = ComplexF64[]
    Δt = Float64[]
    ADC = Bool[]

    # excitation
    RF_excitation  = Array(pssfp.γΔtRF ./ (γ * pssfp.Δt.ex))
    Δt_excitation = fill(pssfp.Δt.ex, length(RF_excitation))
    ADC_excitation = zeros(Bool, length(RF_excitation))

    # free precession
    RF_precession = zeros(2)
    Δt_precession = fill(pssfp.Δt.pr, 2)
    ADC_precession = [true; false]

    # START ASSEMBLING SEQUENCE

    # inversion done using initial guess m = [0,0,-1]
    append!(Δt, pssfp.Δt.inv)
    append!(RF, 0.0 + 0.0im)
    append!(ADC, false)

    # apply "alpha over two" pulse
    α₀ = -pssfp.RF_train[1]/2
    append!(RF, α₀*RF_excitation)
    append!(Δt, Δt_excitation)
    append!(ADC, ADC_excitation)

    # Free precession part: slice select pre and rephaser and precession until RF
    append!(RF, RF_precession[1])
    append!(Δt, Δt_precession[1])
    append!(ADC, false)

    # ADC = zeros(Bool, length(RF))

    for r in 1:length(pssfp.RF_train)

        # RF excitation
        α = pssfp.RF_train[r]
        append!(RF, α*RF_excitation)
        append!(Δt, Δt_excitation)
        append!(ADC, ADC_excitation)
        # Remainder of the TR for which RF is off
        append!(RF, RF_precession)
        append!(Δt, Δt_precession)
        append!(ADC, ADC_precession)

    end

    GRx = zero(Δt)
    GRy = zero(Δt)
    GRz = zero(Δt)
    
    GR = permutedims([GRx ;; GRy ;; GRz])
    samples = BitVector(ADC)

    return BlochSimulators.Generic3D(RF, GR, samples, Δt)
end

# function to assemble BlochSimulators.FISP sequence simulator
function assemble_fisp2d(nTR, nz, max_state)

    # RF_train = invivo_RF_train[1:nTR]; # flip angles in degrees (and complex for phase)
    RF_train = LinRange(1,90,nTR) |> collect; # flip angles in degrees (and complex for phase)
    TR = 0.008; # s
    TE = TR/2; # s
    TI = 0.01; # s

    # matrix with slice profile correction factors
    spc = complex.(ones(nTR, nz))

    return BlochSimulators.FISP2D(RF_train, spc, TR, TE, Val(max_state), TI)
end