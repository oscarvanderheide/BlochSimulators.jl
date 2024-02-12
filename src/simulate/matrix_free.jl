function Mv_matrixfree(::CPU1, sequence::pSSFP2D, trajectory::CartesianTrajectory, parameters, coil_sensitivities, inputvec)

    nvoxels = length(parameters)

    signal = zeros(eltype(coil_sensitivities), nsamplesperreadout(trajectory,1), nreadouts(trajectory))
    echos = zeros(ComplexF64, nreadouts(trajectory))

    m = Isochromat{Float64}(0,0,0)

    for v in 1:nvoxels

        echos .= 0

        p = parameters[v]

        T₁,T₂ = p.T₁, p.T₂

        γΔtRFᵉˣ     = sequence.γΔtRF
        γΔtGRzᵉˣ    = sequence.γΔtGRz.ex
        Δtᵉˣ        = sequence.Δt.ex
        E₁ᵉˣ, E₂ᵉˣ  = E₁(m, Δtᵉˣ, T₁), E₂(m, Δtᵉˣ, T₂)

        γΔtGRzᵖʳ    = sequence.γΔtGRz.pr
        Δtᵖʳ        = sequence.Δt.pr
        E₁ᵖʳ, E₂ᵖʳ  = E₁(m, Δtᵖʳ, T₁), E₂(m, Δtᵖʳ, T₂)

        E₁ⁱⁿᵛ, E₂ⁱⁿᵛ  = E₁(m, sequence.Δt.inv, T₁), E₂(m, sequence.Δt.inv, T₂)

        # Simulate excitation with flip angle θ using hard pulse approximation of the normalized RF-waveform γΔtRF
        excite = @inline function(m,θ,z)
            for ⚡ in (θ * γΔtRFᵉˣ)
                m = rotate(m, ⚡, γΔtGRzᵉˣ, z, Δtᵉˣ, p)
                m = decay(m, E₁ᵉˣ, E₂ᵉˣ)
                m = regrowth(m, E₁ᵉˣ)
            end
            return m
        end

        # Slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth
        precess = @inline function(m,z)
            m = rotate(m, γΔtGRzᵖʳ, z, Δtᵖʳ, p)
            m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
            m = regrowth(m, E₁ᵖʳ)
            return m
        end

        @inbounds for z in sequence.z

            # reset spin to initial conditions
            m = initial_conditions(m)

            # apply inversion pulse
            m = invert(m, p)
            m = decay(m, E₁ⁱⁿᵛ, E₂ⁱⁿᵛ)
            m = regrowth(m, E₁ⁱⁿᵛ)

            # apply "alpha over two" pulse
            θ₀ = -sequence.RF_train[1]/2
            m = excite(m, θ₀, z)

            # slice select re- & prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
            m = rotate(m, 2*γΔtGRzᵖʳ, z, Δtᵖʳ, p)
            m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
            m = regrowth(m, E₁ᵖʳ)

            # simulate pSSFP2D sequence with varying flipangles
            for (TR,θ) ∈ enumerate(sequence.RF_train)
                # simulate RF pulse and slice-selection gradient
                m = excite(m, θ, z)
                # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until TE
                m = precess(m,z)
                # sample magnetization at echo time (sum over slice direction)
                echos[TR] += complex(m.x,m.y)
                # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
                m = precess(m,z)
            end
        end

        # expand echos to full readouts and accumulate

        # Read in constants
        R₂ = inv(p.T₂)
        ns = nsamplesperreadout(trajectory, 1)
        Δt = trajectory.Δt
        Δkₓ = trajectory.Δk_adc
        x = p.x
        y = p.y
        c = coil_sensitivities[v]
        z = inputvec[v]
        cz = c*z

        for r in 1:nreadouts(trajectory)

            # load magnetization at echo time 
            mₑ = echos[r]
            # apply phase encoding
            kʸ = imag(trajectory.k_start_readout[r]) 
            mₑ *= exp(im * (kʸ * y'))
            # Apply readout gradient, T₂ decay and B₀ rotation
            E₂ = exp(-Δt*R₂)
            θ = Δkₓ * x
            hasB₀(p) && (θ += π*p.B₀*Δt*2)
            E₂eⁱᶿ = E₂ * exp(im*θ)
            mₛ = E₂eⁱᶿ^(-(ns÷2)) * mₑ

            for s = 1:ns
                signal[s,r] += mₛ * cz
                mₛ *= E₂eⁱᶿ
            end
        end

    end

    return reshape(signal,nsamples(trajectory),1)
end

function Mᴴv_matrixfree(::CPU1, sequence::pSSFP2D, trajectory::CartesianTrajectory, parameters, coil_sensitivities, inputvec)

    nvoxels = length(parameters)

    outputvec = zeros(ComplexF64, nvoxels)
    echos = zeros(ComplexF64, nreadouts(trajectory))


    inputvec = reshape(inputvec, nsamplesperreadout(trajectory,1), nreadouts(trajectory))

    m = Isochromat{Float64}(0,0,0)

    for v in 1:nvoxels

        echos .= 0

        p = parameters[v]

        T₁,T₂ = p.T₁, p.T₂

        γΔtRFᵉˣ     = sequence.γΔtRF
        γΔtGRzᵉˣ    = sequence.γΔtGRz.ex
        Δtᵉˣ        = sequence.Δt.ex
        E₁ᵉˣ, E₂ᵉˣ  = E₁(m, Δtᵉˣ, T₁), E₂(m, Δtᵉˣ, T₂)

        γΔtGRzᵖʳ    = sequence.γΔtGRz.pr
        Δtᵖʳ        = sequence.Δt.pr
        E₁ᵖʳ, E₂ᵖʳ  = E₁(m, Δtᵖʳ, T₁), E₂(m, Δtᵖʳ, T₂)

        E₁ⁱⁿᵛ, E₂ⁱⁿᵛ  = E₁(m, sequence.Δt.inv, T₁), E₂(m, sequence.Δt.inv, T₂)

        # Simulate excitation with flip angle θ using hard pulse approximation of the normalized RF-waveform γΔtRF
        excite = @inline function(m,θ,z)
            for ⚡ in (θ * γΔtRFᵉˣ)
                m = rotate(m, ⚡, γΔtGRzᵉˣ, z, Δtᵉˣ, p)
                m = decay(m, E₁ᵉˣ, E₂ᵉˣ)
                m = regrowth(m, E₁ᵉˣ)
            end
            return m
        end

        # Slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth
        precess = @inline function(m,z)
            m = rotate(m, γΔtGRzᵖʳ, z, Δtᵖʳ, p)
            m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
            m = regrowth(m, E₁ᵖʳ)
            return m
        end

        @inbounds for z in sequence.z

            # reset spin to initial conditions
            m = initial_conditions(m)

            # apply inversion pulse
            m = invert(m, p)
            m = decay(m, E₁ⁱⁿᵛ, E₂ⁱⁿᵛ)
            m = regrowth(m, E₁ⁱⁿᵛ)

            # apply "alpha over two" pulse
            θ₀ = -sequence.RF_train[1]/2
            m = excite(m, θ₀, z)

            # slice select re- & prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
            m = rotate(m, 2*γΔtGRzᵖʳ, z, Δtᵖʳ, p)
            m = decay(m, E₁ᵖʳ, E₂ᵖʳ)
            m = regrowth(m, E₁ᵖʳ)

            # simulate pSSFP2D sequence with varying flipangles
            for (TR,θ) ∈ enumerate(sequence.RF_train)
                # simulate RF pulse and slice-selection gradient
                m = excite(m, θ, z)
                # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until TE
                m = precess(m,z)
                # sample magnetization at echo time (sum over slice direction)
                echos[TR] += complex(m.x,m.y)
                # slice select prephaser, B₀ rotation, T₂ decay and T₁ regrowth until next RF
                m = precess(m,z)
            end
        end

        # expand echos to full readouts and accumulate

        # Read in constants
        R₂ = inv(p.T₂)
        ns = nsamplesperreadout(trajectory, 1)
        Δt = trajectory.Δt
        Δkₓ = trajectory.Δk_adc
        x = p.x
        y = p.y
        c = coil_sensitivities[v]

        tmp = zero(c)

        for r in 1:nreadouts(trajectory)

            # load magnetization at echo time 
            mₑ = echos[r]
            # apply phase encoding
            kʸ = imag(trajectory.k_start_readout[r]) 
            mₑ *= exp(im * (kʸ * y'))
            # Apply readout gradient, T₂ decay and B₀ rotation
            E₂ = exp(-Δt*R₂)
            θ = Δkₓ * x
            hasB₀(p) && (θ += π*p.B₀*Δt*2)
            E₂eⁱᶿ = E₂ * exp(im*θ)
            mₛ = E₂eⁱᶿ^(-(ns÷2)) * mₑ

            for s = 1:ns
                tmp += conj(mₛ) * inputvec[s,r]
                mₛ *= E₂eⁱᶿ
            end
        end

        outputvec[v] = dot(c,tmp)

    end

    return outputvec
end

function Jv( sequence::Array{Float64,2},
    parameters::Array{Float64,2},
    coordinates::Array{Float64,2},
    coilmaps::Array{Complex{Float64},2},
    subslices::Array{Float64,1},
    spoilerindex::Integer,
    B1_scaling::Float64,
    B0_scaling::Float64,
    v::Array{Float64,1})

    Nc = size(coilmaps,1)           # nr of coils
    Nt = size(sequence, 2)          # nr of time intervals
    Nv = size(coordinates, 2)       # nr of voxels
    Nz = length(subslices)          # nr of spins in slice direction
    Ns = Int64(sum(sequence[1,:]))  # nr of acquired samples
    Np = size(parameters,1)
    @assert length(v) == Np * Nv

    m0     = Vec3(0.0, 0.0, 1.0) # initial condition for all spins
    ∂m0∂T₁ = Vec3(zeros(3)...) # partial derivatives of initial condition
    ∂m0∂T₂ = Vec3(zeros(3)...)
    ∂m0∂B₁ = Vec3(zeros(3)...)
    ∂m0∂B₀ = Vec3(zeros(3)...)

    ∂Mₚ = zeros(Float64, 4, Ns)

    Jʳᵉv = zeros(Float64, Ns, Nc)
    Jⁱᵐv = zeros(Float64, Ns, Nc)

    idxᵃᵈᶜ = findfirst(x->(x==true), sequence[1,:])
    γΔtGRxᵃᵈᶜ = sequence[2,idxᵃᵈᶜ]
    Δtᵃᵈᶜ = sequence[7,idxᵃᵈᶜ]

    #Split coils for real and imaginary parts
    Cʳᵉ = real(coilmaps)
    Cⁱᵐ = imag(coilmaps)

    @inbounds for p = 1:Nv # Loop over in-plane positions

        # Load parameters for spatial position
        T₁ = parameters[1,p]
        T₂ = parameters[2,p]
        R₁ = 1.0 / T₁
        R₂ = 1.0 / T₂
        B₁ = parameters[3,p]
        B₀ = parameters[4,p]
        ρʳᵉ = parameters[5,p]
        ρⁱᵐ = parameters[6,p]

        # Load x and y coordinates for spatial position
        x = coordinates[1,p]
        y = coordinates[2,p]

        # Load elements of v relevant to this spatial position
        vᵀ¹  = v[p + 0 * Nv] * T₁
        vᵀ²  = v[p + 1 * Nv] * T₂
        vᴮ¹  = v[p + 2 * Nv] * B1_scaling
        vᴮ⁰  = v[p + 3 * Nv] * B0_scaling
        vᵖʳᵉ = v[p + 4 * Nv]
        vᵖⁱᵐ = v[p + 5 * Nv]

        # Precompute values that are always the same during readouts

        # Normal part
        sinθᵃᵈᶜ, cosθᵃᵈᶜ = sincos((γΔtGRxᵃᵈᶜ * x) + (B₀ * Δtᵃᵈᶜ))
        E1ᵃᵈᶜ = exp(-Δtᵃᵈᶜ * R₁)
        E2ᵃᵈᶜ = exp(-Δtᵃᵈᶜ * R₂)
        sinθE2ᵃᵈᶜ = sinθᵃᵈᶜ * E2ᵃᵈᶜ
        cosθE2ᵃᵈᶜ = cosθᵃᵈᶜ * E2ᵃᵈᶜ

        # Automatic differentiation part
        ΔtR₁R₁ᵃᵈᶜ = Δtᵃᵈᶜ * R₁ * R₁;
        ΔtR₂R₂ᵃᵈᶜ = Δtᵃᵈᶜ * R₂ * R₂;

        @simd for s = 1:Nz # Loop over slice direction

            z = subslices[s] # Load z coordinate of current spin
            r = Vec3(x,y,z) # Store spatial coordinates of spin in a Vec3
            m   = m0 # Set initial condition for current spin
            ∂m∂T₁ = ∂m0∂T₁ # Set initial partial derivatives (all zeros)
            ∂m∂T₂ = ∂m0∂T₂
            ∂m∂B₁ = ∂m0∂B₁
            ∂m∂B₀ = ∂m0∂B₀

            idx = 1 # Reset sample counter

            for t = 1:Nt # Start time integration

                ADC = sequence[1,t]
                if ADC == 1.0
                    # Jv = Σ vᵢ * ∂m∂αᵢ
                    # We basically compute Np columns per voxel,
                    # scale them with the relevant vᵢ's and add them up
                    # Later on we scale with the coil sensitivity maps
                    tmpx1 =  vᵀ¹ * (ρʳᵉ*∂m∂T₁.x - ρⁱᵐ*∂m∂T₁.y)
                    tmpx1 += vᵀ² * (ρʳᵉ*∂m∂T₂.x - ρⁱᵐ*∂m∂T₂.y)
                    tmpx1 += vᴮ¹ * (ρʳᵉ*∂m∂B₁.x - ρⁱᵐ*∂m∂B₁.y)
                    tmpx1 += vᴮ⁰ * (ρʳᵉ*∂m∂B₀.x - ρⁱᵐ*∂m∂B₀.y)
                    tmpx1 +=  vᵖʳᵉ * m.x
                    tmpx1 += -vᵖⁱᵐ * m.y

                    tmpx2  = vᵀ¹ * (-ρⁱᵐ*∂m∂T₁.x - ρʳᵉ*∂m∂T₁.y)
                    tmpx2 += vᵀ² * (-ρⁱᵐ*∂m∂T₂.x - ρʳᵉ*∂m∂T₂.y)
                    tmpx2 += vᴮ¹ * (-ρⁱᵐ*∂m∂B₁.x - ρʳᵉ*∂m∂B₁.y)
                    tmpx2 += vᴮ⁰ * (-ρⁱᵐ*∂m∂B₀.x - ρʳᵉ*∂m∂B₀.y)
                    tmpx2 += -vᵖʳᵉ * m.y
                    tmpx2 += -vᵖⁱᵐ * m.x

                    tmpy1  = vᵀ¹ * (ρʳᵉ*∂m∂T₁.y + ρⁱᵐ*∂m∂T₁.x)
                    tmpy1 += vᵀ² * (ρʳᵉ*∂m∂T₂.y + ρⁱᵐ*∂m∂T₂.x)
                    tmpy1 += vᴮ¹ * (ρʳᵉ*∂m∂B₁.y + ρⁱᵐ*∂m∂B₁.x)
                    tmpy1 += vᴮ⁰ * (ρʳᵉ*∂m∂B₀.y + ρⁱᵐ*∂m∂B₀.x)
                    tmpy1 += vᵖʳᵉ * m.y
                    tmpy1 += vᵖⁱᵐ * m.x

                    tmpy2  = vᵀ¹ * (ρʳᵉ*∂m∂T₁.x - ρⁱᵐ*∂m∂T₁.y)
                    tmpy2 += vᵀ² * (ρʳᵉ*∂m∂T₂.x - ρⁱᵐ*∂m∂T₂.y)
                    tmpy2 += vᴮ¹ * (ρʳᵉ*∂m∂B₁.x - ρⁱᵐ*∂m∂B₁.y)
                    tmpy2 += vᴮ⁰ * (ρʳᵉ*∂m∂B₀.x - ρⁱᵐ*∂m∂B₀.y)
                    tmpy2 += vᵖʳᵉ * m.x
                    tmpy2 += -vᵖⁱᵐ * m.y

                    ∂Mₚ[1,idx] += tmpx1
                    ∂Mₚ[2,idx] += tmpx2
                    ∂Mₚ[3,idx] += tmpy1
                    ∂Mₚ[4,idx] += tmpy2

                    idx += 1
                    # When there is no RF, rotation and decay can be performed simultaneously
                    # (and no need to load sequence parameters)
                    m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀ = rotatedecay(m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀,  Δtᵃᵈᶜ, sinθE2ᵃᵈᶜ, cosθE2ᵃᵈᶜ, E1ᵃᵈᶜ, ΔtR₁R₁ᵃᵈᶜ, ΔtR₂R₂ᵃᵈᶜ)
                else
                    # Load sequence for current time interval
                    γΔtGR  = Vec3(sequence[2,t], sequence[3,t], sequence[4,t])
                    γΔtRF  = complex(sequence[5,t], sequence[6,t])
                    Δt     = sequence[7,t]

                    m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀ = rotate(m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀, γΔtRF, γΔtGR, Δt, B₁, B₀, r)
                    m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀ = decay(m, ∂m∂T₁, ∂m∂T₂, ∂m∂B₁, ∂m∂B₀, R₁, R₂, Δt)

                    # Spoil transverse magnetization at the end of intervals
                    # It is assumed that spoiler never happens when ADC is on
                    if t == spoilerindex
                        m     = Vec3(0.0, 0.0, m.z)
                        ∂m∂T₁ = Vec3(0.0, 0.0, ∂m∂T₁.z)
                        ∂m∂T₂ = Vec3(0.0, 0.0, ∂m∂T₂.z)
                        ∂m∂B₁ = Vec3(0.0, 0.0, ∂m∂B₁.z)
                        ∂m∂B₀ = Vec3(0.0, 0.0, ∂m∂B₀.z)
                    end
                end

            end # End loop over time
        end # End loop in slice direction

        # Optimized sum & scaling with coil sensitivities
        @inbounds @fastmath for c = 1:Nc
            cʳᵉ = Cʳᵉ[c,p]
            cⁱᵐ = Cⁱᵐ[c,p]
            @fastmath for t = 1:Ns
                Jʳᵉv[t,c] -= (cʳᵉ*∂Mₚ[1,t] + cⁱᵐ*∂Mₚ[2,t])
                Jⁱᵐv[t,c] -= (cʳᵉ*∂Mₚ[3,t] + cⁱᵐ*∂Mₚ[4,t])
            end
        end
        @. ∂Mₚ .= 0.0

    end # End loop over in-plane positions

    return reshape([vec(Jʳᵉv); vec(Jⁱᵐv)],:,1) # Allocates some memory but no problem ... for now and we need a 2D array for Darray
end




