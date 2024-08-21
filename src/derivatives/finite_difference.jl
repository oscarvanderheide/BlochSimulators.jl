# Default step sizes for the different types of tissue parameters
const DEFAULT_ΔT₁ = 0.02
const DEFAULT_ΔT₂ = 0.0025
const DEFAULT_ΔB₁ = 0.001
const DEFAULT_ΔB₀ = 0.01
const DEFAULT_STEPSIZES_FINITE_DIFFERENCE = T₁T₂B₁B₀(DEFAULT_ΔT₁, DEFAULT_ΔT₂, DEFAULT_ΔB₁, DEFAULT_ΔB₀)
const Symbols = Tuple{Vararg{Symbol}}

"""
    simulate_derivatives_finite_difference(
        requested_derivatives::Tuple{Vararg{Symbol}},
        echos::AbstractMatrix{<:Complex},
        sequence::BlochSimulator,
        parameters::StructVector{<:AbstractTissueProperties},
        stepsizes::T₁T₂B₁B₀=DEFAULT_STEPSIZES
    )

Calculates the partial derivatives of pre-calculated echos (contained in `ctx`) with respect to the (non-linear) tissue properties in `requested_derivatives` using finite differences.

# Arguments
- `requested_derivatives::Tuple{Vararg{Symbol}}`: A tuple with symbols corresponding to the (non-linear) tissue properties we want to compute partial derivatives for (e.g. (:T₁, :T₂, :B₁)).
- `echos::AbstractMatrix`: Pre-computed matrix of magnetization at all echo times for all voxels.
- `sequence::BlochSimulator`: The simulator representing the underlying MR pulse sequence.
- `parameters::SimulationParameters`: The simulation parameters (in all voxels) used to generate `echos`.
- `stepsizes::T₁T₂B₁B₀`: The step sizes for finite difference calculations for each of the different tissue properties. Default stepsizes are provided in `DEFAULT_STEPSIZES_FINITE_DIFFERENCE`.

# Returns
- `∂echos`: A NamedTuple containing the partial derivatives of `ctx.echos` with respect to each of the `requested_derivatives`. That is, ∂echos.T₁ contains the partial derivatives w.r.t. T₁, ∂echos.T₂ contains the partial derivatives w.r.t. T₂, etc.

# Notes
- We only calculate partial derivatives for non-linear tissue properties. For linear tissue properties (e.g. proton density) we need nothing more than `echos` itself.
"""
function simulate_derivatives_finite_difference(
    requested_derivatives::Symbols,
    echos::AbstractMatrix{<:Complex},
    sequence::BlochSimulator,
    parameters::StructVector{<:AbstractTissueProperties},
    stepsizes::T₁T₂B₁B₀=DEFAULT_STEPSIZES_FINITE_DIFFERENCE
)

    # Check consistency of "context"
    _validate_requested_derivatives(requested_derivatives, parameters, stepsizes)

    # Helper function to calculate the derivatives of echos w.r.t. a single tissue property
    fd = derivative -> finite_difference_single_tissue_property(derivative, echos, sequence, parameters, stepsizes)

    # Calculate partial derivative of echos w.r.t. each requested derivative
    ∂echos = map(fd, requested_derivatives)

    # Convert to NamedTuple s.t. ∂echos.T₁, ∂echos.T₂, etc. is possible
    ∂echos = NamedTuple{requested_derivatives}(∂echos)

    return ∂echos
end

"""
    finite_difference_single_property(derivative::Symbol, echos, sequence, parameters, stepsizes)

Approximate partial derivatives of `echos` (a matrix of size (# readouts, # voxels) contained in `ctx`) w.r.t. the single tissue property `derivative` (e.g. :T₁) using finite differences.
"""
function finite_difference_single_tissue_property(derivative::Symbol, echos, sequence, parameters, stepsizes)

    # Calculate step size for current parameter type
    Δ = _calculate_stepsize(derivative, real(eltype(echos)), stepsizes)

    # Calculate modified parameters in all voxels
    parameters_modified = _calculate_modified_parameters(derivative, parameters, Δ)

    # Simulate magnetization with modified parameters
    echos_modified = simulate_magnetization(sequence, parameters_modified)

    # Approximate partial derivatives of `echos` w.r.t. `derivative`
    _finite_difference_quotient!(echos_modified, echos, Δ)

    # `echos_modified` was modified in place, rename it to ∂echos
    ∂echos = echos_modified

    return ∂echos
end

"""
    simulate_derivatives_finite_difference(sequence, parameters)

If only a `sequence` and `parameters` are provided, calculate the `echos` and then calculate the partial derivatives of `echos` w.r.t. the non-linear tissue properties using finite differences with the default stepsizes. Returns both the `echos` and the partial derivatives `∂echos`.
"""
function simulate_derivatives_finite_difference(
    sequence::BlochSimulator,
    parameters::StructVector{<:AbstractTissueProperties},
    stepsizes::T₁T₂B₁B₀=DEFAULT_STEPSIZES_FINITE_DIFFERENCE
)
    # Calculate echos
    echos = simulate_magnetization(sequence, parameters)

    # Extract names of the non-linear tissue properties
    nonlinear_propnames = fieldnames(get_nonlinear_part(eltype(parameters)))

    # Calculate partial derivatives of echos w.r.t. all tissue properties
    ∂echos = simulate_derivatives_finite_difference(nonlinear_propnames, echos, sequence, parameters, stepsizes)

    return echos, ∂echos
end

"""
If the tissue properties for a single voxel are provided only, the simulation is performed on the CPU in a single-threaded fashion.
"""
function simulate_derivatives_finite_difference(
    sequence::BlochSimulator,
    tissue_properties::AbstractTissueProperties,
    stepsizes::T₁T₂B₁B₀=DEFAULT_STEPSIZES_FINITE_DIFFERENCE
)
    # Assemble "SimulationParameters" 
    parameters = StructVector([tissue_properties])

    return simulate_derivatives_finite_difference(sequence, parameters, stepsizes)
end

"""
    _validate_requested_derivatives, parameters, stepsizes)

Check that the tissue properties we want to compute partial derivatives for are a subset of the tissue properties used in the simulations. Also check that a stepsize is available for each requested derivative.
"""
function _validate_requested_derivatives(requested_derivatives::Symbols, parameters, stepsizes)

    # Extract the names of the different tissue properties used in forward simulations
    simulation_property_names = propertynames(parameters)

    # Check that the requested derivatives are a subset of the simulation properties
    if !(requested_derivatives ⊆ simulation_property_names)
        error("The parameters ($requested_derivatives) for which finite difference calculations are requested are not a subset of the simulation parameters ($simulation_property_names)")
    end

    # Check that a step size is available for each requested derivative
    if !(requested_derivatives ⊆ propertynames(stepsizes))
        error("The parameters for which finite difference calculations are requested ($(requested_derivatives)) are not a subset of the parameters for which step sizes are provided ($(propertynames(stepsizes))")
    end

    return nothing
end

"""
    _calculate_stepsize(derivative::Symbol, T::Type{<:Real})

Calculate the step size Δ for finite difference approximation of derivatives.

# Arguments
- `derivative::Symbol`: The type of derivative to calculate the step size for. Valid options are `:T₁`, `:T₂`, `:B₁`, and `:B₀`.
- `T::Type{<:Real}`: The type of the step size, which should be a subtype of `Real`.
- `stepsizes::T₁T₂B₁B₀`: A struct from BlochSimulators containing the step sizes for all potential tissue properties.

# Returns
- `Δ`: The calculated step size.
"""
function _calculate_stepsize(
    derivative::Symbol,
    T::Type{<:Real},
    stepsizes::T₁T₂B₁B₀
)
    # Check that derivative is a valid parameter
    if derivative ∉ propertynames(stepsizes)
        error("Parameter $(derivative) not (yet) implemented")
    end

    # Get stepsize for the parameter we want to approximate the partial derivatives for
    Δ = getproperty(stepsizes, derivative)

    # Convert to desired precision
    Δ = convert(T, Δ)

    return Δ
end

"""
    _calculate_modified_parameters(derivative, parameters, Δ)

Calculate a copy of the simulation `parameters` and for each element, add `Δ` to its property corresponding to `derivative`.

That is, if we want to calculate `f(x+h)-f(x)/h`, we are calculating the `x+h` part here.

# Arguments
- `derivative::Symbol`: The derivative symbol.
- `parameters::StructVector{<:AbstractTissueProperties}`: The original parameters.
- `Δ::Real`: The step size.

# Returns
- `modified_parameters::StructVector{<:AbstractTissueProperties}`: The modified parameters (e.g. `x+h`).
"""
function _calculate_modified_parameters(
    derivative::Symbol,
    parameters::StructVector{<:AbstractTissueProperties},
    Δ::Real
)
    modified_parameters = copy(parameters)

    getproperty(modified_parameters, derivative) .+= Δ

    return modified_parameters
end

"""
    _finite_difference_quotient!(Δm, m, Δ)

Calculates the finite difference quotient (Δm - m) / Δ in-place in Δm. Note that an out-of-place version of this function is also available.
"""
function _finite_difference_quotient!(Δm::AbstractArray{T}, m::AbstractArray{T}, Δ::Real) where {T}

    if size(Δm) != size(m)
        error("Size of Δm and m do not match")
    end

    if iszero(Δ)
        error("Step size Δ is zero")
    end

    # Calculate finite difference quotient in-place
    Δm .= (Δm .- m) ./ Δ

    return nothing
end

"""
    _finite_difference_quotient(Δm, m, Δ)

Calculates the finite difference quotient (Δm - m) / Δ. Note that an in-place version of this function is also available.
"""
function _finite_difference_quotient(Δm::AbstractArray{T}, m::AbstractArray{T}, Δ::Real) where {T}

    if size(Δm) != size(m)
        error("Size of Δm and m do not match")
    end

    if iszero(Δ)
        error("Step size Δ is zero")
    end

    # Calculate finite difference quotient out-of-place
    return (Δm .- m) ./ Δ
end

