# Forward sensitivity propagation for EPG-based sequences
# -----------------------------------------------------------
# Purpose: compute exact derivatives of a (discrete, truncated) EPG
# simulation w.r.t. T₁ and T₂ (and, when the tissue properties carry a B₁,
# w.r.t. B₁) by propagating tangent states ∂Ω/∂θ alongside the ordinary EPG
# state Ω, instead of using finite differences.
#
# Nothing here is specific to FISP3D: everything is written in terms of the
# operators in `src/operators/epg.jl` (`excite!`, `decay!`, `rotate_decay!`,
# `regrowth!`, `invert!`, `spoil!`, `dephasing!`, `diffuse!`, ...), which any
# `EPGSimulator{T,Ns}` sequence is built from -- FISP2D uses exactly the same
# operator set as FISP3D. So the operator overloads below work for any such
# sequence, and the public `simulate_derivatives_forward_sensitivity` API is
# written against `EPGSimulator`, not `FISP3D`.
#
# How it works
# ------------
# The simulation is handed, instead of a single configuration state matrix Ω,
# the pair
#
#   (Ω, ∂Ω)      with   ∂Ω = (∂Ω/∂T₁, ∂Ω/∂T₂)  or  (∂Ω/∂T₁, ∂Ω/∂T₂, ∂Ω/∂B₁)
#
# where every entry is an *ordinary* configuration state matrix, exactly like
# the one an ordinary simulation uses. Each EPG operator then gets a method
# for this pair, which applies the *existing, unmodified* operator to Ω and
# to each tangent, plus a source term wherever the operator itself depends on
# a parameter. So this file contains the calculus and `epg.jl` keeps the
# physics.
#
# A sequence's own `simulate_magnetization!(magnetization, sequence, Ω, p)`
# method -- the exact same method used for ordinary (non-derivative)
# simulation -- is then reused completely verbatim, called with this pair as
# `Ω` and a matching pair of output arrays as `magnetization`.
#
# Only three operators have a source term, because only three things in the
# simulation depend on the tissue properties at all:
#
#   decay! / rotate_decay!   E₁ and E₂ depend on T₁ and T₂
#   regrowth!                E₁ depends on T₁
#   excite! / invert!        the rotation angle depends on B₁
#
# Every other operator (dephasing!, spoil!, diffuse!, ...) is the same linear
# map on Ω and on each tangent, so its method just applies the original
# operator to each of them in turn.
#
# There are two tangent sets, `∂Ω∂T₁T₂` and `∂Ω∂T₁T₂B₁`, and every operator
# below is a single method that takes either. Whatever the B₁ tangent needs is
# guarded by `has_B₁(∂Ω)`, which is a compile-time constant: each set gets its
# own specialized code with the guard resolved and the other set's line gone,
# so the guard costs nothing (measured: same registers, same runtime as
# writing the two out separately). Adding a parameter means adding a struct,
# a `has_B₁`-style trait, and one guarded line per operator.
#
# Why one matrix per tangent, and not one matrix of (value, ∂T₁, ∂T₂, ...)
# entries: the latter widens the state's *element type*, which is fine for
# the elementwise operators but not for `excite!`'s `mul!` (a fully-unrolled
# StaticArrays matrix multiply) or GPU `dephasing!`'s warp shuffle -- both
# hold many intermediates live at once, so at N times the width they spill to
# local memory. Keeping separate native-width matrices means every individual
# StaticArrays/`shfl_sync` call is exactly as wide as in an ordinary
# simulation.
#
# Limitations
# -----------
# Because there are exactly two tangent structs, exactly two combinations
# exist: T₁/T₂, and T₁/T₂/B₁. So:
#
#   * T₁ and T₂ always travel together. Requesting only `:T₁` still
#     propagates both and returns one of them, so it costs the same as
#     asking for both. B₁ is genuinely optional -- it is propagated only
#     when asked for, and costs one tangent state plus a second matrix
#     multiply in `excite!`.
#
#   * B₀ and D cannot be differentiated at all, though neither is hard.
#     Both enter multiplicatively, which is the case the product rule below
#     already covers. B₀ needs `off_resonance_rotation` to return
#     `(eⁱᶿ, ∂eⁱᶿ∂B₀)` plus a source term in `rotate_decay!`'s two
#     transverse rows. D differs in shape rather than in principle:
#     `diffuse!` scales elementwise by a whole matrix, so its `∂f` is a
#     matrix rather than a 3-tuple of per-row factors.
#
# Adding one more parameter costs a struct, a `has_*` trait, and one guarded
# line per operator. Adding several is where this stops scaling, because the
# combinations multiply.
#
# The general version, and why it is not the one here
# ---------------------------------------------------
# A version generic over an arbitrary set of parameters was written and
# measured before this one. It produced *identical* GPU code -- same
# registers, same runtime -- and was replaced because it was much harder to
# read, not because it cost anything. If the combinations ever do multiply,
# that is the way out. This is the shape it took, and the traps it hit, each
# of which was paid for once already:
#
#   * Hold the tangents in a `NamedTuple` keyed by parameter name, wrapped
#     in a struct so that giving the tuple its own methods is not type
#     piracy: `struct ∂Ω∂θ{names,T<:Tuple}; ∂::NamedTuple{names,T}; end`,
#     with `getproperty` forwarding so `∂Ω.∂T₁` still reads the same. Putting
#     `names` in the type is what makes each requested set compile its own
#     kernel and pay only for the tangents it asked for.
#
#   * "Do this to every tangent" has to become a `@generated` helper that
#     emits one explicit call per parameter. It cannot be a loop or a
#     `foreach`: collecting the (mutable) GPU state wrappers into a tuple
#     and passing them through a closure defeats the GPU compiler's escape
#     analysis, and every state is then heap-allocated. Measured symptom:
#     48-byte device allocations and 216-664 B/thread of local memory
#     instead of 64, with the device heap exhausting at 20k voxels.
#
#   * Those `@generated` helpers must emit `Expr(:meta, :inline)`. Without
#     it the states created inside them escape in exactly the same way, with
#     exactly the same symptom. This is the easiest thing here to get wrong,
#     because it looks like a formatting detail.
#
#   * A `@generated` body may not contain a closure (`all(j -> ...)` and the
#     like); Julia rejects it as "not pure". Plain loops with `push!` are
#     fine, and so are comprehensions.
#
#   * The driver must choose the parameter set with literal `Val`s in an
#     if/else, not derive it from the runtime `requested_derivatives`.
#     Deriving it at runtime made inference give up with a compiler internal
#     error ("type does not have a definite number of fields", during SROA).
#
#   * `Zero` carries over unchanged, and generated code can go further than
#     dispatch does here: it can inspect the tangent field types and drop
#     whole terms at compile time, rather than relying on a
#     `::NTuple{3,Zero}` method existing.
#
#   * Watch the test suite. Each distinct specialization costs 10-18 s to
#     compile, so a generic version that compiles one kernel per requested
#     set can make the tests far slower than the feature is worth. See the
#     `check_bounds` note in .github/workflows/CI.yml.

# The pieces
# =====================================================================

"""
    Zero()

A zero that is known at compile time. Used in the source terms below to say
"this row contributes nothing to this parameter's derivative" -- e.g. E₂ does
not depend on T₁ -- so that the term costs nothing instead of being a runtime
multiply-add by zero.
"""
struct Zero end

@inline Base.:*(::Zero, ::Number) = Zero()
@inline Base.:*(::Number, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()
@inline Base.:+(x::Number, ::Zero) = x
@inline Base.:+(::Zero, x::Number) = x
@inline Base.:+(::Zero, ::Zero) = Zero()

"""
    ∂Ω∂T₁T₂(∂T₁, ∂T₂)
    ∂Ω∂T₁T₂B₁(∂T₁, ∂T₂, ∂B₁)

The tangent states: one ordinary configuration state matrix per
differentiated tissue property, accessed by name as `∂Ω.∂T₁`. Which of the two
is in play decides which derivatives the simulation propagates.

The same two types also hold the *output* arrays -- one magnetization
derivative per property -- so the field type is `AbstractArray` rather than
`AbstractConfigurationStates`: it has to admit the output views as well as
the states, and it leaves room for a forward-sensitivity isochromat model,
whose `Isochromat` is a `FieldVector` and not a configuration state either.
"""
struct ∂Ω∂T₁T₂{A<:AbstractArray}
    ∂T₁::A
    ∂T₂::A
end

struct ∂Ω∂T₁T₂B₁{A<:AbstractArray}
    ∂T₁::A
    ∂T₂::A
    ∂B₁::A
end

# Whether a B₁ tangent is being propagated. A compile-time constant, so
# `has_B₁(∂Ω) && ...` costs nothing. Not to be confused with `hasB₁(p)` from
# tissueproperties.jl, which asks whether the tissue *properties* carry a B₁:
# the properties can have one without its derivative being asked for.
@inline has_B₁(::∂Ω∂T₁T₂) = false
@inline has_B₁(::∂Ω∂T₁T₂B₁) = true

"""
    S

The `(value, tangents)` pair every operator below takes, destructured in the
argument list as `(Ω, ∂Ω)`. Deliberately a short, unobtrusive name: the
annotation is there to dispatch -- to pick these methods over the ordinary
ones in epg.jl, which take an `AbstractConfigurationStates` -- rather than to
describe the arguments, which the destructuring and the Ω/∂Ω convention
already do.

The value is a configuration state matrix during the simulation and an output
array when sampling. Which of the two tangent sets it holds is then a matter
of `has_B₁(∂Ω)` above.

Parametric in the value type, so a forward-sensitivity isochromat model can
dispatch its own methods on `S{<:Isochromat}` against this one's
`S{<:AbstractConfigurationStates}` and reuse the same tangent structs. A bare
`Tuple` would not leave that room, since the two models share `E₁`, `E₂` and
`off_resonance_rotation`.
"""
const S{A} = Union{Tuple{A,<:∂Ω∂T₁T₂},Tuple{A,<:∂Ω∂T₁T₂B₁}}

# Relaxation factors
# =====================================================================
# The only quantities in the simulation that depend on T₁ or T₂. Where an
# ordinary simulation gets a plain relaxation factor, the operators below get
# the pair `(E₁, ∂E₁∂T₁)` of that factor and its derivative w.r.t. its own
# time constant, which they destructure straight out of their argument list.
# E₁ = exp(-Δt/T₁) so dE₁/dT₁ = E₁ Δt/T₁², and likewise dE₂/dT₂ = E₂ Δt/T₂².

@inline function E₁(::S, Δt, T₁)
    E₁ = _E₁(Δt, T₁)
    return (E₁, E₁ * Δt / T₁^2)
end

@inline function E₂(::S, Δt, T₂)
    E₂ = _E₂(Δt, T₂)
    return (E₂, E₂ * Δt / T₂^2)
end

# Off-resonance depends only on B₀, which is not among the supported
# derivatives, so it stays an ordinary number.
@inline off_resonance_rotation(::S, Δt, p) = off_resonance_rotation(Δt, p)

# Relaxation: decay!, rotate_decay!, regrowth!
# =====================================================================
# An ordinary simulation just does `Ω .*= f` with the per-row factors
# f = (E₂, E₂, E₁): the transverse rows (F₊, F̄₋) are scaled by E₂ and the
# longitudinal row (Z) by E₁. Since those factors depend on T₁ and T₂, each
# tangent picks up a source term by the product rule,
#
#   ∂(f Ω)/∂θ = f ∂Ω/∂θ + (∂f/∂θ) Ω
#
# which puts T₂'s source in the first two rows and T₁'s in the third. Every
# tangent is updated before Ω itself, because the source term reads the old
# Ω. B₁ has no source term: relaxation does not depend on it, so its tangent
# simply relaxes along with the state.

@inline function decay!((Ω, ∂Ω)::S, (E₁, ∂E₁∂T₁), (E₂, ∂E₂∂T₂))
    f = (E₂, E₂, E₁)
    ∂f∂T₁ = (Zero(), Zero(), ∂E₁∂T₁)
    ∂f∂T₂ = (∂E₂∂T₂, ∂E₂∂T₂, Zero())

    ∂Ω.∂T₁ .= f .* ∂Ω.∂T₁ .+ ∂f∂T₁ .* Ω
    ∂Ω.∂T₂ .= f .* ∂Ω.∂T₂ .+ ∂f∂T₂ .* Ω
    has_B₁(∂Ω) && (∂Ω.∂B₁ .*= f)
    Ω .*= f
    return nothing
end

@inline function rotate_decay!((Ω, ∂Ω)::S, (E₁, ∂E₁∂T₁), (E₂, ∂E₂∂T₂), eⁱᶿ)
    f = (E₂ * eⁱᶿ, E₂ * conj(eⁱᶿ), complex(E₁))
    ∂f∂T₁ = (Zero(), Zero(), complex(∂E₁∂T₁))
    ∂f∂T₂ = (∂E₂∂T₂ * eⁱᶿ, ∂E₂∂T₂ * conj(eⁱᶿ), Zero())

    ∂Ω.∂T₁ .= f .* ∂Ω.∂T₁ .+ ∂f∂T₁ .* Ω
    ∂Ω.∂T₂ .= f .* ∂Ω.∂T₂ .+ ∂f∂T₂ .* Ω
    has_B₁(∂Ω) && (∂Ω.∂B₁ .*= f)
    Ω .*= f
    return nothing
end

# regrowth! does Z₀ += 1 - E₁, so the T₁ tangent needs Z₀ += -∂E₁∂T₁. Passing
# the unmodified regrowth! a factor of `1 + ∂E₁∂T₁` gives exactly that
# (1 - (1 + ∂E₁∂T₁) = -∂E₁∂T₁), and reuses its GPU lane-1 guard rather than
# duplicating it here. T₂ and B₁ are untouched: E₁ does not depend on them.

@inline function regrowth!((Ω, ∂Ω)::S, (E₁, ∂E₁∂T₁))
    regrowth!(Ω, E₁)
    regrowth!(∂Ω.∂T₁, 1 + ∂E₁∂T₁)
    return nothing
end

# RF excitation
# =====================================================================
# B₁ scales the flip angle, so the rotation matrix R depends on it:
#
#   ∂(R Ω)/∂B₁ = R ∂Ω/∂B₁ + (∂R/∂α)(∂α/∂B₁) Ω
#
# with α = deg2rad(RF) * B₁, hence ∂α/∂B₁ = deg2rad(RF). T₁ and T₂ have no
# source term here: the rotation does not depend on them.

"""
    ∂RF_rotation_matrix∂B₁(RF, p::AbstractTissueProperties)

Derivative of [`RF_rotation_matrix`](@ref) w.r.t. `B₁`, which scales the flip
angle: `∂R/∂B₁ = (∂R/∂α)(∂α/∂B₁)` with `∂α/∂B₁ = deg2rad(RF)`. Only ever
called when the tissue properties carry a `B₁`, since otherwise there is no
`B₁` derivative to propagate.
"""
@inline function ∂RF_rotation_matrix∂B₁(RF::T, p::AbstractTissueProperties) where {T<:Union{Complex,Quantity{<:Complex}}}
    ∂α∂B₁ = deg2rad(abs(RF))
    α = flip_angle(RF, p)

    x = α / 2
    sinx, cosx = sincos(x)
    sinα, cosα = 2 * sinx * cosx, 2 * cosx^2 - one(α)
    cosφ, sinφ = reim(normalize(RF))
    sin2φ, cos2φ = 2 * sinφ * cosφ, 2 * cosφ^2 - one(α)
    ℯⁱᵠ = complex(cosφ, sinφ)
    ℯ²ⁱᵠ = complex(cos2φ, sin2φ)
    ℯ⁻ⁱᵠ = conj(ℯⁱᵠ)
    ℯ⁻²ⁱᵠ = conj(ℯ²ⁱᵠ)
    # d(cos²x)/dα = -sinα/2, d(sin²x)/dα = sinα/2, d(sinα)/dα = cosα
    R₁₁, R₁₂, R₁₃ = -sinα / 2, ℯ²ⁱᵠ * sinα / 2, -im * ℯⁱᵠ * cosα
    R₂₁, R₂₂, R₂₃ = ℯ⁻²ⁱᵠ * sinα / 2, -sinα / 2, 1im * ℯ⁻ⁱᵠ * cosα
    R₃₁, R₃₂, R₃₃ = -im * ℯ⁻ⁱᵠ * cosα / 2, 1im * ℯⁱᵠ * cosα / 2, -sinα
    return SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃) * ∂α∂B₁
end

@inline function ∂RF_rotation_matrix∂B₁(RF::T, p::AbstractTissueProperties) where {T<:Union{Real,Quantity{<:Real}}}
    ∂α∂B₁ = deg2rad(RF)
    α = flip_angle(RF, p)

    x = α / 2
    sinx, cosx = sincos(x)
    sinα, cosα = 2 * sinx * cosx, 2 * cosx^2 - one(α)
    R₁₁, R₁₂, R₁₃ = -sinα / 2, -sinα / 2, -cosα
    R₂₁, R₂₂, R₂₃ = -sinα / 2, -sinα / 2, -cosα
    R₃₁, R₃₂, R₃₃ = cosα / 2, cosα / 2, -sinα
    return SMatrix{3,3}(R₁₁, R₂₁, R₃₁, R₁₂, R₂₂, R₃₂, R₁₃, R₂₃, R₃₃) * ∂α∂B₁
end

@inline function excite!((Ω, ∂Ω)::S, RF, p::AbstractTissueProperties)
    # (∂R/∂B₁) Ω, from the old Ω, before excite! overwrites it. Note this is
    # deliberately not skipped when the flip angle is zero the way excite!
    # itself is: a voxel with B₁ = 0 has α = 0 but a perfectly well-defined,
    # nonzero ∂R/∂B₁.
    source = has_B₁(∂Ω) ? ∂RF_rotation_matrix∂B₁(RF, p) * Ω.matrix : Zero()

    excite!(Ω, RF, p)
    excite!(∂Ω.∂T₁, RF, p)
    excite!(∂Ω.∂T₂, RF, p)
    if has_B₁(∂Ω)
        excite!(∂Ω.∂B₁, RF, p)
        ∂Ω.∂B₁ .+= source
    end
    return nothing
end

# Inversion
# =====================================================================
# The adiabatic inversion is B₁-insensitive, so it acts alike on everything.
# The B₁-dependent one multiplies Z by cos(θ) with θ = π B₁, so it is an
# ordinary per-row factor with a B₁ source term.

@inline function invert!((Ω, ∂Ω)::S)
    invert!(Ω)
    invert!(∂Ω.∂T₁)
    invert!(∂Ω.∂T₂)
    has_B₁(∂Ω) && invert!(∂Ω.∂B₁)
    return nothing
end

@inline function invert!((Ω, ∂Ω)::S, p::AbstractTissueProperties)
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    sinθ, cosθ = sincos(θ)
    f = (one(cosθ), one(cosθ), cosθ)
    ∂f∂B₁ = (Zero(), Zero(), -π * sinθ)

    ∂Ω.∂T₁ .*= f
    ∂Ω.∂T₂ .*= f
    has_B₁(∂Ω) && (∂Ω.∂B₁ .= f .* ∂Ω.∂B₁ .+ ∂f∂B₁ .* Ω)
    Ω .*= f
    return nothing
end

# Operators that depend on no tissue property at all
# =====================================================================
# Dephasing is a permutation, spoiling zeroes the transverse states and
# diffusion scales by a precomputed matrix -- all the same linear map on the
# state and on every tangent.

@inline function dephasing!((Ω, ∂Ω)::S)
    dephasing!(Ω)
    dephasing!(∂Ω.∂T₁)
    dephasing!(∂Ω.∂T₂)
    has_B₁(∂Ω) && dephasing!(∂Ω.∂B₁)
    return nothing
end

@inline function spoil!((Ω, ∂Ω)::S)
    spoil!(Ω)
    spoil!(∂Ω.∂T₁)
    spoil!(∂Ω.∂T₂)
    has_B₁(∂Ω) && spoil!(∂Ω.∂B₁)
    return nothing
end

@inline function diffuse!((Ω, ∂Ω)::S, diffusion_decay)
    diffuse!(Ω, diffusion_decay)
    diffuse!(∂Ω.∂T₁, diffusion_decay)
    diffuse!(∂Ω.∂T₂, diffusion_decay)
    has_B₁(∂Ω) && diffuse!(∂Ω.∂B₁, diffusion_decay)
    return nothing
end

# The per-state diffusion decay factors depend on D, not on the
# differentiated properties, so this is an ordinary matrix.
@inline diffusion_decay_matrix((Ω, ∂Ω)::S, D) = diffusion_decay_matrix(Ω, D)

# Initial conditions and sampling
# =====================================================================
# The state starts at thermal equilibrium whatever the tissue properties, so
# every tangent starts at zero.

@inline function initial_conditions!((Ω, ∂Ω)::S)
    initial_conditions!(Ω)
    fill!(∂Ω.∂T₁, zero(eltype(∂Ω.∂T₁)))
    fill!(∂Ω.∂T₂, zero(eltype(∂Ω.∂T₂)))
    has_B₁(∂Ω) && fill!(∂Ω.∂B₁, zero(eltype(∂Ω.∂B₁)))
    return nothing
end

# The output is a magnetization array plus one array per derivative, so they
# all come straight out of the simulation with no unpacking afterwards.

@inline function sample_transverse!((output, ∂output)::S, index, (Ω, ∂Ω)::S)
    sample_transverse!(output, index, Ω)
    sample_transverse!(∂output.∂T₁, index, ∂Ω.∂T₁)
    sample_transverse!(∂output.∂T₂, index, ∂Ω.∂T₂)
    has_B₁(∂Ω) && sample_transverse!(∂output.∂B₁, index, ∂Ω.∂B₁)
    return nothing
end

# Allocating the states and the output
# =====================================================================

"""
    initialize_derivative_states(resource, sequence::EPGSimulator, Val(derivatives))

Allocate `(Ω, ∂Ω)`: one ordinary configuration state matrix for the value and
one per differentiated property, each allocated exactly as an ordinary
simulation's state would be.
"""
@inline function initialize_derivative_states(resource, sequence::EPGSimulator, ::Val{(:T₁, :T₂)})
    Ω = initialize_states(resource, sequence)
    ∂T₁ = initialize_states(resource, sequence)
    ∂T₂ = initialize_states(resource, sequence)
    return (Ω, ∂Ω∂T₁T₂(∂T₁, ∂T₂))
end

@inline function initialize_derivative_states(resource, sequence::EPGSimulator, ::Val{(:T₁, :T₂, :B₁)})
    Ω = initialize_states(resource, sequence)
    ∂T₁ = initialize_states(resource, sequence)
    ∂T₂ = initialize_states(resource, sequence)
    ∂B₁ = initialize_states(resource, sequence)
    return (Ω, ∂Ω∂T₁T₂B₁(∂T₁, ∂T₂, ∂B₁))
end

# Turn the NamedTuple of output arrays (one per derivative) into the matching
# tangent struct.
@inline tangents_of(∂m::NamedTuple{(:T₁, :T₂)}) = ∂Ω∂T₁T₂(∂m.T₁, ∂m.T₂)
@inline tangents_of(∂m::NamedTuple{(:T₁, :T₂, :B₁)}) = ∂Ω∂T₁T₂B₁(∂m.T₁, ∂m.T₂, ∂m.B₁)

# Driver
# =====================================================================

"""
    supported_derivatives(parameters)

Which tissue properties forward sensitivity can differentiate with respect to:
`T₁` and `T₂` always, plus `B₁` when the parameters carry one.
"""
@inline supported_derivatives(::Type{P}) where {P<:AbstractTissueProperties} =
    :B₁ ∈ fieldnames(P) ? (:T₁, :T₂, :B₁) : (:T₁, :T₂)

@inline supported_derivatives(parameters::AbstractVector{<:AbstractTissueProperties}) =
    supported_derivatives(eltype(parameters))

function _validate_derivatives(requested, parameters)
    supported = supported_derivatives(parameters)
    isempty(requested) && error("No derivatives requested")
    requested ⊆ supported || error(
        "Forward sensitivity supports $supported for tissue properties of type " *
        "$(eltype(parameters)), got $requested")
    return nothing
end

"""
    simulate_derivatives_forward_sensitivity(requested_derivatives, sequence, parameters)
    simulate_derivatives_forward_sensitivity(sequence, parameters)

Simulate the magnetization of an EPG-based sequence (e.g. [`FISP3D`](@ref),
[`FISP2D`](@ref)) together with its exact partial derivatives with respect to
the requested tissue properties. The derivatives come from forward sensitivity
propagation inside the simulation -- not finite differences, not reverse-mode
AD -- so they are exact up to floating point rather than limited by a step
size. See the comments at the top of this file for how.

`requested_derivatives` must be a subset of [`supported_derivatives`](@ref)
for the given parameters; omit it to get all of them. `T₁` and `T₂` are always
propagated together, so asking for one of them costs the same as asking for
both; `B₁` is propagated only when requested.

Runs on the GPU when `sequence` and `parameters` are on the GPU, and on
threaded CPU otherwise, matching `simulate_magnetization`.

# Returns
`(magnetization, ∂magnetization)`, where `∂magnetization` is a `NamedTuple`
with one entry per requested derivative, each the same size and eltype as
`magnetization` (the same shape `simulate_derivatives_finite_difference`
returns).
"""
function simulate_derivatives_forward_sensitivity(
    requested_derivatives::Symbols,
    sequence::EPGSimulator,
    parameters::AbstractVector{<:AbstractTissueProperties}
)
    _validate_derivatives(requested_derivatives, parameters)

    on_gpu = _all_arrays_are_cuarrays(sequence)
    on_gpu == _all_arrays_are_cuarrays(parameters) || throw(ArgumentError(
        "Both sequence and parameters must be on the GPU or not on the GPU"))
    resource = on_gpu ? CUDALibs() : CPUThreads()

    # Spelled out with literal `Val`s so that each branch is concretely typed;
    # deriving the set from the runtime `requested_derivatives` instead leaves
    # the whole call inferring badly.
    magnetization, ∂propagated = if :B₁ ∈ requested_derivatives
        _simulate(Val((:T₁, :T₂, :B₁)), resource, sequence, parameters)
    else
        _simulate(Val((:T₁, :T₂)), resource, sequence, parameters)
    end

    return magnetization, NamedTuple{requested_derivatives}(
        map(θ -> ∂propagated[θ], requested_derivatives))
end

simulate_derivatives_forward_sensitivity(sequence::EPGSimulator, parameters::AbstractVector{<:AbstractTissueProperties}) =
    simulate_derivatives_forward_sensitivity(supported_derivatives(parameters), sequence, parameters)

# Single-voxel convenience methods.
simulate_derivatives_forward_sensitivity(requested_derivatives::Symbols, sequence::EPGSimulator, p::AbstractTissueProperties) =
    simulate_derivatives_forward_sensitivity(requested_derivatives, sequence, StructVector([p]))

simulate_derivatives_forward_sensitivity(sequence::EPGSimulator, p::AbstractTissueProperties) =
    simulate_derivatives_forward_sensitivity(sequence, StructVector([p]))

# Allocate the magnetization and one derivative array per propagated property,
# then fill them in.
function _simulate(::Val{derivatives}, resource, sequence, parameters) where {derivatives}
    T = output_eltype(sequence)
    dims = (output_size(sequence)..., length(parameters))

    magnetization = _allocate_array_on_resource(resource, T, dims)
    ∂magnetization = NamedTuple{derivatives}(
        map(_ -> _allocate_array_on_resource(resource, T, dims), derivatives))

    _simulate!(magnetization, ∂magnetization, resource, sequence, parameters)
    return magnetization, ∂magnetization
end

# One voxel's slice of the magnetization array and of each derivative array,
# sliced the way `simulate_magnetization!` slices. The kernel below uses `view`
# instead, for the same reason it does: `selectdim`'s bounds-check path builds
# a string, which cannot be compiled for the GPU.
@inline function _voxel_output(magnetization, ∂magnetization, voxel)
    vd = ndims(magnetization)
    return (selectdim(magnetization, vd, voxel),
        tangents_of(map(∂m -> selectdim(∂m, vd, voxel), ∂magnetization)))
end

function _simulate!(magnetization, ∂magnetization::NamedTuple{derivatives}, ::CPUThreads, sequence, parameters) where {derivatives}
    Threads.@threads for voxel ∈ eachindex(parameters)
        Ω = initialize_derivative_states(CPUThreads(), sequence, Val(derivatives))
        output = _voxel_output(magnetization, ∂magnetization, voxel)
        simulate_magnetization!(output, sequence, Ω, parameters[voxel])
    end
    return nothing
end

function _simulate!(magnetization, ∂magnetization::NamedTuple{derivatives}, ::CUDALibs, sequence, parameters) where {derivatives}

    function kernel!(magnetization, ∂magnetization, sequence, parameters)
        voxel = cld((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x, WARPSIZE)
        Ω = initialize_derivative_states(CUDALibs(), sequence, Val(derivatives))

        # extra threads in the last block have no voxel to simulate
        voxel > length(parameters) && return nothing

        output = (view(magnetization, :, voxel),
            tangents_of(map(∂m -> view(∂m, :, voxel), ∂magnetization)))
        simulate_magnetization!(output, sequence, Ω, @inbounds parameters[voxel])
        return nothing
    end

    nr_blocks = cld(length(parameters) * WARPSIZE, THREADS_PER_BLOCK)
    CUDA.@sync @cuda blocks = nr_blocks threads = THREADS_PER_BLOCK kernel!(
        magnetization, ∂magnetization, sequence, parameters)
    return nothing
end
