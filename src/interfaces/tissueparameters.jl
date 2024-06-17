"""
    AbstractTissueProperties{N,T} <: FieldVector{N,T}

Abstract type for custom structs that hold tissue properties used for a simulation within one voxel. For simulations, `StructArray{<:AbstractTissueProperties}`s are used that can be assembled with the `@parameters` macro.

# Possible fields:
- `T₁::T`: T₁ relaxation parameters of a voxel
- `T₂::T`: T₂ relaxation parameters of a voxel
- `B₁::T`: Scaling factor for effective B₁ excitation field within a voxel
- `B₀::T`: Off-resonance with respect to main magnetic field within a voxel
- `ρˣ::T`: Real part of proton density within a voxel
- `ρʸ::T`: Imaginary part of proton density within a voxel

# Implementation details:
The structs are subtypes of FieldVector, which is a StaticVector with named fields
(see the documentation of StaticArrays.jl). There are three reasons for letting the structs
be subtypes of FieldVector:
1) FieldVectors/StaticVectors have sizes that are known at compile time. This is beneficial for performance reasons
2) The named fields improve readability of the code (e.g. `p.B₁` vs `p[3]`)
3) Linear algebra operations can be performed on instances of the structs. This allows, for example, subtraction (without having to manually define methods) and that is useful for comparing parameter maps.
"""
abstract type AbstractTissueProperties{N,T} <: FieldVector{N,T} end

# Define different TissueParameters types
"""
    T₁T₂{T} <: AbstractTissueProperties{2,T}
"""
struct T₁T₂{T} <: AbstractTissueProperties{2,T}
    T₁::T
    T₂::T
end

"""
    T₁T₂B₁{T} <: AbstractTissueProperties{3,T}
"""
struct T₁T₂B₁{T} <: AbstractTissueProperties{3,T}
    T₁::T
    T₂::T
    B₁::T
end

"""
    T₁T₂B₀{T} <: AbstractTissueProperties{2,T}
"""
struct T₁T₂B₀{T} <: AbstractTissueProperties{3,T}
    T₁::T
    T₂::T
    B₀::T
end

"""
    T₁T₂B₁B₀{T} <: AbstractTissueProperties{4,T}
"""
struct T₁T₂B₁B₀{T} <: AbstractTissueProperties{4,T}
    T₁::T
    T₂::T
    B₁::T
    B₀::T
end

# For each subtype of AbstractTissueProperties created above, we use meta-programming to create
# additional types that also hold proton density (ρˣ and ρʸ)
#
# For example, given T₁T₂ <: AbstractTissueProperties, we automatically define:
#
# struct T₁T₂ρˣρʸ{T} <: AbstractTissueProperties{4,T}
#     T₁::T
#     T₂::T
#     ρˣ::T
#     ρʸ::T
# end
#
# as well as T₁T₂xy, T₁T₂, T₁T₂ρˣρʸ and T₁T₂ρˣρʸ.

for P in subtypes(AbstractTissueProperties)

    # Create new struct name by appending new fieldnames to name of struct
    structname_ρˣρʸ = Symbol(fieldnames(P)..., :ρˣρʸ)

    # Create new tuples of fieldnames
    fnames_ρˣρʸ = (fieldnames(P)..., :ρˣ, :ρʸ)

    fnames_typed_ρˣρʸ = [:($(fn)::T) for fn ∈ fnames_ρˣρʸ]

    N = length(fieldnames(P))

    @eval begin
        """
            $($(structname_ρˣρʸ)){T} <: AbstractTissueProperties{$($(N+2)),T}
        """
        struct $(structname_ρˣρʸ){T} <: AbstractTissueProperties{$(N + 2),T}
            $(fnames_typed_ρˣρʸ...)
        end
    end
end

# The following is needed to make sure that operations with/on subtypes of AbstractTissueProperties return the appropriate type, see the FieldVector example from StaticArrays documentation
for S in subtypes(AbstractTissueProperties)
    @eval StaticArrays.similar_type(::Type{$(S){T}}, ::Type{T}, s::Size{(fieldcount($(S)),)}) where {T} = $(S){T}
end

# Define trait functions to check whether B₁ or B₀ is part of the type
# Set default value to false:
hasB₁(::AbstractTissueProperties) = false
hasB₀(::AbstractTissueProperties) = false

for P in subtypes(AbstractTissueProperties)
    @eval hasB₁(::$(P)) = $(:B₁ ∈ fieldnames(P))
    @eval hasB₀(::$(P)) = $(:B₀ ∈ fieldnames(P))
end

# Programatically export all subtypes of AbstractTissueProperties
for P in subtypes(AbstractTissueProperties)
    @eval export $(Symbol(nameof(P)))
end

# Function to get the nonlinear part of the tissue properties
function get_nonlinear_part(p::Type{<:AbstractTissueProperties})
    parameter_set = fieldnames(p)
    if (:ρˣ ∉ parameter_set) && (:ρʸ ∉ parameter_set)
        return p
    elseif (:ρˣ ∈ parameter_set) && (:ρʸ ∈ parameter_set)

        if p <: T₁T₂ρˣρʸ
            return T₁T₂
        elseif p <: T₁T₂B₁ρˣρʸ
            return T₁T₂B₁
        elseif p <: T₁T₂B₀ρˣρʸ
            return T₁T₂B₀
        elseif p <: T₁T₂B₁B₀ρˣρʸ
            return T₁T₂B₁B₀
        else
            error("Unknown parameter type: $p")
        end
    else
        error("Either both :ρˣ and :ρʸ should be included, or neither should be")
    end
end

"""
    macro parameters(args...)

Create a `StructArray{<:AbstractTissueProperties}` with the actual struct type being determined by the arguments passed to the macro.

# Examples
```julia
# Create a StructArray{T₁T₂} with T₁ and T₂ values
T₁, T₂ = rand(100), 0.1*rand(100)
parameters = @parameters T₁ T₂

# Create a StructArray{T₁T₂B₁} with T₁, T₂ and B₁ values
T₁, T₂, B₁ = rand(100), 0.1*rand(100), ones(100)
parameters = @parameters T₁ T₂ B₁

# Create a StructArray{T₁T₂B₀} with T₁, T₂ and B₀ values
# This time use the aliases that don't use unicode characters
T1, T2, B0 = rand(100), 0.1*rand(100), ones(100)
parameters = @parameters T1 T2 B0
```
"""
macro parameters(args...)
    type_name = Symbol(join(string.(args)))
    first_arg_type = :(eltype($(esc(args[1]))))
    return :(StructArray{$(type_name){$(first_arg_type)}}(($(esc.(args)...),)))
end

# Define aliases for the tissue parameter types that do not use unicode characters such that, for example, `@parameters T1 T2 B0` is equivalent to `@parameters T₁ T₂ B₀`. This makes it easier for users of the package to generate tissue parameter arrays without having to type unicode characters.
const T1T2 = T₁T₂
const T1T2B1 = T₁T₂B₁
const T1T2B0 = T₁T₂B₀
const T1T2B1B0 = T₁T₂B₁B₀

const T1T2PDxPDy = T₁T₂ρˣρʸ
const T1T2B1PDxPDy = T₁T₂B₁ρˣρʸ
const T1T2B0PDxPDy = T₁T₂B₀ρˣρʸ
const T1T2B1B0PDxPDy = T₁T₂B₁B₀ρˣρʸ