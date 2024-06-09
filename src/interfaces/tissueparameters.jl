"""
    AbstractTissueParameters{N,T} <: FieldVector{N,T}

Abstract type for structs that hold different combinations of tissue parameters.
# Possible fields

- `T₁::T`: T₁ relaxation parameters of a voxel
- `T₂::T`: T₂ relaxation parameters of a voxel
- `B₁::T`: Scaling factor for effective B₁ excitation field within a voxel
- `B₀::T`: Off-resonance with respect to main magnetic field within a voxel
- `ρˣ::T`: Real part of proton density within a voxel
- `ρʸ::T`: Imaginary part of proton density within a voxel

The structs are subtypes of FieldVector, which is a StaticVector with named fields
(see the documentation of StaticArrays.jl). There are three reasons for letting the structs
be subtypes of FieldVector:
1) FieldVectors/StaticVectors have sizes that are known at compile time. This is beneficial for performance reasons
2) The named fields improve readability of the code (e.g. `p.B₁` vs `p[3]`)
3) Linear algebra operations can be performed on instances of the structs. This allows, for example,
   subtraction (without having to manually define methods) and that is useful for comparing parameter maps.
"""
abstract type AbstractTissueParameters{N,T} <: FieldVector{N,T} end

# Define different TissueParameters types
"""
    T₁T₂{T} <: AbstractTissueParameters{2,T}
"""
struct T₁T₂{T} <: AbstractTissueParameters{2,T}
    T₁::T
    T₂::T
end

"""
    T₁T₂B₁{T} <: AbstractTissueParameters{3,T}
"""
struct T₁T₂B₁{T} <: AbstractTissueParameters{3,T}
    T₁::T
    T₂::T
    B₁::T
end

"""
    T₁T₂B₀{T} <: AbstractTissueParameters{2,T}
"""
struct T₁T₂B₀{T} <: AbstractTissueParameters{3,T}
    T₁::T
    T₂::T
    B₀::T
end

"""
    T₁T₂B₁B₀{T} <: AbstractTissueParameters{4,T}
"""
struct T₁T₂B₁B₀{T} <: AbstractTissueParameters{4,T}
    T₁::T
    T₂::T
    B₁::T
    B₀::T
end

# For each subtype of AbstractTissueParameters created above, we use meta-programming to create
# additional types that also hold proton density (ρˣ and ρʸ) and/or spatial coordinates
# (x,y for 2D simulations or x,y,z for 3D simulations).
#
# For example, given T₁T₂ <: AbstractTissueParameters, we automatically define:
#
# struct T₁T₂ρˣρʸ{T} <: AbstractTissueParameters{4,T}
#     T₁::T
#     T₂::T
#     ρˣ::T
#     ρʸ::T
# end
#
# as well as T₁T₂xy, T₁T₂, T₁T₂ρˣρʸ and T₁T₂ρˣρʸ.

for P in subtypes(AbstractTissueParameters)

    # Create new struct name by appending new fieldnames to name of struct
    structname_ρˣρʸ = Symbol(fieldnames(P)..., :ρˣρʸ)

    # Create new tuples of fieldnames
    fnames_ρˣρʸ = (fieldnames(P)..., :ρˣ, :ρʸ)

    fnames_typed_ρˣρʸ = [:($(fn)::T) for fn ∈ fnames_ρˣρʸ]

    N = length(fieldnames(P))

    @eval begin
        """
            $($(structname_ρˣρʸ)){T} <: AbstractTissueParameters{$($(N+2)),T}
        """
        struct $(structname_ρˣρʸ){T} <: AbstractTissueParameters{$(N + 2),T}
            $(fnames_typed_ρˣρʸ...)
        end
    end
end

# The following is needed to make sure that operations with/on subtypes of AbstractTissueParameters return the appropriate type, see the FieldVector example from StaticArrays documentation
for S in subtypes(AbstractTissueParameters)
    @eval StaticArrays.similar_type(::Type{$(S){T}}, ::Type{T}, s::Size{(fieldcount($(S)),)}) where {T} = $(S){T}
end

# Define trait functions to check whether B₁ or B₀ is part of the type
# Set default value to false:
hasB₁(::AbstractTissueParameters) = false
hasB₀(::AbstractTissueParameters) = false

for P in subtypes(AbstractTissueParameters)
    @eval hasB₁(::$(P)) = $(:B₁ ∈ fieldnames(P))
    @eval hasB₀(::$(P)) = $(:B₀ ∈ fieldnames(P))
end

# Programatically export all subtypes of AbstractTissueParameters
for P in subtypes(AbstractTissueParameters)
    @eval export $(Symbol(nameof(P)))
end

# Function to get the nonlinear part of the tissue parameters
function get_nonlinear_part(p::Type{<:AbstractTissueParameters})
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