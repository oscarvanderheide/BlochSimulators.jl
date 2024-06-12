"""
    Coordinates{T<:Real}

Basic type that holds the spatial coordinates of a voxel. Note that when performing signal simulations, `StructArray{<:Coordinates}`s are used to store the coordinates of all voxels. Such arrays are created using the `make_coordinates`.

# Fields
- `x::T`: Position of voxel along the x direction
- `y::T`: Position of voxel along the y direction
- `z::T`: Position of voxel along the z direction (not used for 2D simulations)
"""
struct Coordinates{T<:Real}
    x::T
    y::T
    z::T
end

# Needed for f32 and gpu to work
@functor Coordinates
@adapt_structure Coordinates

"""
CoordinatesCollection(x::T, y::T, z::T) where {T<:AbstractArray{<:Real}}

Create a 3D meshgrid of Coordinates from arrays `x`, `y`, and `z` and return it as a StructArray.

# Arguments
- `x::T`: Array of x-coordinates per voxel.
- `y::T`: Array of y-coordinates per voxel.
- `z::T`: Array of z-coordinates per voxel.
"""
function make_coordinates(x::T, y::T, z::T) where {T<:AbstractArray{<:Real}}

    if x isa CuArray
        throw(ArgumentError("Use gpu(make_coordinates(x, y, z)) instead of make_coordinates(gpu(x), gpu(y), gpu(z))"))
    end

    # Create a 3D "meshgrid" with each element being a tuple of x,y,z values
    xyz = Iterators.product(x, y, z)

    # Map the tuples to Coordinates and store them in a StructArray
    return StructArray(Coordinates(x, y, z) for (x, y, z) in xyz) |> vec
end

# macro coordinates s.t. `@coordinates x y z` is equivalent to `make_coordinates(x, y, z)`
macro coordinates(x, y, z)
    quote
        make_coordinates($(esc(x)), $(esc(y)), $(esc(z)))
    end
end
