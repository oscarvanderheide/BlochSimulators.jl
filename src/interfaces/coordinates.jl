"""
    Coordinates{T<:Real}

Basic type that holds the spatial coordinates of a voxel.

# Fields
- `x::T`: Position of voxel along the x direction
- `y::T`: Position of voxel along the y direction
- `z::T`: Position of voxel along the z direction (not used for 2D simulations)

# Notes
The alias `xyz` is also defined for this struct.
"""
struct Coordinates{T<:Real}
    x::T
    y::T
    z::T
end

const xyz = Coordinates

@functor Coordinates
@adapt_structure Coordinates