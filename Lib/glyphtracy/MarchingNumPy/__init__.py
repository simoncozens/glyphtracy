from . import MarchingCubesLorensen
from . import MarchingSquares
from . import MarchingTriangles

marching_cubes_lorensen = MarchingCubesLorensen.marching_cubes_lorensen
marching_squares = MarchingSquares.marching_squares
marching_triangles = MarchingTriangles.marching_triangles
marching_triangles_reversed = MarchingTriangles.marching_triangles_reversed

__all__ = [
    "MarchingCubesLorensen",
    "MarchingSquares",
    "MarchingTriangles",
]
