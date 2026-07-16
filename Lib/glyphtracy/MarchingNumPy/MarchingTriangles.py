from __future__ import annotations
from typing import Tuple, List, Callable
from numpy.typing import NDArray

import numpy

from . import Marching
from . import IndexingTools


marching_triangles: Callable
"""
Convert a 2d :attr:`volume` into isolines
by splitting the volume into triangles and evaluating
where the values in the volume cross a :attr:`level` threshold
along the edges of the triangles.
The triangles are formed by splitting squares along the bottom left - top right diagonal.
Ambiguity resolution is not relevant for this function.

Args
----
volume: NDArray
    volume
level: float | NDArray
    level

Keyword Args
------------
interpolation : {'LINEAR', 'HALFWAY', 'COSINE'}
    Interpolation method for :func:`.FindIntersects.find_intersects`.
step_size : int
    Defaults to 1.

Returns
-------
vertices: NDArray
geometry: NDArray

"""

marching_triangles_reversed: Callable
"""
Convert a 2d :attr:`volume` into isolines
by splitting the volume into triangles and evaluating
where the values in the volume cross a :attr:`level` threshold
along the edges of the triangles.
The triangles are formed by splitting squares along the bottom right - top left diagonal.
Ambiguity resolution is not relevant for this function.

Args
----
volume : NDArray
    volume
level : float | NDArray
    level

Keyword Args
------------
interpolation : {'LINEAR', 'HALFWAY', 'COSINE'}
    Interpolation method for :func:`.FindIntersects.find_intersects`.
step_size : int
    Step size. Defaults to 1.

Returns
-------
vertices: NDArray
geometry: NDArray

"""

VOLUME_TYPE_SLICES: Tuple[Tuple[slice, ...], ...]
VOLUME_TYPE_SLICES = IndexingTools.str_to_index_tuple(
    "[:-1, :-1]",  # 1: x,   y
    "[1: , :-1]",  # 2: x+1, y
    "[1: , 1: ]",  # 4: x+1, y+1
    "[:-1, 1: ]",  # 8: x,   y+1
)

INTERSECT_SLICE_INDEXES: List[Tuple[Tuple[slice, ...], ...]]
INTERSECT_SLICE_INDEXES = [
    IndexingTools.str_to_index_tuple("[:-1, :]", "[1:, :]"),  # x,y --> x+1,y
    IndexingTools.str_to_index_tuple("[:, :-1]", "[:, 1:]"),  # x,y --> x,y+1
    IndexingTools.str_to_index_tuple("[:-1, :-1]", "[1:, 1:]"),  # x,y,z --> x+1,y+1
]

INTERSECT_SLICE_INDEXES_REVERSED: List[Tuple[Tuple[slice, ...], ...]]
INTERSECT_SLICE_INDEXES_REVERSED = [
    IndexingTools.str_to_index_tuple("[:-1, :]", "[1:, :]"),  # x,y --> x+1,y
    IndexingTools.str_to_index_tuple("[:, :-1]", "[:, 1:]"),  # x,y --> x,y+1
    IndexingTools.str_to_index_tuple("[:-1, 1:]", "[1:, :-1]"),  # x,y+1 --> x+1,y
]

EDGE_DELTA: NDArray[numpy.uint8]
EDGE_DELTA = numpy.asarray(
    [
        [0, 0],  # 0 Bottom
        [1, 0],  # 1 Right
        [0, 1],  # 2 Top
        [0, 0],  # 3 Left
        [0, 0],  # 4 Diagonal
    ],
    dtype=numpy.uint8,
)

EDGE_DIRECTION: NDArray[numpy.uint8]
EDGE_DIRECTION = numpy.asarray(
    [
        0,  # 0 Bottom
        1,  # 1 Right
        0,  # 2 Top
        1,  # 3 Left
        2,  # 4 Diagonal
    ],
    dtype=numpy.uint8,
)


# Geometry look up tables

_NONE: List[int] = [-1, -1]

# Squares split like : /

_TOP_HORIZONTAL: List[int] = [4, 3]
_TOP_VERTICAL: List[int] = [2, 4]
_TOP_CORNER: List[int] = [2, 3]

_BOTTOM_VERTICAL: List[int] = [0, 4]
_BOTTOM_CORNER: List[int] = [1, 0]
_BOTTOM_HORIZONTAL: List[int] = [1, 4]

GEOMETRY_LOOKUP_TOP: NDArray[numpy.int8] = numpy.asarray(
    [
        _NONE,
        _TOP_HORIZONTAL,
        _NONE,
        _TOP_HORIZONTAL,
        _TOP_VERTICAL,
        _TOP_CORNER,
        _TOP_VERTICAL,
        _TOP_CORNER,
        _TOP_CORNER[::-1],
        _TOP_VERTICAL[::-1],
        _TOP_CORNER[::-1],
        _TOP_VERTICAL[::-1],
        _TOP_HORIZONTAL[::-1],
        _NONE,
        _TOP_HORIZONTAL[::-1],
        _NONE,
    ],
    dtype=numpy.int8,
)
GEOMETRY_LOOKUP_BOTTOM: NDArray[numpy.int8] = numpy.asarray(
    [
        _NONE,
        _BOTTOM_VERTICAL,
        _BOTTOM_CORNER,
        _BOTTOM_HORIZONTAL,
        _BOTTOM_HORIZONTAL[::-1],
        _BOTTOM_CORNER[::-1],
        _BOTTOM_VERTICAL[::-1],
        _NONE,
        _NONE,
        _BOTTOM_VERTICAL,
        _BOTTOM_CORNER,
        _BOTTOM_HORIZONTAL,
        _BOTTOM_HORIZONTAL[::-1],
        _BOTTOM_CORNER[::-1],
        _BOTTOM_VERTICAL[::-1],
        _NONE,
    ],
    dtype=numpy.int8,
)
# create lookup table by combining top and bottom triangles
GEOMETRY_LOOKUP: NDArray[numpy.int8]
GEOMETRY_LOOKUP = numpy.zeros((16, 4), dtype=numpy.int8)
GEOMETRY_LOOKUP[:, :2] = GEOMETRY_LOOKUP_TOP
GEOMETRY_LOOKUP[:, 2:] = GEOMETRY_LOOKUP_BOTTOM

# Squares split like : \
# variable names with "_" suffix

_TOP_CORNER_: List[int] = [2, 1]
_TOP_HORIZONTAL_: List[int] = [1, 4]
_TOP_VERTICAL_: List[int] = [2, 4]

_BOTTOM_CORNER_: List[int] = [0, 3]
_BOTTOM_VERTICAL_: List[int] = [4, 0]
_BOTTOM_HORIZONTAL_: List[int] = [4, 3]

GEOMETRY_LOOKUP_TOP_: NDArray[numpy.int8] = numpy.asarray(
    [
        _NONE,
        _NONE,
        _TOP_HORIZONTAL_,
        _TOP_HORIZONTAL_,
        _TOP_CORNER_,
        _TOP_CORNER_,
        _TOP_VERTICAL_,
        _TOP_VERTICAL_,
        _TOP_VERTICAL_[::-1],
        _TOP_VERTICAL_[::-1],
        _TOP_CORNER_[::-1],
        _TOP_CORNER_[::-1],
        _TOP_HORIZONTAL_[::-1],
        _TOP_HORIZONTAL_[::-1],
        _NONE,
        _NONE,
    ],
    dtype=numpy.int8,
)
GEOMETRY_LOOKUP_BOTTOM_: NDArray[numpy.int8] = numpy.asarray(
    [
        _NONE,
        _BOTTOM_CORNER_,
        _BOTTOM_VERTICAL_,
        _BOTTOM_HORIZONTAL_,
        _NONE,
        _BOTTOM_CORNER_,
        _BOTTOM_VERTICAL_,
        _BOTTOM_HORIZONTAL_,
        _BOTTOM_HORIZONTAL_[::-1],
        _BOTTOM_VERTICAL_[::-1],
        _BOTTOM_CORNER_[::-1],
        _NONE,
        _BOTTOM_HORIZONTAL_[::-1],
        _BOTTOM_VERTICAL_[::-1],
        _BOTTOM_CORNER_[::-1],
        _NONE,
    ],
    dtype=numpy.int8,
)

# create lookup table by combining top and bottom triangles
GEOMETRY_LOOKUP_REVERSED: NDArray[numpy.int8]
GEOMETRY_LOOKUP_REVERSED = numpy.zeros((16, 4), dtype=numpy.int8)
GEOMETRY_LOOKUP_REVERSED[:, :2] = GEOMETRY_LOOKUP_TOP_
GEOMETRY_LOOKUP_REVERSED[:, 2:] = GEOMETRY_LOOKUP_BOTTOM_


marching_triangles = Marching.marching_factory(
    nD=2,
    nEdges=3,
    intersect_slice_indexes=INTERSECT_SLICE_INDEXES,
    volume_type_slices=VOLUME_TYPE_SLICES,
    ambiguity_resolution=None,
    geometry_array=GEOMETRY_LOOKUP,
    edge_direction=EDGE_DIRECTION,
    edge_delta=EDGE_DELTA,
)

marching_triangles_reversed = Marching.marching_factory(
    nD=2,
    nEdges=3,
    intersect_slice_indexes=INTERSECT_SLICE_INDEXES_REVERSED,
    volume_type_slices=VOLUME_TYPE_SLICES,
    ambiguity_resolution=None,
    geometry_array=GEOMETRY_LOOKUP_REVERSED,
    edge_direction=EDGE_DIRECTION,
    edge_delta=EDGE_DELTA,
)
