from __future__ import annotations
from typing import Dict, Tuple, Callable, Collection, Union, Optional, Any
from numpy.typing import NDArray

import numpy

from . import Checking
from . import FindIntersects
from . import VolumeTypes
from . import LookUpGeometry
from . import ConvertIndexes
from . import Types


def marching_factory(
    *,
    nD: int,
    nEdges: int,
    intersect_slice_indexes: Collection[Any],
    volume_type_slices: Collection[Tuple[slice, ...]],
    ambiguity_resolution: Optional[Callable[[NDArray, NDArray], None]] = None,
    geometry_array: NDArray[numpy.integer[Any]],
    edge_direction: NDArray[numpy.unsignedinteger[Any]],
    edge_delta: NDArray[numpy.unsignedinteger[Any]],
) -> Callable:
    """
    Factory method for creating marching methods.

    Keyword Args:
        nD (int): Number of dimensions.
        nDir (int): Number of edge directions.
        intersection_slice_indexes (Collection[Any]):
            Intersection slice indexes for :func:`.FindIntersects.find_intersects`.
        volume_type_slices (Collection[Tuple[slice, ...]]):
            Slices for :func:`.VolumeTypes.volume_types`.
        geometry_array (NDArray[numpy.integer[Any]]):
            Geometry lookup array for :func:`.LookUpGeometry.look_up_geometry`.
        edge_id_direction (NDArray[numpy.unsignedinteger[Any]]):
            Edge ID direction for :func:`.LookUpGeometry.look_up_geometry`.
        edge_id_delta (NDArray[numpy.unsignedinteger[Any]]):
            Edge ID delta for :func:`.LookUpGeometry.look_up_geometry`.
        ambiguity_reaolution (Optional[Callable[[NDArray, NDArray], NDArray["bool"]]]):
            Callable function to modify types in ambiguous sitiations.
            Defaults to None.
    """

    def marching(
        volume: NDArray[numpy.float64],
        level: Union[float, NDArray[numpy.float64]] = 0.0,
        *,
        interpolation: str = "LINEAR",
        step_size: int = 1,
        resolve_ambiguous: bool = True,
    ) -> Tuple[NDArray[numpy.float16], NDArray[numpy.uint64]]:
        """
        Marching Algorithm.

        This is a generalised protocol for performing a marching algorithm
        that can be used for 2D squares, 3D cubes, etc...

        This proceeds according to the following:

        -   Find vertices
                :func:`.FindIntersects.find_intersects`
        -   Calculate volume types
                :func:`.VolumeTypes.volume_types`
        -   Resolve ambiguous cases
                :func:`.ResolveAmbiguous.resolve_ambiguous_types`
        -   Look up geometry
                :func:`.LookUpGeometry.look_up_geometry`

        It returns a tuple of NDArrays: vertices and geometry
        that can be used in graphics software.

        Args
        ----
        volume: NDArray[numpy.float64]
            volume to be evaluated.
            Passed through :func:`numpy.asarray`.
        level: float | NDArray[numpy.float64]
            level to test :attr:`volume` against.
            Passed through :func:`numpy.asarray`.
            Defaults to 0.0.

        Keyword Args
        ------------
        interpolation: str
            see :meth:`FindIntersects.find_intersects`
            Defaults to "LINEAR".
        step_size: int,
            step size that will be used to resample the volume before calculation if > 1.
            Defaults to 1.
        resolve_ambiguous: bool,
            If ambiguity resolution is available will attempt to resolve ambiguity.
            Defaults to True.

        Returns
        -------
        vertices : NDArray[numpy.float16]
            NDArray of coordinates of calculated vertices.
        geometry : NDArray[numpy.uint64]]
            NDArray list of vertex numbers to describe how each vertex is connected.

        N.B. This docstring should be overwritten by the target of the factory
        to include more specific information.
        """

        # check input
        volume = numpy.asarray(volume, dtype=Types.Volume)
        level = numpy.asarray(level, dtype=Types.Volume)
        interpolation = str(interpolation).upper()  # passed to find_intersects
        step_size = int(step_size)
        resolve_ambiguous = bool(resolve_ambiguous)

        # if step_size is more than one then resample the volume
        if step_size > 1:
            volume = volume[tuple([slice(None, None, step_size)] * nD)]

        # check there are at least 2 values in each dimension
        Checking.assert_nd_array(volume, nD, 2)

        # Zero the volume against the level and test the volume
        if level.any():
            volume = volume - level
        volume_test: NDArray[Types.VolumeTest]
        volume_test = (volume >= 0).astype(Types.VolumeTest)

        # Calculate the size multplier for edge_ids
        # this is the cumulative product of the maximum value in each axis
        # multiplied by the number of edges
        size_multiplier: NDArray[Types.SizeMultiplier]
        size_multiplier = numpy.ones(nD, dtype=Types.SizeMultiplier)
        size_multiplier[: (nD - 1)] = numpy.asarray(volume.shape)[::-1].cumprod()[::-1][
            1:
        ]
        size_multiplier *= nEdges

        # Find all vertices where the values cross the zero threshold
        vertices: NDArray[Types.Intersect]
        vertex_ids: NDArray[Types.Intersect_id]
        vertices, vertex_ids = FindIntersects.find_intersects(
            slice_indexes=intersect_slice_indexes,
            volume=volume,
            size_multiplier=size_multiplier,
            volume_test=volume_test,
            interpolation=interpolation,
        )

        # calculate the type of each volume unit according to the value in each corner (given by the slices)
        types: NDArray[Types.VolumeType]
        types = VolumeTypes.volume_types(
            volume_test=volume_test,
            slices=volume_type_slices,
        )

        # resolve ambiguous cases
        if resolve_ambiguous and ambiguity_resolution:
            ambiguity_resolution(types, volume)

        # look up geometry according to square type
        geometry: NDArray[numpy.integer]
        geometry = LookUpGeometry.look_up_geometry(
            types=types,
            geometry_array=geometry_array,
            edge_direction=edge_direction,
            edge_delta=edge_delta,
            size_multiplier=size_multiplier,
        )

        # convert from cupy to numpy for this step
        # CUPYINCLUDE
        # geometry, vertex_ids = geometry.get(), vertex_ids.get()
        # CUPYINCLUDEEND

        # convert geometry indexes from edge_ids to ordered id
        geometry = ConvertIndexes.convert_indexes(geometry, vertex_ids, method="NUMPY")

        return vertices, geometry

    return marching
