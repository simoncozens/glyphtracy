from __future__ import annotations
from typing import Any, Optional
from numpy.typing import NDArray

import numpy


def look_up_geometry(
    types: NDArray[numpy.integer[Any]],
    geometry_array: NDArray[numpy.integer[Any]],
    edge_delta: NDArray[numpy.integer[Any]],
    edge_direction: NDArray[numpy.integer[Any]],
    size_multiplier: NDArray[numpy.integer[Any]],
    nDir: Optional[int] = None,
) -> NDArray[numpy.uint64]:
    """
    Look up geometry from a look up table.

    The value in :attr:`types` is used to lookup sets of `edge number` from :attr:`geometry_array`.
    The `edge number` information is used to lookup values in :attr:`edge_direction` and :attr:`edge_delta`.
    This can be reduced to an array of vertex_id_offsets once the size_multiplier is known.
    The current location is dot with the :attr:`size_multiplier` and added to :attr:`vertex_id_offsets`
    to calculate the `vertex_id`.
    The combined `vertex_ids` make up the geometry.


    Args
    ----

    types : NDArray[numpy.unsignedinteger]
        A NDArray containing values to index the :attr:`geometry_array`.
        Passed through :func:`numpy.asarray`.

    geometry_array : NDArray[numpy.integer]
        The reference array containing the `edge number information` lookup values.

    edge_delta : NDArray[numpy.integer]
        Edge delta coordinate lookup values.

    edge_direction : NDArray[numpy.integer]
        Edge direction lookup values.

    size_multiplier : NDArray[numpy.integer]
        Size multiplier to convert the `edge number` into an `edge id`.

    nDir : int | None
        The number of vertex directions used to calculate the :attr:`edge_id`.
        Defaults to `None` when ``nDir = 1 + edge_id_direction.max()`` will be used.

    Raises
    ------

    ValueError
        If :attr:`geometry_array` has an incompatible shape.

    Returns
    -------

    geometry : NDArray[numpy.uint64]
        A NDArray of connected vertex_ids having shape:
        number of geometry found, number of vertices per geometry.

    """

    # format the number of directions
    nDir = int(nDir if nDir else 1 + edge_direction.max())

    # number of vertices per geometry, e.g. 3 for triangle
    nV: int = edge_delta.shape[1]
    # Check the shape of the geometry table is consistent with the dimensions of the geometry
    if not geometry_array.shape[1] % nV == 0:
        raise ValueError("Bad shape of geometry array.")

    # intialise the return value
    geometry: NDArray[numpy.uint64]
    geometry = numpy.zeros((0, nV), dtype=numpy.uint64)

    # get the geometry lookup values from the supplied types
    geometry_lookups: NDArray[numpy.integer[Any]]
    geometry_lookups = geometry_array[types]

    # The edge information in EDGE_DELTA and EDGE_DIRECTION
    # Can be reduced to a single array once size_multiplier is known
    # vertex_id_offset_lookup = numpy.dot(edge_delta, size_multiplier) + edge_direction
    vertex_id_offset_lookup = (edge_delta * size_multiplier).sum(
        axis=-1
    ) + edge_direction

    # consider each set of vertices from the look up table
    i: int
    for i in range(0, geometry_array.shape[-1], nV):

        # filter the geometry indexes for -1 [null / no geometry]
        geometry_lookups_filter = numpy.nonzero(geometry_lookups[..., i] != -1)

        # if there are no nore matches then there are no more geometry to be found
        if geometry_lookups_filter[0].size == 0:
            break

        # fetch the set of nV edge numbers from the geometry information
        geometry_type_column = geometry_lookups[geometry_lookups_filter][
            ..., i : i + nV
        ].astype(numpy.int16)

        # get the corner coordinates from the filter tuple
        # convert to corner_ids by taking dot product with size_multiplier
        # CUPYIGNORE
        corner_ids = numpy.einsum(
            "ij,i",
            numpy.asarray(geometry_lookups_filter, dtype="uint64"),
            size_multiplier,
        )
        # CUPYIGNOREEND
        # CUPYINCLUDE
        # corner_ids = (numpy.asarray(geometry_lookups_filter, dtype="uint64").transpose() * size_multiplier).sum(axis=-1)
        # CUPYINCLUDEEND

        # add the corner_ids to the looked up vertex_id_offset to get the vertex ids
        geometry_vertex_ids = (
            corner_ids[..., None] + vertex_id_offset_lookup[geometry_type_column]
        )

        # extend the geometry by the current vertex_ids
        geometry = numpy.concatenate((geometry, geometry_vertex_ids), axis=0)

    return geometry
