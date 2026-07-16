from __future__ import annotations
from typing import Tuple, Collection, Set, Any, Optional
from numpy.typing import NDArray, ArrayLike

import numpy

from .Types import Intersect, Intersect_id

INTERPOLATION_VALUES: Set[str] = {"LINEAR", "HALFWAY", "COSINE"}


def find_intersects(
    slice_indexes: Any,
    volume: NDArray[Any],
    size_multiplier: NDArray[Any],
    volume_test: Optional[NDArray[numpy.bool_]] = None,
    level: ArrayLike = 0.0,
    interpolation: str = "LINEAR",
) -> Tuple[NDArray[Intersect], NDArray[Intersect_id]]:
    r"""Calculate intersects in volume data.

    This function considers values in a **volume** along slices
    as determined by **slice_indexes**.
    Each intersect where the **volume** crosses the **level**
    is recorded in the **intersects** NDArray.
    Each intersect has an associated unique id that
    is recorded in the **intersect_ids** NDArray.

    The **level** is used to generate a
    zero-referenced volume and volume_test as below.
    If **level** is non-truthy (i.e. `0.0` or `None`) then
    **volume** is assumed to be zero-referenced.
    If **volume_test** is already calculated then this can be passed in
    and, as long as **level** is non-truthy,
    this part of the calculation will be skipped. ::

        volume = volume - level
        volume_test = volume >= 0

    The crossing point is calculated according to value of **interpolation**:

    "HALFWAY"
        The crossing point is marked at 0.5 between the two corners.
        This is a fast interpolation but gives geometry with a block appearence.

    "LINEAR"
        The crossing point is interpolated using linear interpolation.

        .. math::
            \begin{align*}
            y &= mx + c \\
            y &= (v_{n+1}-v_{n})x + v_{n} \\
            x &= \frac{y - v_{n}}{v_{n+1} - v_{n}}
            \end{align*}

        At the zeroed volume crossing point, :math:`y = 0`

        .. math::
            x = \frac{v_{n}}{v_{n} - v_{n+1}}

    "COSINE"
        The crossing point is interpolated using cosine interpolation.

        .. math::
            \begin{align*}
            y &= \frac{h}{2}\cos{(\pi x)} + c \\
            y &= \frac{v_{n}-v_{n+1}}{2} \cos({\pi x}) + \frac{v_{n} + v_{n+1}}{2} \\
            x &= \frac{1}{\pi}\arccos\left({\frac{2y - v_{n} - v_{n+1}}{v_{n}-v_{n+1}}}\right)
            \end{align*}

        At the zeroed volume crossing point, :math:`y = 0`

        .. math::
            x = \frac{1}{\pi}\arccos\left({\frac{v_{n+1} + v_{n}}{v_{n+1}-v_{n}}}\right)

    Parameters
    ----------
    slice_indexes : Collection[Tuple[slice, ...], ...]
        A Collection of Tuple of Slices that describe where
        to look for the intersects.

    volume : ArrayLike
        A n-dimensional array containing the volume data for testing.
        Passed through :func:`numpy.asarray`.

    size_multiplier : NDArray[Any]
        A multiplier to convert a coordinate and direction into a unique id.

    volume_test : NDArray[numpy.bool\_], default = None
        Optional bool-like results of the results of testing the volume against the :attr:`level`.
        The size must match the size of :attr:`volume`.

    level : ArrayLike, default = 0.0
        The level against which volume is evaluated.
        Passed through :func:`numpy.asarray` if truthy.

    interpolation : {"LINEAR", "HALFWAY", "COSINE"}
        Interpolation of the crossing point.

    Returns
    -------
        intersects : NDArray[Intersect]
            A NDArray containing intersects, ordered by axis.
        intersect_ids : NDArray[Intersect_id]
            A NDArray containing a unique id for each intersect in same order as intersects for indexing.


    Raises
    ------
    ValueError
        If the :attr:`volume` is zero-sized; or
        if the shape of the supplied :attr:`volume_test` is not consistent with :attr:`volume`; or
        if :attr:`interpolation` is an invalid value.

    """

    # check the arguments
    volume = numpy.asarray(volume)
    if not volume.size > 0:
        raise ValueError("There must be at least one value is each supplied dimension.")
    if level:
        volume = volume - numpy.asarray(level)
    if volume_test is None or level:
        volume_test = (volume >= 0.0).astype(numpy.bool_)
    else:
        volume_test = numpy.asarray(volume_test, dtype=numpy.bool_)
        if not volume_test.shape == volume.shape:
            raise ValueError("The volume_test shape must match the volume shape.")
    interpolation = str(interpolation).upper()
    if interpolation not in INTERPOLATION_VALUES:
        raise ValueError(
            f'interpolation must be one of: {"; ".join(INTERPOLATION_VALUES)}'
        )

    nD: int = volume.ndim  # number of dimensions

    # initialise intersects and their ids
    intersects: NDArray[Intersect]
    intersects = numpy.zeros((0, nD), dtype=Intersect)
    intersect_ids: NDArray[Intersect_id]
    intersect_ids = numpy.zeros((0,), dtype=Intersect_id)

    i: int
    n_slices: Tuple[slice, ...]
    nplus1_slices: Tuple[slice, ...]
    for i, (n_slices, nplus1_slices) in enumerate(slice_indexes):

        # compare each volume_test value with the next value along the vector
        value_filter: Tuple[NDArray[numpy.int64], ...]  # Type from numpy.nonzero
        value_filter = numpy.nonzero(
            volume_test[n_slices] != volume_test[nplus1_slices]
        )

        # convert the filter into indices where there are crossing points
        intersect_indices: NDArray[Intersect_id]
        intersect_indices = numpy.asarray(value_filter, dtype=Intersect_id).transpose()

        # initialise the intersects to the corner of the cube
        intersects_along_axis: NDArray[Intersect]
        intersects_along_axis = intersect_indices.astype(Intersect)

        # if there are no intersects we can continue
        if intersects_along_axis.size == 0:
            continue

        # do the interpolation
        # TODO: Transfer interpolation  to indvidual functions
        # TODO: Cubic interpolation - more challenging because need nminus1 and nplus2 values
        interpolated_offset: NDArray[Intersect]
        if interpolation == "HALFWAY":
            # assign the crossing point half way along the edge
            interpolated_offset = numpy.asarray(0.5, dtype=Intersect)

        else:
            # Get the volume values for interpolation
            n_values: NDArray[Any]
            nplus1_values: NDArray[Any]
            n_values = volume[n_slices][value_filter]
            nplus1_values = volume[nplus1_slices][value_filter]

            if interpolation == "LINEAR":
                # linear interpolate the distance along the axis of the crossing point
                interpolated_offset = (n_values / (n_values - nplus1_values)).astype(
                    Intersect
                )

            if interpolation == "COSINE":
                # cosine interpolate the distance along the axis of the crossing point
                interpolated_offset = (
                    numpy.arccos(
                        (nplus1_values + n_values) / (nplus1_values - n_values)
                    )
                    / numpy.pi
                ).astype(Intersect)

        # calculate a vector based upon the slice directions
        slice_vector: NDArray[numpy.integer[Any]]
        slice_vector = vector_from_slices(n_slices, nplus1_slices, absolute=True)

        # fill out the interpolated value
        # double transpose for broadcasting!
        interpolated_offset = (
            interpolated_offset
            * numpy.full_like(
                intersect_indices,
                slice_vector,
                dtype=Intersect,
            ).transpose()
        ).transpose()
        # numpy.einsum is tidier but a little slower and horroble for cupy
        # numpy.einsum("i,j->ij", interpolated_offset, slice_vector)

        # add the appropriate amount according to interpolation
        intersects_along_axis += interpolated_offset

        # add any results to the complete list
        # extend the list of intersects with the ones from this axis
        intersects = numpy.concatenate((intersects, intersects_along_axis), axis=0)

        # convert the intersect_indices into intersect_ids and append to list
        intersect_ids = numpy.concatenate(
            # CUPYIGNORE
            (
                intersect_ids,
                numpy.einsum("ij,j->i", intersect_indices, size_multiplier) + i,
            ),
            # CUPYIGNOREEND
            # CUPYINCLUDE
            # (intersect_ids, (intersect_indices * size_multiplier).sum(axis=1) + i),
            # CUPYINCLUDEEND
            # (intersect_ids, numpy.dot(intersect_indices, size_multiplier) + i),
            axis=0,
        )

        # einsum is faster than equivalent .sum(axis=1)
        # multiply and .sum is faster than numpy.dot

    return intersects, intersect_ids


def vector_from_slices(
    from_slices: Collection[slice],
    to_slices: Collection[slice],
    absolute: bool = False,
):
    """
    Calculates a direction vector based
    upon two Collections of slices.

    Args:
        from_slices (Collection[slice]): from Collection of slices.
        to_slices (Collection[slice]): to Collection of slices.
        absolute (bool): If ```True``` returns absolute values. Default False.

    Returns:
        NDArray[numpy.int8]: vector representing direction.
    """
    if absolute:
        return numpy.asarray(
            [
                abs(int(to_slice.start or 0) - int(from_slice.start or 0))
                for from_slice, to_slice in zip(from_slices, to_slices)
            ],
            dtype=numpy.int8,
        )
    return numpy.asarray(
        [
            int(to_slice.start or 0) - int(from_slice.start or 0)
            for from_slice, to_slice in zip(from_slices, to_slices)
        ],
        dtype=numpy.int8,
    )
