from __future__ import annotations
from typing import Tuple, Type, Collection, Any
from numpy.typing import NDArray

import numpy

from . import Checking
from . import Types


def volume_types(
    volume_test: NDArray[numpy.bool_],
    slices: Collection[Tuple[slice, ...]],
    *,
    dtype: Type[numpy.integer[Any]] = Types.VolumeType,
) -> NDArray[Types.VolumeType]:
    r"""
    Calculate the type of each `unit` of a volume based on the values at each corner.

    The returned numpy array :attr:`types` is initalised to a :func:`numpy.zeros` array
    with shape ``n-1`` for each ``n`` in :attr:`volume_test` shape.
    Then, :attr:`types` is updated depending on the values in :attr:`volume_test`
    (where :attr:`volume_test` is the result of an
    operation such as ``volume_test = volume >= level``)
    in the direction stipulated by each of the :attr:`slices`.
    The function iterates through the collection of :attr:`slices` and
    :attr:`types` is updated with the index
    bit (shifted left every iteration of the loop).

    Example :attr:`slices` for standard 3D and 2D volumes (with
    corresponding bit set as the comment).
    See :func:`.IndexingTools.str_to_index` for a tool
    to generate these :attr:`slices`.
    ::

        # Standard slices for a 3D volume
        slices = [
            slice(None, -1), slice(None, -1), slice(None, -1), # 1:   x,   y,   z
            slice( 1, None), slice(None, -1), slice(None, -1), # 2:   x+1, y,   z
            slice( 1, None), slice( 1, None), slice(None, -1), # 4:   x+1, y+1, z
            slice(None, -1), slice( 1, None), slice(None, -1), # 8:   x,   y+1, z
            slice(None, -1), slice(None, -1), slice( 1, None), # 16:  x,   y,   z+1
            slice( 1, None), slice(None, -1), slice( 1, None), # 32:  x+1, y,   z+1
            slice( 1, None), slice( 1, None), slice( 1, None), # 64:  x+1, y+1, z+1
            slice(None, -1), slice( 1, None), slice( 1, None), # 128: x,   y+1, z+1
        ]

        # Standard slices for a 2D volume
        slices = [
            slice(None, -1), slice(None, -1), # 1: x,   y
            slice( 1, None), slice(None, -1), # 2: x+1, y
            slice( 1, None), slice( 1, None), # 4: x+1, y+1
            slice(None, -1), slice( 1, None), # 8: x,   y+1
        ]


    Args
    ----

    volume_test : NDArray[numpy.bool\_]
        A n-dimensional bool-like volume test.
        There must be at least one value in each dimension.
        Passed through :func:`numpy.asarray` with `dtype=numpy.bool_`.

    slices : Collection[Tuple[slice, ...]]
        Collection of tuple of slices that is enumerated to give the type index
        and used to test where :attr:`volume_test` is ``True``.
        See :func:`.IndexingTools.str_to_index` for a tool to generate these :attr:`slices`.

    Keyword Args
    ------------

    dtype : Type[numpy.integer]
        The datatype used in the returned NDArray. Defaults to :const:`Types.VolumeType`.

    Returns
    -------

    types : NDArray[numpy.integer]
        A NDArray containing the results of the tests
        with of size with shape ``n-1`` for ``n`` in each :attr:`volume_test.shape`.

    Raises
    ------

    ValueError
        If the arguments are not appropriate or the shapes of
        the supplied arrays are not consistent.

    TypeError
        If the supplied :attr:`dtype` is too small to hold the
        biggest possible type value.

    """

    # play with stop = :1 instead of 1: for slicing to get cool visual results!

    # check the arguments
    volume_test = numpy.asarray(volume_test, dtype=numpy.bool_)
    # number of dimensions in indexing
    nD: int
    nD = int(len(next(iter(slices))))
    # number of directions to slice in
    nDir: int
    nDir = int(len(slices))
    # check for at least one value in each dimension
    Checking.assert_nd_array(volume_test, nD, 1)

    # intialise the types array to zero with size n-1 vs volume_test
    n: int  # ruff: noqa: F842
    types: NDArray[Types.VolumeType]
    types = numpy.zeros(tuple(n - 1 for n in volume_test.shape), dtype=dtype)

    # check the biggest type value will fit in the supplied dtype
    if 1 << nDir > numpy.iinfo(types.dtype).max:
        raise TypeError(
            f"The largest index {(1<<nDir)-1} will not fit into dtype={dtype.__name__}."
        )

    # Cool but much slower approach
    # numpy.lib.stride_tricks.sliding_window_view(volume_test, [2] * nD)
    # then numpy.packbits ...

    # iterate over the slices
    # the order determines the bit shift, i, of the resulting test
    i: int
    slice_i: Tuple[slice, ...]
    for i, slice_i in enumerate(slices):

        # update the types with a bitwise_or according to
        # the index i of this slice
        # CUPYIGNORE

        numpy.bitwise_or(
            types,
            1 << i,
            out=types,
            where=volume_test[slice_i],
        )

        # CUPYIGNOREEND
        # slower alternative that is compatible with cupy
        # CUPYINCLUDE
        # types[volume_test[slice_i]] |= 1 << i
        # CUPYINCLUDEEND

    return types
