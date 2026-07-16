from __future__ import annotations
from numpy.typing import ArrayLike

import numpy


def assert_nd_array(a: ArrayLike, nD: int, minsize: int = 0) -> None:
    """
    Check the numpy array **a** has exactly **nD** dimensions
    and the shape in each dimension is greater than **minsize**.

    Args
    ----

    a: ArrayLike
        A numpy ndarray. Passed through :func:`numpy.asarray`.

    nD: int
        The number of dimensions to test for.

    minsize: int, default = 0
        The minimum required size in each dimension.

    Raises
    ------

    ValueError
        If any of the assertions fail.

    """

    nD = int(nD)
    minsize = int(minsize)
    a = numpy.asarray(a)

    n: int
    try:
        assert a.ndim == nD
        for n in range(nD):
            assert a.shape[n] >= minsize
    except AssertionError:
        raise ValueError(
            "A {}D array with at least {} elements in each direction must be supplied as the original volume.".format(
                nD, minsize
            )
        )
