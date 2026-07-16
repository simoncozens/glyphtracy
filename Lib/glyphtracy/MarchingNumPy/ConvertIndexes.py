from __future__ import annotations
from typing import Dict, Collection, Type, Any, Optional
from numpy.typing import NDArray
import numpy


def convert_indexes(
    array: Collection[Any],
    ordered_ids: Collection[Any],
    *,
    method: str = "NUMPY",
    **kwargs: Any,
) -> Any:
    """
    Replace values in :attr:`array` with the index of
    those values within an :attr:`ordered_ids` array.

    Args
    ----
    array : ArrayLike
        array.
        Passed through :func:`numpy.asarray`.
    ordered_ids : ArrayLike
        ordered array.
        Passed through :func:`numpy.asarray`.
    method : {"NUMPY", "DICT"}
        use a numpy array for the lookup
        or use a dict for the lookup.

    Returns
    -------
    NDArray
        convered array of values after looukp.
    """

    # check the arguments
    array = numpy.asarray(array)
    ordered_ids = numpy.asarray(ordered_ids)

    # no point generating the dict for an empty array
    if array.size == 0:
        return array

    # generate the dict based upon the ordered list of ids

    method = method.upper()

    if method == "NUMPY":
        return ndarray_numpy_ordered_lookup(
            array=array,
            ordered_ids=ordered_ids,
        )
    return ndarray_dict_ordered_lookup(
        array=array,
        ordered_ids=ordered_ids,
        dtype=numpy.uint64,
    )


def ndarray_dict_ordered_lookup(
    array: Collection[Any],
    ordered_ids: Collection[Any],
    *,
    dtype: Optional[Type[numpy.integer[Any]]] = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """
    Replace values in an **array** values with the index of
    those values within an **ordered_ids** array using :meth:`.ndarray_dict_lookup`.

    Args:
        array (NDArray): array.
        ordered_ids (NDArray): lookup values in order.

    For keyword parameters, see :func:`.ndarray_dict_lookup`.

    Returns:
        NDArray: results of the array lookup.

    """
    return ndarray_dict_lookup(
        array,
        {ordered_id: i for i, ordered_id in enumerate(ordered_ids)},
        dtype=dtype,
        **kwargs,
    )


def ndarray_dict_lookup(
    array: Collection[Any],
    dictionary: Dict[Any, Any],
    *,
    dtype: Optional[Type[numpy.integer[Any]]] = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """
    Vectorized lookup of **array** values within a ``dict``.

    Args:
        array (NDArray): array
        dictionary (dict): lookup values.
    Keyword Args:
        dtype (Type, optional): dtype of the results. Defaults to None.
        default (Any, optional): the default value for :func:`dictionary.get`.

    Returns:
        NDArray: results of the dictionary lookup.

    Raises:
        KeyError:
            If a value in :attr:`array` is not in :attr:`ordered_ids`
            and :attr:`default` is not supplied.

    """
    # If default is provided then missing ids will return a default values
    # N.B. fetch default from kwargs so it can be None, etc.
    if "default" in kwargs:
        return numpy.asarray(
            numpy.vectorize(dictionary.get)(array, kwargs["default"]), dtype=dtype
        )
    # Otherwise missing ids will raise a KeyError
    return numpy.asarray(numpy.vectorize(dictionary.get)(array), dtype=dtype)


def ndarray_numpy_ordered_lookup(
    array: NDArray[Any],
    ordered_ids: NDArray[Any],
    *,
    dtype: Optional[Type[Any]] = None,
) -> Any:
    """
    Replace values in an **array** values with the index of
    those values within an **ordered_ids** array using :module:`numpy`.

    N.B. Creates a *large* NDArray for the lookup!

    Args:
        array (NDArray): array.
        ordered_ids (NDArray): lookup values in order.

    Keyword Args:
        dtype (Type, optional): dtype of the results. Defaults to None.

    Returns:
        NDArray: results of the array lookup.

    Raises:
        ValueError:
            If the array of values is too large for a numpy.uint64.

    """

    if dtype is None:
        for dtype in [numpy.uint32, numpy.uint64]:
            if len(ordered_ids) < numpy.iinfo(dtype).max:  # type: ignore
                break
        else:
            raise ValueError("Index size error")

    numpy_lookup = numpy.zeros(int(ordered_ids.max()) + 1, dtype=dtype)
    for i, ordered_id in enumerate(ordered_ids):
        numpy_lookup[ordered_id] = i
    return numpy_lookup[array]
