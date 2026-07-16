from __future__ import annotations
from typing import Union, Any

"""
A selection of tools for creating slices from strings
imitating the Python interpreter.
"""


def int_or_none(i: Any) -> Union[int, None]:
    """Convert to ``int``, or on ValueError return ``None``.

    ::

        >>> print(int_or_none("3"))
        3
        >>> print(int_or_none(3.2))
        3
        >>> print(int_or_none(""))
        None
        >>> print(int_or_none("3.2"))
        None

    Arguments
    ---------
    i : Any
        input for conversion.

    Returns
    -------

    int
        If the input can be conveted to an ``int``.
    None
        If a ValueError arises during conversion.

    """
    try:
        return int(i)
    except ValueError:
        pass
    return None


def str_to_index(string: str) -> Any:
    """
    Convert a ``str`` representing an index to the approriate object
    in a similar way to the Python interpreter.

    If the string contains a "," then it will be interpreted as a
    multimensional NumPy-type index. It will be converted to a
    ``tuple`` of ``int``, ``slices`` or ``Ellipsis`` using
    a single recursion of this function.

    Otherwise the input wil be interpreted as a standard index.
    Single integer indexes are converted to ``int``.
    "Ellipis" or "..." is converted to ``Ellipsis``.
    If the string contains a ":" then it will be converted to a ``slice``.

    Square brackets are optional.

    ::

        >>> str_to_index("1")
        1
        >>> str_to_index("[-1]")
        -1
        >>> str_to_index("[...]")
        Ellipsis
        >>> str_to_index("Ellipsis")
        Ellipsis
        >>> str_to_index("[1:]")
        slice(1, None, None)
        >>> str_to_index("[:-1]")
        slice(None, -1, None)
        >>> str_to_index("::2")
        slice(None, None, 2)
        >>> str_to_index("[1:, 1:, 1:]")
        (slice(1, None, None), slice(1, None, None), slice(1, None, None))
        >>> str_to_index("[..., 5]")
        (Ellipsis, 5)
        >>> str_to_index("[..., ::5]")
        (Ellipsis, slice(None, None, 5))

    Example to generate the slices for marching cubes :func:`VolumeTypes.volume_types`.

    ::

        list(
            map(
                str_to_index,
                [
                    "[:-1, :-1, :-1]",  # x,   y  , z       1
                    "[1: , :-1, :-1]",  # x+1, y  , z       2
                    "[1: , 1: , :-1]",  # x+1, y+1, z       4
                    "[:-1, 1: , :-1]",  # x,   y+1, z       8
                    "[:-1, :-1, 1: ]",  # x,   y  , z+1    16
                    "[1: , :-1, 1: ]",  # x+1, y  , z+1    32
                    "[1: , 1: , 1: ]",  # x+1, y+1, z+1    64
                    "[:-1, 1: , 1: ]",  # x,   y+1, z+1   128
                ],
            )
        )

    Arguments
    ---------
    string : str
        input string (square brackets are optional).

    Returns
    -------
    tuple[int, Ellipsis, slice]
        if "," in the input then it is interpreted as a multidimensional index.
    int
        if the input can be converted to an ``int``.
    Ellipsis
        if the input is "..." or "Ellipsis".
    slice
        if ":" in the input then it will be converted to a slice.

    """
    # Square brackets are optional
    string = string.replace("[", "").replace("]", "")
    # Single index
    try:
        return int(string)
    except Exception:
        pass
    # Multidimensional slices for e.g. NumPy have a "," in the slice
    if "," in string:
        # One level of recursion [all "," are removed by split]
        return tuple(map(str_to_index, string.split(",")))
    # Return Ellipsis for Ellipsis
    if "..." in string or "Ellipsis" in string:
        return Ellipsis
    # Single dimensional slice
    if ":" in string:
        return slice(*map(int_or_none, string.split(":")))


def str_to_index_tuple(*string: str) -> Any:
    """
    See: :func:`.str_to_index`
    Takes one or more str values
    and returns a tuple of the result
    of mapping :func:`.str_to_index` to each.
    """
    return tuple(map(str_to_index, string))


def _test_examples() -> None:
    # test examples in docstrings

    # int_or_none
    assert int_or_none("3") == 3
    assert int_or_none(3.2) == 3
    assert int_or_none("3.2") is None
    assert int_or_none("") is None

    # str_to_index
    assert str_to_index("1") == 1
    assert str_to_index("[-1]") == -1
    assert str_to_index("[1:]") == slice(1, None, None)
    assert str_to_index("[:-1]") == slice(None, -1, None)
    assert str_to_index("::2") == slice(None, None, 2)
    assert str_to_index("[1:, 1:, 1:]") == (
        slice(1, None, None),
        slice(1, None, None),
        slice(1, None, None),
    )
    assert str_to_index("[..., 5]") == (Ellipsis, 5)
    assert str_to_index("[..., ::5]") == (Ellipsis, slice(None, None, 5))

    print("Test passed")
