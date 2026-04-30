"""Structural geometry utilities."""

import numpy as np

from glyphtracy.node import AxisExtremaTag


def span_indices_closed(start: int, end: int, size: int) -> list[int]:
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, size)) + list(range(0, end + 1))


def spans_between_anchors_closed(
    anchor_indices: list[int], size: int
) -> list[list[int]]:
    if size < 2:
        return []
    anchors = sorted(set(anchor_indices))
    if len(anchors) < 2:
        return [list(range(size))]

    spans: list[list[int]] = []
    for i, start in enumerate(anchors):
        end = anchors[(i + 1) % len(anchors)]
        span = span_indices_closed(start, end, size)
        if len(span) >= 2:
            spans.append(span)
    return spans if spans else [list(range(size))]


def dedupe_closed_contour(contour: np.ndarray) -> np.ndarray:
    """Remove trailing point if it duplicates the first point (closed contour)."""
    if contour.shape[0] > 1 and np.allclose(contour[0], contour[-1], atol=1e-6):
        return contour[:-1]
    return contour


def dedupe_open_indices(indices: list[int], size: int, min_index_gap: int) -> list[int]:
    if size < 3:
        return []
    selected: list[int] = []
    for index in sorted(set(indices)):
        if index <= 0 or index >= size - 1:
            continue
        if all(abs(index - existing) >= min_index_gap for existing in selected):
            selected.append(index)
    return selected


def prune_open_split_positions(
    split_positions: list[int], span_size: int, min_span_points: int
) -> list[int]:
    positions = list(sorted(set(split_positions)))
    if not positions:
        return []

    min_span_points = max(2, int(min_span_points))
    while positions:
        starts = [0] + positions
        ends = positions + [span_size - 1]
        span_lengths = [end - start + 1 for start, end in zip(starts, ends)]
        shortest_len = min(span_lengths)
        if shortest_len >= min_span_points:
            break

        shortest_ix = span_lengths.index(shortest_len)
        if shortest_ix == 0:
            positions.pop(0)
        elif shortest_ix == len(span_lengths) - 1:
            positions.pop()
        else:
            positions.pop(shortest_ix)

    return positions


def split_open_span_indices(
    span_indices: list[int], split_positions: list[int]
) -> list[list[int]]:
    if not split_positions:
        return [span_indices]

    fragments: list[list[int]] = []
    positions = sorted(set(split_positions))
    start = 0
    for split_pos in positions:
        fragment = span_indices[start : split_pos + 1]
        if len(fragment) >= 2:
            fragments.append(fragment)
        start = split_pos

    tail_fragment = span_indices[start:]
    if len(tail_fragment) >= 2:
        fragments.append(tail_fragment)

    return fragments if fragments else [span_indices]


def global_axis_extrema_axes(
    contour: np.ndarray,
    short_plateau_max_len: int = 2,
) -> dict[int, set[AxisExtremaTag]]:
    if contour.shape[0] == 0:
        return {}

    def _plateau_representatives(values: np.ndarray, target: float) -> list[int]:
        value_span = float(np.max(values) - np.min(values))
        atol = max(1e-9, value_span * 1e-6)
        matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=atol))
        if matches.size == 0:
            return []

        runs: list[tuple[int, int]] = []
        run_start = int(matches[0])
        run_end = run_start
        for raw_index in matches[1:]:
            index = int(raw_index)
            if index == run_end + 1:
                run_end = index
            else:
                runs.append((run_start, run_end))
                run_start = index
                run_end = index
        runs.append((run_start, run_end))

        representatives: list[int] = []
        for start, end in runs:
            run_len = end - start + 1
            if run_len <= short_plateau_max_len:
                representatives.append((start + end) // 2)
            elif start == end:
                representatives.append(start)
            else:
                representatives.extend([start, end])
        return representatives

    axis_map: dict[int, set[AxisExtremaTag]] = {}
    for axis in range(2):
        axis_values = contour[:, axis]
        min_value = float(np.min(axis_values))
        max_value = float(np.max(axis_values))
        axis_tag: AxisExtremaTag = "y" if axis == 0 else "x"
        reps = _plateau_representatives(axis_values, min_value)
        reps.extend(_plateau_representatives(axis_values, max_value))
        for local_ix in reps:
            axis_map.setdefault(int(local_ix), set()).add(axis_tag)
    return axis_map


def cyclic_distance(index_a: int, index_b: int, size: int) -> int:
    direct = abs(index_a - index_b)
    return min(direct, size - direct)


def dedupe_cyclic_indices(
    indices: list[int], size: int, min_index_gap: int
) -> list[int]:
    if not indices:
        return []
    selected: list[int] = []
    for index in sorted(set(indices)):
        if all(
            cyclic_distance(index, existing, size) >= min_index_gap
            for existing in selected
        ):
            selected.append(index)

    while (
        len(selected) > 1
        and cyclic_distance(selected[0], selected[-1], size) < min_index_gap
    ):
        selected.pop()
    return selected


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / norm


def is_polyline_window_line_like(points: np.ndarray, tolerance: float = 0.25) -> bool:
    if points.shape[0] < 3:
        return True
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 1e-9:
        return True

    unit = chord / chord_len
    distances = []
    for p in points[1:-1]:
        v = p - start
        dist = abs(v[0] * unit[1] - v[1] * unit[0])
        distances.append(dist)

    max_dist = max(distances) if distances else 0.0
    return max_dist <= tolerance


def smooth_cyclic_signal(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < 3:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1

    radius = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def smooth_signal(values: np.ndarray, window: int, *, closed: bool) -> np.ndarray:
    if closed:
        return smooth_cyclic_signal(values, window)
    if window <= 1 or values.shape[0] < 3:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1

    radius = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def axis_prominent_extrema_indices(
    values: np.ndarray,
    radius: int,
    min_prominence: float,
    *,
    closed: bool = True,
) -> list[int]:
    n = values.shape[0]
    if n < 3:
        return list(range(n))

    radius = max(1, min(radius, max(1, n // 4)))
    indices: list[int] = []
    for i in range(n):
        if closed:
            neighbor_ix = [((i + off) % n) for off in range(-radius, radius + 1)]
            local = values[neighbor_ix]
        else:
            lo = max(0, i - radius)
            hi = min(n - 1, i + radius)
            local = values[lo : hi + 1]

        v = values[i]
        local_min = float(np.min(local))
        local_max = float(np.max(local))
        is_local_max = np.all(v >= local)
        is_local_min = np.all(v <= local)
        if is_local_max:
            prominence = v - local_min
        elif is_local_min:
            prominence = local_max - v
        else:
            prominence = 0.0

        if (is_local_max or is_local_min) and prominence >= min_prominence:
            indices.append(i)
    return indices


def is_basically_a_line(
    contour: np.ndarray,
    pixel_tolerance: float = 1.5,
) -> bool:
    """Return True when all interior points lie within `pixel_tolerance` pixels
    of the chord from contour[0] to contour[-1].  The fixed pixel floor means
    rasterization noise (~0.5–1 px) never falsely rejects a straight segment."""
    if contour.shape[0] < 2:
        return True
    start = contour[0]
    end = contour[-1]
    vec = end - start
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return True
    unit_vec = vec / norm

    for i in range(1, contour.shape[0] - 1):
        point_vec = contour[i] - start
        point_dist = abs(point_vec[0] * unit_vec[1] - point_vec[1] * unit_vec[0])
        if point_dist > pixel_tolerance:
            return False
    return True
