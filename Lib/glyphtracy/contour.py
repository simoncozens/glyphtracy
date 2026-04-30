import math
from dataclasses import dataclass

import numpy as np

from glyphtracy.node import AxisExtremaTag
from glyphtracy.utils import (
    axis_prominent_extrema_indices,
    dedupe_cyclic_indices,
    dedupe_open_indices,
    global_axis_extrema_axes,
    is_basically_a_line,
    normalized,
    prune_open_split_positions,
    smooth_signal,
    span_indices_closed,
    split_open_span_indices,
)


@dataclass
class Contour:
    contour_id: int
    points_rc: np.ndarray
    closed: bool = True

    @property
    def size(self) -> int:
        return int(self.points_rc.shape[0])

    def points_for_indices(self, indices: list[int]) -> np.ndarray:
        return self.points_rc[np.asarray(indices, dtype=np.int64)]

    def slice_before(self, contour_index: int, count: int) -> np.ndarray:
        n = self.size
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        count = max(1, min(count, n))
        if self.closed:
            indices = [((contour_index - i) % n) for i in range(count, 0, -1)]
        else:
            start = max(0, contour_index - count)
            indices = list(range(start, contour_index))
        return self.points_for_indices(indices)

    def slice_after(self, contour_index: int, count: int) -> np.ndarray:
        n = self.size
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        count = max(1, min(count, n))
        if self.closed:
            indices = [((contour_index + i) % n) for i in range(1, count + 1)]
        else:
            end = min(n, contour_index + count + 1)
            indices = list(range(contour_index + 1, end))
        return self.points_for_indices(indices)

    def find_inflection_indices(self, stride: int = 2) -> set[int]:
        n = self.size
        if n < 7:
            return set()

        signs: list[int] = []
        for i in range(n):
            cross = self.turning_cross_sign(i, stride=stride)
            if abs(cross) < 1e-5:
                signs.append(0)
            elif cross > 0:
                signs.append(1)
            else:
                signs.append(-1)

        inflections: set[int] = set()
        for i in range(n):
            prev_sign = signs[(i - 1) % n]
            cur_sign = signs[i]
            if prev_sign == 0 or cur_sign == 0:
                continue
            if prev_sign != cur_sign:
                inflections.add(i)
        return inflections

    def turning_cross_sign(self, i: int, stride: int = 1) -> float:
        n = self.size
        prev_i = (i - stride) % n if self.closed else max(0, i - stride)
        next_i = (i + stride) % n if self.closed else min(n - 1, i + stride)
        p_prev = self.points_rc[prev_i]
        p_cur = self.points_rc[i]
        p_next = self.points_rc[next_i]
        v_in = p_cur - p_prev
        v_out = p_next - p_cur
        return float(v_in[1] * v_out[0] - v_in[0] * v_out[1])

    def corner_angle(self, i: int, stride: int = 1) -> float:
        n = self.size
        if n < 3:
            return 0.0
        prev_i = (i - stride) % n if self.closed else max(0, i - stride)
        next_i = (i + stride) % n if self.closed else min(n - 1, i + stride)
        if prev_i == i or next_i == i:
            return 0.0
        p_prev = self.points_rc[prev_i]
        p_cur = self.points_rc[i]
        p_next = self.points_rc[next_i]
        v_in = normalized(p_cur - p_prev)
        v_out = normalized(p_next - p_cur)
        dot = float(np.dot(v_in, v_out))
        dot = max(-1.0, min(1.0, dot))
        return float(math.acos(dot))

    def iterative_axis_extrema_axes(
        self,
        *,
        min_index_gap: int = 3,
        min_span_points: int = 8,
        max_iterations: int = 8,
    ) -> dict[int, set[AxisExtremaTag]]:
        n = self.size
        if n < 3:
            return {}

        seed_map = global_axis_extrema_axes(
            self.points_rc,
            short_plateau_max_len=max(2, n // 80),
        )
        seed_indices = set(seed_map.keys())
        for axis in range(2):
            smoothed = smooth_signal(self.points_rc[:, axis], window=9, closed=True)
            axis_tag: AxisExtremaTag = "y" if axis == 0 else "x"
            axis_extrema = axis_prominent_extrema_indices(
                smoothed,
                radius=max(2, min(8, n // 30)),
                min_prominence=1.5,
                closed=True,
            )
            seed_indices.update(axis_extrema)
            for idx in axis_extrema:
                seed_map.setdefault(int(idx), set()).add(axis_tag)

        split_indices = dedupe_cyclic_indices(
            list(seed_indices), n, min_index_gap=max(2, min_index_gap)
        )
        discovered: dict[int, set[AxisExtremaTag]] = {
            int(idx): set(seed_map.get(int(idx), set())) for idx in split_indices
        }

        if len(split_indices) < 2:
            spans = [list(range(n))]
        else:
            spans: list[list[int]] = []
            for span_ix, start in enumerate(split_indices):
                end = split_indices[(span_ix + 1) % len(split_indices)]
                span = span_indices_closed(start, end, n)
                if len(span) >= 2:
                    spans.append(span)
            if not spans:
                spans = [list(range(n))]

        min_span_points = max(4, int(min_span_points))
        for _ in range(max_iterations):
            did_split = False
            next_spans: list[list[int]] = []

            for span_indices in spans:
                if len(span_indices) < max(2 * min_span_points, 6):
                    next_spans.append(span_indices)
                    continue

                subcontour = self.points_for_indices(span_indices)
                local_candidates = global_axis_extrema_axes(
                    subcontour,
                    short_plateau_max_len=max(2, min_index_gap),
                )
                for axis in range(2):
                    smoothed = smooth_signal(
                        subcontour[:, axis], window=5, closed=False
                    )
                    axis_tag: AxisExtremaTag = "y" if axis == 0 else "x"
                    axis_extrema = axis_prominent_extrema_indices(
                        smoothed,
                        radius=2,
                        min_prominence=0.75,
                        closed=False,
                    )
                    for local_ix in axis_extrema:
                        local_candidates.setdefault(int(local_ix), set()).add(axis_tag)

                split_positions = dedupe_open_indices(
                    list(local_candidates.keys()),
                    len(span_indices),
                    min_index_gap=max(2, min_index_gap - 1),
                )
                split_positions = prune_open_split_positions(
                    split_positions, len(span_indices), min_span_points=min_span_points
                )

                if not split_positions:
                    next_spans.append(span_indices)
                    continue

                did_split = True
                for local_ix in split_positions:
                    global_ix = span_indices[local_ix]
                    discovered.setdefault(int(global_ix), set()).update(
                        local_candidates.get(int(local_ix), set())
                    )
                next_spans.extend(
                    split_open_span_indices(span_indices, split_positions)
                )

            spans = next_spans
            if not did_split:
                break

        return discovered

    def iterative_axis_extrema_axes_on_open_span(
        self,
        span_indices: list[int],
        *,
        min_index_gap: int = 3,
        min_span_points: int = 8,
        max_iterations: int = 8,
    ) -> dict[int, set[AxisExtremaTag]]:
        if len(span_indices) < 3:
            return {}

        # Sharp-corner-bounded spans that are effectively straight should not
        # receive interior extrema nodes due to raster noise.
        span_points = self.points_for_indices(span_indices)
        if is_basically_a_line(span_points, pixel_tolerance=1.5):
            return {}

        local_all = list(range(len(span_indices)))
        discovered: dict[int, set[AxisExtremaTag]] = {}
        spans_local: list[list[int]] = [local_all]
        min_span_points = max(4, int(min_span_points))

        for _ in range(max_iterations):
            did_split = False
            next_local_spans: list[list[int]] = []

            for local_span in spans_local:
                if len(local_span) < max(2 * min_span_points, 6):
                    next_local_spans.append(local_span)
                    continue

                global_span = [span_indices[ix] for ix in local_span]
                subcontour = self.points_for_indices(global_span)

                local_candidates = global_axis_extrema_axes(
                    subcontour,
                    short_plateau_max_len=max(2, min_index_gap),
                )
                for axis in range(2):
                    smoothed = smooth_signal(
                        subcontour[:, axis], window=5, closed=False
                    )
                    axis_tag: AxisExtremaTag = "y" if axis == 0 else "x"
                    axis_extrema = axis_prominent_extrema_indices(
                        smoothed,
                        radius=2,
                        min_prominence=0.75,
                        closed=False,
                    )
                    for local_ix in axis_extrema:
                        local_candidates.setdefault(int(local_ix), set()).add(axis_tag)

                split_positions = dedupe_open_indices(
                    list(local_candidates.keys()),
                    len(local_span),
                    min_index_gap=max(2, min_index_gap - 1),
                )
                split_positions = prune_open_split_positions(
                    split_positions, len(local_span), min_span_points=min_span_points
                )

                if not split_positions:
                    next_local_spans.append(local_span)
                    continue

                did_split = True
                for local_ix in split_positions:
                    global_ix = span_indices[local_span[local_ix]]
                    discovered.setdefault(int(global_ix), set()).update(
                        local_candidates.get(int(local_ix), set())
                    )
                next_local_spans.extend(
                    split_open_span_indices(local_span, split_positions)
                )

            spans_local = next_local_spans
            if not did_split:
                break

        return discovered
