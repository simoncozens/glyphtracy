from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
try:
    from kurbopy import BezPath, CubicBez, Point
except ImportError:
    from glyphtracy.minikurbo import BezPath, CubicBez, Point


@dataclass
class _FittedSegment:
    cubic: CubicBez
    max_error: float


def _as_xy(point_rc: np.ndarray) -> np.ndarray:
    return np.array([float(point_rc[1]), float(point_rc[0])], dtype=np.float64)


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _unit(vec: np.ndarray) -> np.ndarray:
    n = _norm(vec)
    if n <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / n


def _sign_with_default(value: float, default: float = 1.0) -> float:
    if abs(value) <= 1e-9:
        return float(default)
    return 1.0 if value > 0.0 else -1.0


def _constrain_tangent_for_axis_extrema(
    tangent_xy: np.ndarray,
    axis_extrema: set[str],
) -> np.ndarray:
    if not axis_extrema:
        return _unit(tangent_xy)

    # Contour points are in xy here:
    # - X-axis extrema => x is extremal => tangent must be vertical.
    # - Y-axis extrema => y is extremal => tangent must be horizontal.
    if "x" in axis_extrema and "y" not in axis_extrema:
        return np.array(
            [0.0, _sign_with_default(float(tangent_xy[1]))], dtype=np.float64
        )
    if "y" in axis_extrema and "x" not in axis_extrema:
        return np.array(
            [_sign_with_default(float(tangent_xy[0])), 0.0], dtype=np.float64
        )

    return _unit(tangent_xy)


def _chord_t_values(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return np.array([0.0, 1.0], dtype=np.float64)
    diffs = points_xy[1:] - points_xy[:-1]
    seg = np.linalg.norm(diffs, axis=1)
    total = float(seg.sum())
    if total <= 1e-9:
        return np.linspace(0.0, 1.0, points_xy.shape[0], dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)], axis=0)
    return (cumulative / total).astype(np.float64)


def _fit_single_cubic_with_tangents(
    points_xy: np.ndarray,
    tan_start: np.ndarray,
    tan_end: np.ndarray,
    *,
    segment_balance_weight: float,
    handle_shrink_weight: float,
) -> CubicBez:
    p0 = points_xy[0]
    p3 = points_xy[-1]
    t0 = _unit(tan_start)
    t3 = _unit(tan_end)
    t_values = _chord_t_values(points_xy)

    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # Unknowns are handle lengths alpha (start) and beta (end):
    # p1 = p0 + alpha * t0
    # p2 = p3 - beta  * t3
    for ti, q in zip(t_values[1:-1], points_xy[1:-1]):
        mt = 1.0 - ti
        b1 = 3.0 * (mt**2) * ti
        b2 = 3.0 * mt * (ti**2)
        const = (mt**3 + b1) * p0 + (ti**3 + b2) * p3
        rhs = q - const

        row_x = np.array([b1 * t0[0], -b2 * t3[0]], dtype=np.float64)
        row_y = np.array([b1 * t0[1], -b2 * t3[1]], dtype=np.float64)
        a_rows.append(row_x)
        b_rows.append(float(rhs[0]))
        a_rows.append(row_y)
        b_rows.append(float(rhs[1]))

    if segment_balance_weight > 0.0:
        w = float(np.sqrt(segment_balance_weight))
        a_rows.append(np.array([w, -w], dtype=np.float64))
        b_rows.append(0.0)

    if handle_shrink_weight > 0.0:
        w = float(np.sqrt(handle_shrink_weight))
        a_rows.append(np.array([w, 0.0], dtype=np.float64))
        b_rows.append(0.0)
        a_rows.append(np.array([0.0, w], dtype=np.float64))
        b_rows.append(0.0)

    if not a_rows:
        alpha = 0.0
        beta = 0.0
    else:
        a = np.vstack(a_rows)
        b = np.asarray(b_rows, dtype=np.float64)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        alpha = max(0.0, float(x[0]))
        beta = max(0.0, float(x[1]))

    chord = _norm(p3 - p0)
    max_len = max(1.0, 3.0 * chord)
    alpha = min(alpha, max_len)
    beta = min(beta, max_len)

    c1 = p0 + alpha * t0
    c2 = p3 - beta * t3
    return CubicBez(
        Point(float(p0[0]), float(p0[1])),
        Point(float(c1[0]), float(c1[1])),
        Point(float(c2[0]), float(c2[1])),
        Point(float(p3[0]), float(p3[1])),
    )


def _fit_single_cubic_with_partial_constraints(
    points_xy: np.ndarray,
    tan_start: np.ndarray,
    tan_end: np.ndarray,
    *,
    free_start_handle: bool,
    free_end_handle: bool,
    segment_balance_weight: float,
    handle_shrink_weight: float,
) -> CubicBez:
    p0 = points_xy[0]
    p3 = points_xy[-1]
    t0 = _unit(tan_start)
    t3 = _unit(tan_end)
    t_values = _chord_t_values(points_xy)

    next_var = 0
    i_alpha = None
    i_beta = None
    i_c1x = None
    i_c1y = None
    i_c2x = None
    i_c2y = None

    if free_start_handle:
        i_c1x = next_var
        i_c1y = next_var + 1
        next_var += 2
    else:
        i_alpha = next_var
        next_var += 1

    if free_end_handle:
        i_c2x = next_var
        i_c2y = next_var + 1
        next_var += 2
    else:
        i_beta = next_var
        next_var += 1

    n_vars = next_var
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    for ti, q in zip(t_values[1:-1], points_xy[1:-1]):
        mt = 1.0 - ti
        b0 = mt**3
        b1 = 3.0 * (mt**2) * ti
        b2 = 3.0 * mt * (ti**2)
        b3 = ti**3
        # Include endpoint contributions from constrained handles:
        # c1 = p0 + alpha*t0  =>  b1*c1 = b1*p0 + b1*alpha*t0  =>  b1*p0 is constant
        # c2 = p3 - beta *t3  =>  b2*c2 = b2*p3 - b2*beta *t3  =>  b2*p3 is constant
        const = b0 * p0 + b3 * p3
        if not free_start_handle:
            const = const + b1 * p0
        if not free_end_handle:
            const = const + b2 * p3
        rhs = q - const

        row_x = np.zeros(n_vars, dtype=np.float64)
        row_y = np.zeros(n_vars, dtype=np.float64)

        if free_start_handle:
            row_x[i_c1x] = b1
            row_y[i_c1y] = b1
        else:
            row_x[i_alpha] = b1 * t0[0]
            row_y[i_alpha] = b1 * t0[1]

        if free_end_handle:
            row_x[i_c2x] = b2
            row_y[i_c2y] = b2
        else:
            row_x[i_beta] = -b2 * t3[0]
            row_y[i_beta] = -b2 * t3[1]

        a_rows.append(row_x)
        b_rows.append(float(rhs[0]))
        a_rows.append(row_y)
        b_rows.append(float(rhs[1]))

    if (
        segment_balance_weight > 0.0
        and (not free_start_handle)
        and (not free_end_handle)
    ):
        w = float(np.sqrt(segment_balance_weight))
        row = np.zeros(n_vars, dtype=np.float64)
        row[i_alpha] = w
        row[i_beta] = -w
        a_rows.append(row)
        b_rows.append(0.0)

    if handle_shrink_weight > 0.0:
        w = float(np.sqrt(handle_shrink_weight))
        if free_start_handle:
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_c1x] = w
            a_rows.append(row)
            b_rows.append(w * float(p0[0]))
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_c1y] = w
            a_rows.append(row)
            b_rows.append(w * float(p0[1]))
        else:
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_alpha] = w
            a_rows.append(row)
            b_rows.append(0.0)

        if free_end_handle:
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_c2x] = w
            a_rows.append(row)
            b_rows.append(w * float(p3[0]))
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_c2y] = w
            a_rows.append(row)
            b_rows.append(w * float(p3[1]))
        else:
            row = np.zeros(n_vars, dtype=np.float64)
            row[i_beta] = w
            a_rows.append(row)
            b_rows.append(0.0)

    if not a_rows:
        c1 = p0.copy()
        c2 = p3.copy()
    else:
        a = np.vstack(a_rows)
        b = np.asarray(b_rows, dtype=np.float64)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)

        if free_start_handle:
            c1 = np.array([float(x[i_c1x]), float(x[i_c1y])], dtype=np.float64)
        else:
            alpha = max(0.0, float(x[i_alpha]))
            c1 = p0 + alpha * t0

        if free_end_handle:
            c2 = np.array([float(x[i_c2x]), float(x[i_c2y])], dtype=np.float64)
        else:
            beta = max(0.0, float(x[i_beta]))
            c2 = p3 - beta * t3

    return CubicBez(
        Point(float(p0[0]), float(p0[1])),
        Point(float(c1[0]), float(c1[1])),
        Point(float(c2[0]), float(c2[1])),
        Point(float(p3[0]), float(p3[1])),
    )


def _fit_single_cubic_unconstrained(
    points_xy: np.ndarray,
    *,
    handle_shrink_weight: float,
) -> CubicBez:
    return _fit_single_cubic_with_partial_constraints(
        points_xy,
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        free_start_handle=True,
        free_end_handle=True,
        segment_balance_weight=0.0,
        handle_shrink_weight=handle_shrink_weight,
    )


def _segment_max_error(cubic: CubicBez, points_xy: np.ndarray) -> tuple[float, int]:
    t_values = _chord_t_values(points_xy)
    max_error = 0.0
    max_index = 0
    for i, (ti, q) in enumerate(zip(t_values, points_xy)):
        p: Point = cubic.eval(float(ti))
        err = p.distance(Point(q[0], q[1]))
        if err > max_error:
            max_error = err
            max_index = i
    return max_error, max_index


def _split_and_refit(
    points_xy: np.ndarray,
    tan_start: np.ndarray,
    tan_end: np.ndarray,
    *,
    tolerance: float,
    max_depth: int,
    segment_balance_weight: float,
    handle_shrink_weight: float,
    unconstrained_handles: bool = False,
    free_start_handle: bool = False,
    free_end_handle: bool = False,
) -> list[_FittedSegment]:
    if unconstrained_handles:
        cubic = _fit_single_cubic_unconstrained(
            points_xy,
            handle_shrink_weight=handle_shrink_weight,
        )
    elif free_start_handle or free_end_handle:
        cubic = _fit_single_cubic_with_partial_constraints(
            points_xy,
            tan_start,
            tan_end,
            free_start_handle=free_start_handle,
            free_end_handle=free_end_handle,
            segment_balance_weight=segment_balance_weight,
            handle_shrink_weight=handle_shrink_weight,
        )
    else:
        cubic = _fit_single_cubic_with_tangents(
            points_xy,
            tan_start,
            tan_end,
            segment_balance_weight=segment_balance_weight,
            handle_shrink_weight=handle_shrink_weight,
        )
    max_error, split_ix = _segment_max_error(cubic, points_xy)

    if max_error <= tolerance or max_depth <= 0 or points_xy.shape[0] < 8:
        return [_FittedSegment(cubic=cubic, max_error=max_error)]

    split_ix = max(2, min(int(split_ix), points_xy.shape[0] - 3))
    left = points_xy[: split_ix + 1]
    right = points_xy[split_ix:]
    split_tangent = _unit(points_xy[split_ix + 1] - points_xy[split_ix - 1])

    left_fitted = _split_and_refit(
        left,
        tan_start,
        split_tangent,
        tolerance=tolerance,
        max_depth=max_depth - 1,
        segment_balance_weight=segment_balance_weight,
        handle_shrink_weight=handle_shrink_weight,
        unconstrained_handles=unconstrained_handles,
        free_start_handle=free_start_handle,
        free_end_handle=False,
    )
    right_fitted = _split_and_refit(
        right,
        split_tangent,
        tan_end,
        tolerance=tolerance,
        max_depth=max_depth - 1,
        segment_balance_weight=segment_balance_weight,
        handle_shrink_weight=handle_shrink_weight,
        unconstrained_handles=unconstrained_handles,
        free_start_handle=False,
        free_end_handle=free_end_handle,
    )
    return left_fitted + right_fitted


def _global_handle_lengths(
    node_points_xy: list[np.ndarray],
    node_tangents: list[np.ndarray],
    segment_points_xy: list[np.ndarray],
    continuities: list[str | None],
    *,
    node_balance_weight: float,
    segment_balance_weight: float,
    g2_weight: float,
    handle_shrink_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(node_points_xy)
    if n_nodes < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Unknown vector [out_0..out_n-1, in_0..in_n-1]
    n_vars = 2 * n_nodes
    out_off = 0
    in_off = n_nodes

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for si, points_xy in enumerate(segment_points_xy):
        if points_xy.shape[0] < 3:
            continue
        p0 = node_points_xy[si]
        p3 = node_points_xy[si + 1]
        t0 = node_tangents[si]
        t3 = node_tangents[si + 1]
        t_values = _chord_t_values(points_xy)

        for ti, q in zip(t_values[1:-1], points_xy[1:-1]):
            mt = 1.0 - ti
            b1 = 3.0 * (mt**2) * ti
            b2 = 3.0 * mt * (ti**2)
            const = (mt**3 + b1) * p0 + (ti**3 + b2) * p3
            target = q - const

            row_x = np.zeros(n_vars, dtype=np.float64)
            row_y = np.zeros(n_vars, dtype=np.float64)
            row_x[out_off + si] = b1 * t0[0]
            row_y[out_off + si] = b1 * t0[1]
            row_x[in_off + si + 1] = -b2 * t3[0]
            row_y[in_off + si + 1] = -b2 * t3[1]
            rows.append(row_x)
            rhs.append(float(target[0]))
            rows.append(row_y)
            rhs.append(float(target[1]))

    if segment_balance_weight > 0.0:
        w = float(np.sqrt(segment_balance_weight))
        for si in range(n_nodes - 1):
            row = np.zeros(n_vars, dtype=np.float64)
            row[out_off + si] = w
            row[in_off + si + 1] = -w
            rows.append(row)
            rhs.append(0.0)

    if node_balance_weight > 0.0:
        w = float(np.sqrt(node_balance_weight))
        for ni in range(1, n_nodes - 1):
            row = np.zeros(n_vars, dtype=np.float64)
            row[in_off + ni] = w
            row[out_off + ni] = -w
            rows.append(row)
            rhs.append(0.0)

    if g2_weight > 0.0:
        w = float(np.sqrt(g2_weight))
        for ni in range(1, n_nodes - 1):
            if continuities[ni] != "G2":
                continue
            row = np.zeros(n_vars, dtype=np.float64)
            row[in_off + ni] = w
            row[out_off + ni] = -w
            rows.append(row)
            rhs.append(0.0)

    if handle_shrink_weight > 0.0:
        w = float(np.sqrt(handle_shrink_weight))
        for vi in range(n_vars):
            row = np.zeros(n_vars, dtype=np.float64)
            row[vi] = w
            rows.append(row)
            rhs.append(0.0)

    if not rows:
        out_len = np.zeros(n_nodes, dtype=np.float64)
        in_len = np.zeros(n_nodes, dtype=np.float64)
        return out_len, in_len

    a = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    out_len = np.maximum(0.0, x[out_off : out_off + n_nodes])
    in_len = np.maximum(0.0, x[in_off : in_off + n_nodes])

    for si in range(n_nodes - 1):
        chord = _norm(node_points_xy[si + 1] - node_points_xy[si])
        max_len = max(1.0, 3.0 * chord)
        out_len[si] = min(out_len[si], max_len)
        in_len[si + 1] = min(in_len[si + 1], max_len)

    return out_len, in_len


def _node_tangents_from_run(
    node_points_xy: list[np.ndarray],
    continuities: list[str | None],
    axis_extrema_by_node: list[set[str]],
) -> list[np.ndarray]:
    n = len(node_points_xy)
    tangents: list[np.ndarray] = []
    for i in range(n):
        if i == 0:
            tangents.append(_unit(node_points_xy[1] - node_points_xy[0]))
            continue
        if i == n - 1:
            tangents.append(_unit(node_points_xy[-1] - node_points_xy[-2]))
            continue

        prev_vec = _unit(node_points_xy[i] - node_points_xy[i - 1])
        next_vec = _unit(node_points_xy[i + 1] - node_points_xy[i])
        continuity = continuities[i]
        if continuity in {"G1", "G2"}:
            base_tangent = _unit(prev_vec + next_vec)
        else:
            base_tangent = next_vec
        tangents.append(base_tangent)

    return [
        _constrain_tangent_for_axis_extrema(tangent, axis_extrema_by_node[i])
        for i, tangent in enumerate(tangents)
    ]


def fit_curve_run(
    contour_points_rc: np.ndarray,
    run_nodes: Sequence[Any],
    run_spans: list[list[int]],
    *,
    tolerance: float,
    node_balance_weight: float = 0.8,
    segment_balance_weight: float = 0.8,
    g2_weight: float = 1.0,
    handle_shrink_weight: float = 0.05,
    max_split_depth: int = 5,
) -> list[BezPath]:
    """Fit a contiguous curve run between run_nodes using least squares.

    The solver balances:
    - sample fit error,
    - continuity preference (`G2` nodes enforce stronger in/out equality),
    - around-node handle balancing (incoming ~= outgoing),
    - across-segment handle balancing (start handle ~= end handle).

    Segments that still exceed `tolerance` are recursively split at the point of
    maximum divergence and refit.

    When all structural weights are zero, handles are fitted as unconstrained
    2D control points (not forced to precomputed tangent directions).
    """

    if len(run_nodes) < 2 or len(run_spans) != len(run_nodes) - 1:
        return []

    def _is_unconstrained_boundary_node(node: Any) -> bool:
        reasons = set(getattr(node, "reason_tags", set()) or set())
        continuity = getattr(node, "continuity", None)
        return "sharp_corner" in reasons or continuity == "non-continuous"

    contour_points_rc = np.asarray(contour_points_rc, dtype=np.float64)
    node_points_xy = [
        _as_xy(contour_points_rc[int(node.contour_index)]) for node in run_nodes
    ]
    continuities = [getattr(node, "continuity", None) for node in run_nodes]
    axis_extrema_by_node = [
        set(getattr(node, "axis_extrema", set()) or set()) for node in run_nodes
    ]
    node_tangents = _node_tangents_from_run(
        node_points_xy,
        continuities,
        axis_extrema_by_node,
    )

    segment_points_xy: list[np.ndarray] = []
    for span in run_spans:
        pts = contour_points_rc[np.asarray(span, dtype=np.int64)]
        segment_points_xy.append(np.stack([pts[:, 1], pts[:, 0]], axis=1))

    out_len, in_len = _global_handle_lengths(
        node_points_xy,
        node_tangents,
        segment_points_xy,
        continuities,
        node_balance_weight=node_balance_weight,
        segment_balance_weight=segment_balance_weight,
        g2_weight=g2_weight,
        handle_shrink_weight=handle_shrink_weight,
    )

    paths: list[BezPath] = []
    boundary_start_free = _is_unconstrained_boundary_node(run_nodes[0])
    boundary_end_free = _is_unconstrained_boundary_node(run_nodes[-1])
    unconstrained_handles = (
        node_balance_weight <= 0.0
        and segment_balance_weight <= 0.0
        and g2_weight <= 0.0
    )

    for si, points_xy in enumerate(segment_points_xy):
        t0 = node_tangents[si]
        t3 = node_tangents[si + 1]
        segment_free_start = (not unconstrained_handles) and (
            si == 0 and boundary_start_free
        )
        segment_free_end = (not unconstrained_handles) and (
            si == len(segment_points_xy) - 1 and boundary_end_free
        )

        if unconstrained_handles:
            seed_cubic = _fit_single_cubic_unconstrained(
                points_xy,
                handle_shrink_weight=handle_shrink_weight,
            )
        elif segment_free_start or segment_free_end:
            seed_cubic = _fit_single_cubic_with_partial_constraints(
                points_xy,
                t0,
                t3,
                free_start_handle=segment_free_start,
                free_end_handle=segment_free_end,
                segment_balance_weight=segment_balance_weight,
                handle_shrink_weight=handle_shrink_weight,
            )
        else:
            p0 = node_points_xy[si]
            p3 = node_points_xy[si + 1]
            c1 = p0 + out_len[si] * t0
            c2 = p3 - in_len[si + 1] * t3
            seed_cubic = CubicBez(
                Point(float(p0[0]), float(p0[1])),
                Point(float(c1[0]), float(c1[1])),
                Point(float(c2[0]), float(c2[1])),
                Point(float(p3[0]), float(p3[1])),
            )

        max_error, _ = _segment_max_error(seed_cubic, points_xy)
        if max_error <= tolerance:
            fitted = [_FittedSegment(cubic=seed_cubic, max_error=max_error)]
        else:
            fitted = _split_and_refit(
                points_xy,
                t0,
                t3,
                tolerance=tolerance,
                max_depth=max_split_depth,
                segment_balance_weight=segment_balance_weight,
                handle_shrink_weight=handle_shrink_weight,
                unconstrained_handles=unconstrained_handles,
                free_start_handle=segment_free_start,
                free_end_handle=segment_free_end,
            )

        path = BezPath()
        first = fitted[0].cubic
        path.move_to(Point(float(first.p0.x), float(first.p0.y)))
        for frag in fitted:
            path.curve_to(frag.cubic.p1, frag.cubic.p2, frag.cubic.p3)
        paths.append(path)

    return paths
