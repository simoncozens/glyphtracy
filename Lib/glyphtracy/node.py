from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from glyphtracy.contour import Contour

import numpy as np

NodeTransition = Literal["curve-curve", "curve-line", "line-line"]
Continuity = Literal["G1", "G2", "non-continuous"]
SegmentKind = Literal["line", "bezier"]
AxisExtremaTag = Literal["x", "y"]


@dataclass
class Node:
    node_id: int
    contour: "Contour"
    contour_index: int
    reason_tags: set[str]
    axis_extrema: set[AxisExtremaTag] = field(default_factory=set)
    transition: Optional[NodeTransition] = None
    continuity: Optional[Continuity] = None

    @property
    def point_rc(self) -> np.ndarray:
        return self.contour.points_rc[self.contour_index]

    def classify_continuity(
        self,
        before_points_rc: np.ndarray,
        after_points_rc: np.ndarray,
    ) -> Continuity:
        if self.transition != "curve-curve":
            return "non-continuous"

        def _normalized(vec: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-9:
                return np.array([1.0, 0.0], dtype=np.float64)
            return vec / norm

        def _weighted_tangent(polyline: np.ndarray) -> np.ndarray:
            if polyline.shape[0] < 2:
                return np.array([1.0, 0.0], dtype=np.float64)
            segments = polyline[1:] - polyline[:-1]
            dirs = np.array([_normalized(seg) for seg in segments], dtype=np.float64)
            weights = np.linspace(1.0, 2.0, dirs.shape[0], dtype=np.float64)
            vec = np.sum(dirs * weights[:, None], axis=0)
            return _normalized(vec)

        def _local_chain(polyline: np.ndarray, *, from_end: bool) -> np.ndarray:
            if polyline.shape[0] <= 4:
                return polyline

            segment_lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
            total_length = float(np.sum(segment_lengths))
            target_length = min(24.0, max(8.0, 0.12 * total_length))

            if from_end:
                acc_length = 0.0
                start_ix = polyline.shape[0] - 2
                while start_ix > 0 and acc_length < target_length:
                    acc_length += float(segment_lengths[start_ix])
                    start_ix -= 1
                return polyline[start_ix:]

            acc_length = 0.0
            end_ix = 1
            while end_ix < polyline.shape[0] - 1 and acc_length < target_length:
                acc_length += float(segment_lengths[end_ix - 1])
                end_ix += 1
            return polyline[:end_ix]

        def _signed_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ab = b - a
            bc = c - b
            ac = c - a
            ab_len = float(np.linalg.norm(ab))
            bc_len = float(np.linalg.norm(bc))
            ac_len = float(np.linalg.norm(ac))
            if min(ab_len, bc_len, ac_len) <= 1e-6:
                return 0.0
            cross = float(ab[0] * bc[1] - ab[1] * bc[0])
            return 2.0 * cross / (ab_len * bc_len * ac_len)

        if before_points_rc.shape[0] < 3 or after_points_rc.shape[0] < 3:
            return "non-continuous"

        before_local = _local_chain(before_points_rc, from_end=True)
        after_local = _local_chain(after_points_rc, from_end=False)
        if before_local.shape[0] < 3 or after_local.shape[0] < 3:
            return "non-continuous"

        tan_in = _weighted_tangent(before_local)
        tan_out = _weighted_tangent(after_local)

        dot = float(np.dot(tan_in, tan_out))
        dot = max(-1.0, min(1.0, dot))
        turn = float(np.arccos(dot))
        if turn > 0.35:
            return "non-continuous"

        k_in = _signed_curvature(
            before_local[-3],
            before_local[-2],
            before_local[-1],
        )
        k_out = _signed_curvature(
            after_local[0],
            after_local[1],
            after_local[2],
        )
        k_scale = max(abs(k_in), abs(k_out), 1e-6)
        k_match = abs(k_in - k_out) <= (0.06 + 0.35 * k_scale)

        if turn <= 0.14 and k_match:
            return "G2"
        return "G1"

    def debug_entry(self) -> dict:
        point_rc = self.point_rc
        return {
            "node_id": int(self.node_id),
            "contour_id": int(self.contour.contour_id),
            "contour_index": int(self.contour_index),
            "point_rc": [float(point_rc[0]), float(point_rc[1])],
            "reasons": sorted(self.reason_tags),
            "axis_extrema": sorted(self.axis_extrema),
            "transition": self.transition,
            "continuity": self.continuity,
        }
