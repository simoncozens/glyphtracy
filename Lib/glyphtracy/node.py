from typing import Literal, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

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
