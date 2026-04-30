"""
Glyphtracy: A raster to vector conversion library focused on high-quality curve fitting and node placement for font design and similar applications.

This module provides the main Vectorizer class which implements the full pipeline of contour extraction, node identification, path segment creation, and curve fitting. It also includes a command-line interface for processing images and outputting SVG paths along with detailed debug information.
The Vectorizer class is highly configurable with parameters for tuning node placement sensitivity, curve fitting tolerance, and handle balancing. The output includes both the final SVG path and a structured debug payload containing details of the contours, nodes, and segments for further analysis or visualization.

Usage:

    python -m glyphtracy input_image.png --output output_image.svg --debug-json debug_output.json

The debug JSON file contains a detailed breakdown of the contours, nodes, and segments identified and created during the vectorization process, which can be used for debugging or visualization purposes.
A debugging viewer is available at debugger/split_debug_viewer.html that can load the debug JSON output and visualize the contours, nodes, and segments interactively.
Open this HTML file in a web browser, and load the generated debug_output.json to see the results of the vectorization process, including the identified nodes and fitted segments overlaid on the original contours.

Programmatic usage:

    from glyphtracy import Vectorizer

    vectorizer = Vectorizer(
        image_source="input_image.png",
        sharp_threshold=math.radians(30),
        pixel_tolerance=1.5,
        # ... - see Vectorizer.__init__ for all parameters
    )
    final_path, debug_data = vectorizer.run()
    # final_path is a BezPath object representing the vectorized image
    # debug_data is a structured dictionary containing details of contours, nodes, and segments for debugging/visualization
"""

from typing import Tuple
from typing import TypedDict
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import skimage.measure
from kurbopy import BezPath, CubicBez, Line, Point
from PIL import Image

from glyphtracy.contour import Contour
from glyphtracy.node import AxisExtremaTag, Node, SegmentKind
from glyphtracy.fit import fit_curve_run
from glyphtracy.utils import (
    spans_between_anchors_closed,
    cyclic_distance,
    span_indices_closed,
    dedupe_closed_contour,
    is_basically_a_line,
)


@dataclass
class PathSegment:
    segment_id: int
    contour: Contour
    start_node: Node
    end_node: Node
    contour_indices: list[int]
    kind: SegmentKind
    bezpath: BezPath

    def debug_entry(self) -> dict:
        start_rc = self.contour.points_rc[self.contour_indices[0]]
        end_rc = self.contour.points_rc[self.contour_indices[-1]]
        return {
            "segment_id": int(self.segment_id),
            "contour_id": int(self.contour.contour_id),
            "start_node_id": int(self.start_node.node_id),
            "end_node_id": int(self.end_node.node_id),
            "kind": self.kind,
            "stage": "path_segment",
            "num_points": int(len(self.contour_indices)),
            "contour_indices": [int(i) for i in self.contour_indices],
            "start_point_rc": [float(start_rc[0]), float(start_rc[1])],
            "end_point_rc": [float(end_rc[0]), float(end_rc[1])],
        }


class DebugResult(TypedDict):
    final_path: str
    contours: list[dict]
    nodes: list[dict]
    segments: list[dict]


class Vectorizer:
    """Vectorize raster images to Bezier curves with tunable parameters.

    Resolution-dependent parameters (fit_tolerance, pixel_tolerance, extrema_min_index_gap,
    extrema_min_span_points) are automatically scaled based on image size.
    """

    # Reference image size for resolution-dependent parameter defaults
    _REFERENCE_SIZE = 512

    def __init__(
        self,
        image_source,
        *,
        sharp_threshold: float = math.radians(30),
        pixel_tolerance: float = 1.5,
        extrema_min_index_gap: int = 3,
        extrema_min_span_points: int = 8,
        extrema_max_iterations: int = 8,
        fit_tolerance: float = 5.0,
        node_balance_weight: float = 0.8,
        segment_balance_weight: float = 0.8,
        g2_weight: float = 1.0,
        handle_shrink_weight: float = 0.05,
        max_split_depth: int = 5,
        resolution_scale: Optional[float] = None,
    ):
        """Initialize vectorizer with image and tuning parameters.

        Args:
            image_source: Path to image file or numpy array (float [0,1]).
            sharp_threshold: Angle threshold (radians) for corner detection.
            pixel_tolerance: Max deviation for line detection (pixels at reference size).
            extrema_min_index_gap: Minimum gap between detected extrema nodes (at reference size).
            extrema_min_span_points: Minimum points in extrema search spans (at reference size).
            extrema_max_iterations: Max iterations for iterative extrema search.
            fit_tolerance: Max error tolerance for Bezier fitting (pixels at reference size).
            node_balance_weight: Weight for incoming/outgoing handle balance.
            segment_balance_weight: Weight for segment-level handle balance.
            g2_weight: Weight for G2 continuity preference at nodes.
            handle_shrink_weight: Regularization to keep handles small.
            max_split_depth: Max recursion depth for split-on-error fitting.
            resolution_scale: Override automatic resolution scaling (None = auto-compute from image size).
        """
        # Load image
        if isinstance(image_source, str):
            self.image_array = 1.0 - (
                np.asarray(Image.open(image_source).convert("L")) / 255.0
            )
        else:
            self.image_array = np.asarray(image_source, dtype=np.float64)

        # Compute resolution scale factor based on image size
        self.height, self.width = self.image_array.shape[:2]
        image_size = np.sqrt(self.height * self.width)
        if resolution_scale is None:
            self.resolution_scale = image_size / self._REFERENCE_SIZE
        else:
            self.resolution_scale = float(resolution_scale)

        # Fixed parameters (NOT resolution-dependent)
        self.sharp_threshold = float(sharp_threshold)
        self.node_balance_weight = float(node_balance_weight)
        self.segment_balance_weight = float(segment_balance_weight)
        self.g2_weight = float(g2_weight)
        self.handle_shrink_weight = float(handle_shrink_weight)
        self.max_split_depth = int(max_split_depth)
        self.extrema_max_iterations = int(extrema_max_iterations)

        # Resolution-dependent parameters (scaled by resolution_scale)
        self.fit_tolerance = float(fit_tolerance) * self.resolution_scale
        self.pixel_tolerance = float(pixel_tolerance) * self.resolution_scale
        self.extrema_min_index_gap = max(
            1, int(extrema_min_index_gap * self.resolution_scale)
        )
        self.extrema_min_span_points = max(
            1, int(extrema_min_span_points * self.resolution_scale)
        )

    def dump_paths_for_debug(self, contour_paths):
        # Dump kurbo representations of the fitted contour for debugging/visualization
        dump_contour_segments = []
        for segment in contour_paths[-1].segments():
            if isinstance(segment, Line):
                dump_contour_segments.append(
                    {
                        "type": "line",
                        "start": [
                            float(segment.start().x),
                            float(segment.start().y),
                        ],
                        "end": [float(segment.end().x), float(segment.end().y)],
                    }
                )
            elif isinstance(segment, CubicBez):
                dump_contour_segments.append(
                    {
                        "type": "cubic",
                        "start": [
                            float(segment.start().x),
                            float(segment.start().y),
                        ],
                        "control1": [float(segment.p1.x), float(segment.p1.y)],
                        "control2": [float(segment.p2.x), float(segment.p2.y)],
                        "end": [float(segment.p3.x), float(segment.p3.y)],
                    }
                )
        return dump_contour_segments

    def run(self) -> Tuple[BezPath, DebugResult]:
        """Execute full vectorization pipeline

        Returns:
            final_path: The composed BezPath representing the entire vectorized image.
            debug_data: A structured dictionary containing details of contours, nodes, and segments for debugging/
        """
        contours = self.extract_contours()
        nodes: list[Node] = []
        segments: list[PathSegment] = []
        contour_paths: list[BezPath] = []

        next_node_id = 0
        next_segment_id = 0
        final_contour_segments = []
        for contour in contours:
            contour_nodes = self.identify_nodes(contour, starting_node_id=next_node_id)
            if contour_nodes:
                next_node_id = contour_nodes[-1].node_id
            nodes.extend(contour_nodes)

            contour_segments = self.create_path_segments(
                contour,
                contour_nodes,
                starting_segment_id=next_segment_id,
            )
            if contour_segments:
                next_segment_id = contour_segments[-1].segment_id
            segments.extend(contour_segments)

            annotate_transitions(contour_nodes, contour_segments)

            contour_paths.append(compose_contour_path(contour_segments))
            final_contour_segments.append(self.dump_paths_for_debug(contour_paths))

        final_path = compose_final_path(contour_paths)

        return final_path, DebugResult(
            final_path=final_path.to_svg(),
            contours=[
                {
                    "contour_id": contour.contour_id,
                    "points": [
                        [float(v) for v in point] for point in contour.points_rc
                    ],
                    "fitted": final_contour_segments[ix],
                }
                for ix, contour in enumerate(contours)
            ],
            nodes=[node.debug_entry() for node in nodes],
            segments=[segment.debug_entry() for segment in segments],
        )

    def extract_contours(self) -> list[Contour]:
        """Extract contours from image using skimage."""
        contours: list[Contour] = []
        for contour_id, contour in enumerate(
            skimage.measure.find_contours(self.image_array, level=0.5)
        ):
            rc = dedupe_closed_contour(np.asarray(contour, dtype=np.float64))
            if rc.shape[0] < 3:
                continue
            contours.append(Contour(contour_id=contour_id, points_rc=rc, closed=True))
        return contours

    def identify_nodes(
        self, contour: Contour, *, starting_node_id: int = 0
    ) -> list[Node]:
        """Identify nodes at corners and axis extrema."""
        nodes: list[Node] = []
        node_counter = starting_node_id

        n = contour.size
        if n < 3:
            return nodes

        candidates: dict[int, set[str]] = {}
        axis_extrema: dict[int, set[AxisExtremaTag]] = {}

        sharp_corner_indices: set[int] = set()
        for i in range(n):
            if contour.corner_angle(i, stride=1) >= self.sharp_threshold:
                sharp_corner_indices.add(i)
                candidates.setdefault(i, set()).add("sharp_corner")

        axis_map: dict[int, set[AxisExtremaTag]] = {}
        if len(sharp_corner_indices) >= 2:
            for span_indices in spans_between_anchors_closed(
                list(sharp_corner_indices), n
            ):
                for idx, tags in contour.iterative_axis_extrema_axes_on_open_span(
                    span_indices,
                    min_index_gap=self.extrema_min_index_gap,
                    min_span_points=max(self.extrema_min_span_points, n // 120),
                    max_iterations=self.extrema_max_iterations,
                ).items():
                    axis_map.setdefault(idx, set()).update(tags)
        else:
            axis_map = contour.iterative_axis_extrema_axes(
                min_index_gap=self.extrema_min_index_gap,
                min_span_points=max(self.extrema_min_span_points, n // 120),
                max_iterations=self.extrema_max_iterations,
            )

        for idx, tags in axis_map.items():
            axis_extrema.setdefault(idx, set()).update(tags)
            candidates.setdefault(idx, set()).add("axis_extrema")

        merged = sorted(candidates.keys())
        kept: list[int] = []
        for idx in merged:
            close_matches = [k for k in kept if cyclic_distance(idx, k, n) < 2]
            if not close_matches:
                kept.append(idx)
                continue

            closest = min(close_matches, key=lambda k: cyclic_distance(idx, k, n))
            existing_reasons = candidates[closest]
            current_reasons = candidates[idx]
            existing_axes = axis_extrema.get(closest, set())
            current_axes = axis_extrema.get(idx, set())

            if _reason_priority(current_reasons) > _reason_priority(existing_reasons):
                combined_reasons = set(existing_reasons)
                combined_reasons.update(current_reasons)
                combined_axes = set(existing_axes)
                combined_axes.update(current_axes)
                kept[kept.index(closest)] = idx
                candidates[idx] = combined_reasons
                axis_extrema[idx] = combined_axes
            else:
                existing_reasons.update(current_reasons)
                axis_extrema[closest] = set(existing_axes).union(current_axes)

        kept = sorted(set(kept))

        for idx in kept:
            node_counter += 1
            node = Node(
                node_id=node_counter,
                contour=contour,
                contour_index=idx,
                reason_tags=candidates[idx],
                axis_extrema=set(axis_extrema.get(idx, set())),
                transition=None,
                continuity=contour.classify_continuity(idx),
            )
            nodes.append(node)

        return nodes

    def create_path_segments(
        self,
        contour: Contour,
        nodes: list[Node],
        *,
        starting_segment_id: int = 0,
    ) -> list[PathSegment]:
        """Create path segments between nodes, fitting curves as needed."""

        def _should_break_curve_run_at_node(node: Node) -> bool:
            if "sharp_corner" in node.reason_tags:
                return True
            return node.continuity == "non-continuous"

        def _fit_run_meta(run_meta: list[dict]) -> list[BezPath]:
            run_nodes = [run_meta[0]["start_node"]] + [m["end_node"] for m in run_meta]
            run_spans = [m["span"] for m in run_meta]

            fitted_paths = fit_curve_run(
                contour.points_rc,
                run_nodes,
                run_spans,
                tolerance=self.fit_tolerance,
                node_balance_weight=self.node_balance_weight,
                segment_balance_weight=self.segment_balance_weight,
                g2_weight=self.g2_weight,
                handle_shrink_weight=self.handle_shrink_weight,
                max_split_depth=self.max_split_depth,
            )

            if len(fitted_paths) == len(run_meta):
                return fitted_paths

            fallback_paths: list[BezPath] = []
            for m in run_meta:
                subcontour = m["subcontour"]
                bez = BezPath()
                bez.move_to(Point(subcontour[0, 1], subcontour[0, 0]))
                bez.curve_to(
                    Point(subcontour[0, 1], subcontour[0, 0]),
                    Point(subcontour[-1, 1], subcontour[-1, 0]),
                    Point(subcontour[-1, 1], subcontour[-1, 0]),
                )
                fallback_paths.append(bez)
            return fallback_paths

        segments: list[PathSegment] = []
        segment_counter = starting_segment_id
        synthetic_node_counter = -1
        nodes_for_contour = sorted(nodes, key=lambda node: node.contour_index)
        if len(nodes_for_contour) < 2:
            synthetic_a = Node(
                node_id=synthetic_node_counter,
                contour=contour,
                contour_index=0,
                reason_tags={"synthetic_boundary"},
            )
            synthetic_node_counter -= 1
            synthetic_b = Node(
                node_id=synthetic_node_counter,
                contour=contour,
                contour_index=max(0, contour.size - 1),
                reason_tags={"synthetic_boundary"},
            )
            nodes_for_contour = [synthetic_a, synthetic_b]

        spans_meta: list[dict] = []
        for i, start_node in enumerate(nodes_for_contour):
            end_node = nodes_for_contour[(i + 1) % len(nodes_for_contour)]
            if not contour.closed and i == len(nodes_for_contour) - 1:
                continue

            span = span_indices_closed(
                start_node.contour_index,
                end_node.contour_index,
                contour.size,
            )
            if len(span) < 2:
                continue

            subcontour = contour.points_for_indices(span)
            spans_meta.append(
                {
                    "start_node": start_node,
                    "end_node": end_node,
                    "span": span,
                    "subcontour": subcontour,
                    "is_line": is_basically_a_line(
                        subcontour, pixel_tolerance=self.pixel_tolerance
                    ),
                }
            )

        if not spans_meta:
            return segments

        if contour.closed and len(spans_meta) > 1:
            first_is_curve = not spans_meta[0]["is_line"]
            last_is_curve = not spans_meta[-1]["is_line"]
            if first_is_curve and last_is_curve:
                first_line_ix = next(
                    (ix for ix, meta in enumerate(spans_meta) if meta["is_line"]),
                    None,
                )
                if first_line_ix is not None:
                    spans_meta = spans_meta[first_line_ix:] + spans_meta[:first_line_ix]

        cursor = 0
        while cursor < len(spans_meta):
            meta = spans_meta[cursor]

            if meta["is_line"]:
                subcontour = meta["subcontour"]
                bez = BezPath()
                bez.move_to(Point(subcontour[0, 1], subcontour[0, 0]))
                bez.line_to(Point(subcontour[-1, 1], subcontour[-1, 0]))
                segment_counter += 1
                segments.append(
                    PathSegment(
                        segment_id=segment_counter,
                        contour=contour,
                        start_node=meta["start_node"],
                        end_node=meta["end_node"],
                        contour_indices=meta["span"],
                        kind="line",
                        bezpath=bez,
                    )
                )
                cursor += 1
                continue

            run_end = cursor
            while run_end < len(spans_meta) and not spans_meta[run_end]["is_line"]:
                run_end += 1

            run_meta = spans_meta[cursor:run_end]
            split_after_indices = [
                i
                for i in range(len(run_meta) - 1)
                if _should_break_curve_run_at_node(run_meta[i]["end_node"])
            ]

            chunk_start = 0
            for split_ix in split_after_indices + [len(run_meta) - 1]:
                subrun_meta = run_meta[chunk_start : split_ix + 1]
                fitted_paths = _fit_run_meta(subrun_meta)

                for m, bez in zip(subrun_meta, fitted_paths):
                    segment_counter += 1
                    segments.append(
                        PathSegment(
                            segment_id=segment_counter,
                            contour=contour,
                            start_node=m["start_node"],
                            end_node=m["end_node"],
                            contour_indices=m["span"],
                            kind="bezier",
                            bezpath=bez,
                        )
                    )
                chunk_start = split_ix + 1

            cursor = run_end

        return segments


def _reason_priority(reasons: set[str]) -> tuple[int, int, int]:
    return (
        1 if "sharp_corner" in reasons else 0,
        1 if "inflection" in reasons else 0,
        1 if "axis_extrema" in reasons else 0,
    )


def rounded(point: Point, dp: int = 1) -> Point:
    return Point(round(point.x, dp), round(point.y, dp))


def annotate_transitions(nodes: list[Node], segments: list[PathSegment]) -> None:
    """Set node.transition based on the kind of the full segments on each side."""
    incoming: dict[int, SegmentKind] = {}
    outgoing: dict[int, SegmentKind] = {}
    for seg in segments:
        if seg.end_node.node_id > 0:
            incoming[seg.end_node.node_id] = seg.kind
        if seg.start_node.node_id > 0:
            outgoing[seg.start_node.node_id] = seg.kind

    for node in nodes:
        before_kind = incoming.get(node.node_id)
        after_kind = outgoing.get(node.node_id)
        before_line = before_kind == "line"
        after_line = after_kind == "line"
        if before_line and after_line:
            node.transition = "line-line"
        elif before_line or after_line:
            node.transition = "curve-line"
        else:
            node.transition = "curve-curve"


def compose_contour_path(segments: list[PathSegment]) -> BezPath:
    contour_path = BezPath()
    if not segments:
        return contour_path

    first_elems = list(segments[0].bezpath.elements())
    if not first_elems:
        return contour_path
    contour_path.push(first_elems[0])

    for segment in segments:
        for seg in segment.bezpath.segments():
            if isinstance(seg, Line):
                contour_path.line_to(seg.end())
            elif isinstance(seg, CubicBez):
                contour_path.curve_to(
                    seg.p1,
                    seg.p2,
                    seg.p3,
                )
    return contour_path


def compose_final_path(contour_paths: list[BezPath]) -> BezPath:
    final_path = BezPath()
    for contour_path in contour_paths:
        for element in contour_path.elements():
            final_path.push(element)
        final_path.close_path()
    return final_path


if __name__ == "__main__":
    import json
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Vectorize an image to SVG path.")
    parser.add_argument("image_path", type=str, help="Path to input image file.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output SVG file",
    )
    parser.add_argument(
        "--debug-json",
        type=str,
        default="debug_output.json",
        help="Path to output debug JSON file (default: debug_output.json).",
    )
    node_group = parser.add_argument_group("Node placement parameters")
    node_group.add_argument(
        "--sharp-threshold",
        type=lambda x: math.radians(float(x)),
        default=30,
        help="Angle threshold in degrees for corner detection. Decrease this if nodes are not placed at corners. Increase it if smooth curves are being turned into sharp corners. (default: 30).",
    )
    node_group.add_argument(
        "--pixel-tolerance",
        type=float,
        default=1.5,
        help="Max deviation for line detection (pixels at reference size, default: 1.5).",
    )
    node_group.add_argument(
        "--extrema-min-index-gap",
        type=int,
        default=3,
        help="Minimum gap between detected extrema nodes (at reference size, default: 3).",
    )
    node_group.add_argument(
        "--extrema-min-span-points",
        type=int,
        default=8,
        help="Minimum points in extrema search spans (at reference size, default: 8).",
    )
    node_group.add_argument(
        "--extrema-max-iterations",
        type=int,
        default=8,
        help="Max iterations for iterative extrema search (default: 8).",
    )

    fit_group = parser.add_argument_group("Path fitting parameters")
    fit_group.add_argument(
        "--fit-tolerance",
        type=float,
        default=5.0,
        help="Max error tolerance for Bezier fitting (pixels at reference size, default: 5.0).",
    )
    fit_group.add_argument(
        "--node-balance-weight",
        type=float,
        default=0.8,
        help="Weight for incoming/outgoing handle length balancing. Increase to make node handle lengths more equal. (default: 0.8).",
    )
    fit_group.add_argument(
        "--segment-balance-weight",
        type=float,
        default=0.8,
        help="Weight for segment-level handle length balancing. Increase to make segment handle lengths more equal. (default: 0.8).",
    )
    fit_group.add_argument(
        "--g2-weight",
        type=float,
        default=1.0,
        help="Weight for G2 continuity preference at nodes. Increase to favor smoother curvature transitions at the cost of potentially worse fit error. (default: 1.0).",
    )
    fit_group.add_argument(
        "--handle-shrink-weight",
        type=float,
        default=0.05,
        help="Regularization weight to keep handle lengths small. Increase to discourage long handles that can cause loops and wild curves. (default: 0.05).",
    )
    fit_group.add_argument(
        "--max-split-depth",
        type=int,
        default=5,
        help="Max recursion depth for split-on-error fitting. Increase to allow more splits for difficult curve runs, at the cost of longer fitting time. (default: 5).",
    )
    args = parser.parse_args()

    vectorizer = Vectorizer(
        args.image_path,
        fit_tolerance=args.fit_tolerance,
        node_balance_weight=args.node_balance_weight,
        segment_balance_weight=args.segment_balance_weight,
        g2_weight=args.g2_weight,
        handle_shrink_weight=args.handle_shrink_weight,
        max_split_depth=args.max_split_depth,
        sharp_threshold=args.sharp_threshold,
        pixel_tolerance=args.pixel_tolerance,
        extrema_min_index_gap=args.extrema_min_index_gap,
        extrema_min_span_points=args.extrema_min_span_points,
        extrema_max_iterations=args.extrema_max_iterations,
    )
    path, debug_payload = vectorizer.run()
    if args.debug_json:
        Path(args.debug_json).write_text(
            json.dumps(debug_payload, indent=2),
            encoding="utf-8",
        )
    d_path = path.to_svg()
    if args.output is None:
        args.output = args.image_path.rsplit(".", 1)[0] + ".svg"
    print(f"Writing SVG output to {args.output}...")
    # Put it in a nice SVG wrapper
    viewbox = f"0 0 {vectorizer.width} {vectorizer.height}"
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">
        <path d="{d_path}" fill="black" />
    </svg>"""
    Path(args.output).write_text(svg_content, encoding="utf-8")
