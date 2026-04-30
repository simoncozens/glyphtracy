import paper from "paper";

export type NodeTransition = "curve-curve" | "curve-line" | "line-line";
export type Continuity = "G1" | "G2" | "non-continuous";
export type SegmentKind = "line" | "bezier";
export type AxisExtremaTag = "x" | "y";

export interface DebugResult {
  final_path: string;
  contours: Array<{
    contour_id: number;
    points: number[][];
    fitted: Array<Record<string, unknown>>;
  }>;
  nodes: Array<Record<string, unknown>>;
  segments: Array<Record<string, unknown>>;
}

export interface PathSegment {
  segmentId: number;
  contour: import("./contour.js").Contour;
  startNode: import("./node.js").Node;
  endNode: import("./node.js").Node;
  contourIndices: number[];
  kind: SegmentKind;
  path: paper.Path;
}
