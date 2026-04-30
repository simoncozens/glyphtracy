import paper from "paper";
import { isoLines } from "marchingsquares";
import { Contour } from "./contour.js";
import { fitCurveRun } from "./fit.js";
import { Node } from "./node.js";
import { AxisExtremaTag, DebugResult, PathSegment } from "./types.js";
import {
  cyclicDistance,
  dedupeClosedContour,
  isBasicallyALine,
  spanIndicesClosed,
  spansBetweenAnchorsClosed,
} from "./utils.js";

export interface VectorizerOptions {
  sharpThreshold?: number;
  pixelTolerance?: number;
  extremaMinIndexGap?: number;
  extremaMinSpanPoints?: number;
  extremaMaxIterations?: number;
  contourLevel?: number;
  fitTolerance?: number;
  nodeBalanceWeight?: number;
  segmentBalanceWeight?: number;
  g2Weight?: number;
  handleShrinkWeight?: number;
  maxSplitDepth?: number;
  resolutionScale?: number;
}

function reasonPriority(reasons: Set<string>): [number, number, number] {
  return [
    reasons.has("sharp_corner") ? 1 : 0,
    reasons.has("inflection") ? 1 : 0,
    reasons.has("axis_extrema") ? 1 : 0,
  ];
}

function comparePriority(
  a: [number, number, number],
  b: [number, number, number],
): number {
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return a[i] - b[i];
    }
  }
  return 0;
}

function annotateTransitions(nodes: Node[], segments: PathSegment[]): void {
  const incoming = new Map<number, PathSegment>();
  const outgoing = new Map<number, PathSegment>();

  for (const seg of segments) {
    if (seg.endNode.nodeId > 0) {
      incoming.set(seg.endNode.nodeId, seg);
    }
    if (seg.startNode.nodeId > 0) {
      outgoing.set(seg.startNode.nodeId, seg);
    }
  }

  for (const node of nodes) {
    const beforeSeg = incoming.get(node.nodeId);
    const afterSeg = outgoing.get(node.nodeId);
    const beforeLine = beforeSeg !== undefined && beforeSeg.kind === "line";
    const afterLine = afterSeg !== undefined && afterSeg.kind === "line";

    if (beforeLine && afterLine) {
      node.transition = "line-line";
    } else if (beforeLine || afterLine) {
      node.transition = "curve-line";
    } else {
      node.transition = "curve-curve";
    }

    if (!beforeSeg || !afterSeg) {
      node.continuity = "non-continuous";
      continue;
    }

    const beforePoints = node.contour.pointsForIndices(
      beforeSeg.contourIndices,
    );
    const afterPoints = node.contour.pointsForIndices(afterSeg.contourIndices);
    node.continuity = node.classifyContinuity(beforePoints, afterPoints);
  }
}

function composeContourPath(segments: PathSegment[]): paper.Path {
  const contourPath = new paper.Path();
  if (segments.length === 0) {
    return contourPath;
  }

  contourPath.moveTo(segments[0].path.firstSegment.point);
  for (const segment of segments) {
    if (segment.kind === "line") {
      contourPath.lineTo(segment.path.lastSegment.point);
      continue;
    }

    for (let i = 1; i < segment.path.segments.length; i += 1) {
      const seg = segment.path.segments[i];
      const prev = segment.path.segments[i - 1];
      contourPath.cubicCurveTo(
        prev.point.add(prev.handleOut),
        seg.point.add(seg.handleIn),
        seg.point,
      );
    }
  }

  return contourPath;
}

function composeFinalPath(contourPaths: paper.Path[]): paper.CompoundPath {
  const finalPath = new paper.CompoundPath("");
  for (const contourPath of contourPaths) {
    contourPath.closed = true;
    finalPath.addChild(contourPath);
  }
  return finalPath;
}

function ensurePaperScope(width: number, height: number): void {
  paper.setup(new paper.Size(Math.max(1, width), Math.max(1, height)));
}

export class Vectorizer {
  private static readonly REFERENCE_SIZE = 512;
  imageArray: number[][];
  width: number;
  height: number;
  resolutionScale: number;

  sharpThreshold: number;
  pixelTolerance: number;
  extremaMinIndexGap: number;
  extremaMinSpanPoints: number;
  extremaMaxIterations: number;
  contourLevel: number;
  fitTolerance: number;
  nodeBalanceWeight: number;
  segmentBalanceWeight: number;
  g2Weight: number;
  handleShrinkWeight: number;
  maxSplitDepth: number;

  constructor(imageArray: number[][], options: VectorizerOptions = {}) {
    this.imageArray = imageArray;
    this.height = imageArray.length;
    this.width = imageArray[0]?.length ?? 0;

    const imageSize = Math.sqrt(this.height * this.width);
    this.resolutionScale =
      options.resolutionScale ?? imageSize / Vectorizer.REFERENCE_SIZE;

    this.sharpThreshold = options.sharpThreshold ?? Math.PI / 6;
    this.nodeBalanceWeight = options.nodeBalanceWeight ?? 0.8;
    this.segmentBalanceWeight = options.segmentBalanceWeight ?? 0.8;
    this.g2Weight = options.g2Weight ?? 1;
    this.handleShrinkWeight = options.handleShrinkWeight ?? 0.05;
    this.maxSplitDepth = options.maxSplitDepth ?? 5;
    this.extremaMaxIterations = options.extremaMaxIterations ?? 8;
    this.contourLevel = options.contourLevel ?? 0.5;

    this.fitTolerance = (options.fitTolerance ?? 5) * this.resolutionScale;
    this.pixelTolerance =
      (options.pixelTolerance ?? 1.5) * this.resolutionScale;
    this.extremaMinIndexGap = Math.max(
      1,
      Math.floor((options.extremaMinIndexGap ?? 3) * this.resolutionScale),
    );
    this.extremaMinSpanPoints = Math.max(
      1,
      Math.floor((options.extremaMinSpanPoints ?? 8) * this.resolutionScale),
    );
  }

  extractContours(): Contour[] {
    let raw = isoLines(this.imageArray, this.contourLevel) ?? [];
    // First one is always the image boundary, skip it
    raw = raw.slice(1);
    const contours: Contour[] = [];
    for (let contourId = 0; contourId < raw.length; contourId += 1) {
      const contour = raw[contourId].map((p) => [p[1], p[0]] as number[]);
      const rc = dedupeClosedContour(contour);
      if (rc.length < 3) {
        continue;
      }
      contours.push(new Contour(contourId, rc, true));
    }
    return contours;
  }

  identifyNodes(contour: Contour, startingNodeId = 0): Node[] {
    const nodes: Node[] = [];
    let nodeCounter = startingNodeId;

    const n = contour.size;
    if (n < 3) {
      return nodes;
    }

    const candidates = new Map<number, Set<string>>();
    const axisExtrema = new Map<number, Set<AxisExtremaTag>>();

    const sharpCornerIndices = new Set<number>();
    for (let i = 0; i < n; i += 1) {
      if (contour.cornerAngle(i, 1) >= this.sharpThreshold) {
        sharpCornerIndices.add(i);
        if (!candidates.has(i)) {
          candidates.set(i, new Set());
        }
        candidates.get(i)?.add("sharp_corner");
      }
    }

    let axisMap = new Map<number, Set<AxisExtremaTag>>();
    if (sharpCornerIndices.size >= 2) {
      for (const spanIndices of spansBetweenAnchorsClosed(
        [...sharpCornerIndices],
        n,
      )) {
        const spanMap = contour.iterativeAxisExtremaAxesOnOpenSpan(
          spanIndices,
          {
            minIndexGap: this.extremaMinIndexGap,
            minSpanPoints: Math.max(
              this.extremaMinSpanPoints,
              Math.floor(n / 120),
            ),
            maxIterations: this.extremaMaxIterations,
          },
        );
        for (const [idx, tags] of spanMap.entries()) {
          if (!axisMap.has(idx)) {
            axisMap.set(idx, new Set());
          }
          for (const tag of tags) {
            axisMap.get(idx)?.add(tag);
          }
        }
      }
    } else {
      axisMap = contour.iterativeAxisExtremaAxes({
        minIndexGap: this.extremaMinIndexGap,
        minSpanPoints: Math.max(this.extremaMinSpanPoints, Math.floor(n / 120)),
        maxIterations: this.extremaMaxIterations,
      });
    }

    for (const [idx, tags] of axisMap.entries()) {
      if (!axisExtrema.has(idx)) {
        axisExtrema.set(idx, new Set());
      }
      for (const tag of tags) {
        axisExtrema.get(idx)?.add(tag);
      }
      if (!candidates.has(idx)) {
        candidates.set(idx, new Set());
      }
      candidates.get(idx)?.add("axis_extrema");
    }

    const merged = [...candidates.keys()].sort((a, b) => a - b);
    const kept: number[] = [];
    for (const idx of merged) {
      const closeMatches = kept.filter((k) => cyclicDistance(idx, k, n) < 2);
      if (closeMatches.length === 0) {
        kept.push(idx);
        continue;
      }

      const closest = closeMatches.reduce((best, cur) =>
        cyclicDistance(idx, cur, n) < cyclicDistance(idx, best, n) ? cur : best,
      );

      const existingReasons = candidates.get(closest) ?? new Set<string>();
      const currentReasons = candidates.get(idx) ?? new Set<string>();
      const existingAxes =
        axisExtrema.get(closest) ?? new Set<AxisExtremaTag>();
      const currentAxes = axisExtrema.get(idx) ?? new Set<AxisExtremaTag>();

      if (
        comparePriority(
          reasonPriority(currentReasons),
          reasonPriority(existingReasons),
        ) > 0
      ) {
        const combinedReasons = new Set([
          ...existingReasons,
          ...currentReasons,
        ]);
        const combinedAxes = new Set([...existingAxes, ...currentAxes]);
        kept[kept.indexOf(closest)] = idx;
        candidates.set(idx, combinedReasons);
        axisExtrema.set(idx, combinedAxes);
      } else {
        for (const reason of currentReasons) {
          existingReasons.add(reason);
        }
        for (const axis of currentAxes) {
          existingAxes.add(axis);
        }
        candidates.set(closest, existingReasons);
        axisExtrema.set(closest, existingAxes);
      }
    }

    const finalKept = [...new Set(kept)].sort((a, b) => a - b);
    for (const idx of finalKept) {
      nodeCounter += 1;
      nodes.push(
        new Node({
          nodeId: nodeCounter,
          contour,
          contourIndex: idx,
          reasonTags: new Set(candidates.get(idx) ?? []),
          axisExtrema: new Set(axisExtrema.get(idx) ?? []),
        }),
      );
    }

    return nodes;
  }

  createPathSegments(
    contour: Contour,
    nodes: Node[],
    startingSegmentId = 0,
  ): PathSegment[] {
    const segments: PathSegment[] = [];
    let segmentCounter = startingSegmentId;
    let syntheticNodeCounter = -1;

    const nodesForContour = nodes
      .slice()
      .sort((a, b) => a.contourIndex - b.contourIndex);
    if (nodesForContour.length < 2) {
      const syntheticA = new Node({
        nodeId: syntheticNodeCounter,
        contour,
        contourIndex: 0,
        reasonTags: new Set(["synthetic_boundary"]),
      });
      syntheticNodeCounter -= 1;
      const syntheticB = new Node({
        nodeId: syntheticNodeCounter,
        contour,
        contourIndex: Math.max(0, contour.size - 1),
        reasonTags: new Set(["synthetic_boundary"]),
      });
      nodesForContour.push(syntheticA, syntheticB);
    }

    const spansMeta: Array<{
      startNode: Node;
      endNode: Node;
      span: number[];
      subcontour: number[][];
      isLine: boolean;
    }> = [];

    for (let i = 0; i < nodesForContour.length; i += 1) {
      const startNode = nodesForContour[i];
      const endNode = nodesForContour[(i + 1) % nodesForContour.length];
      if (!contour.closed && i === nodesForContour.length - 1) {
        continue;
      }

      const span = spanIndicesClosed(
        startNode.contourIndex,
        endNode.contourIndex,
        contour.size,
      );
      if (span.length < 2) {
        continue;
      }
      const subcontour = contour.pointsForIndices(span);
      spansMeta.push({
        startNode,
        endNode,
        span,
        subcontour,
        isLine: isBasicallyALine(subcontour, this.pixelTolerance),
      });
    }

    if (spansMeta.length === 0) {
      return segments;
    }

    if (contour.closed && spansMeta.length > 1) {
      const firstIsCurve = !spansMeta[0].isLine;
      const lastIsCurve = !spansMeta[spansMeta.length - 1].isLine;
      if (firstIsCurve && lastIsCurve) {
        const firstLineIx = spansMeta.findIndex((m) => m.isLine);
        if (firstLineIx !== -1) {
          const rotated = spansMeta
            .slice(firstLineIx)
            .concat(spansMeta.slice(0, firstLineIx));
          spansMeta.splice(0, spansMeta.length, ...rotated);
        }
      }
    }

    function shouldBreakCurveRunAtNode(node: Node): boolean {
      if (node.continuity === "non-continuous") {
        return true;
      }
      if (node.reasonTags.has("sharp_corner")) {
        return true;
      }
      return false;
    }

    function fitRunMeta(
      contourPointsRc: number[][],
      runMeta: Array<{
        startNode: Node;
        endNode: Node;
        span: number[];
        subcontour: number[][];
      }>,
      vectorizer: Vectorizer,
    ): paper.Path[] {
      const runNodes = [runMeta[0].startNode].concat(
        runMeta.map((m) => m.endNode),
      );
      const runSpans = runMeta.map((m) => m.span);

      const fittedPaths = fitCurveRun(contourPointsRc, runNodes, runSpans, {
        tolerance: vectorizer.fitTolerance,
        nodeBalanceWeight: vectorizer.nodeBalanceWeight,
        segmentBalanceWeight: vectorizer.segmentBalanceWeight,
        g2Weight: vectorizer.g2Weight,
        handleShrinkWeight: vectorizer.handleShrinkWeight,
        maxSplitDepth: vectorizer.maxSplitDepth,
      });

      if (fittedPaths.length === runMeta.length) {
        return fittedPaths;
      }

      const fallback: paper.Path[] = [];
      for (const m of runMeta) {
        const path = new paper.Path();
        const start = m.subcontour[0];
        const end = m.subcontour[m.subcontour.length - 1];
        path.moveTo(new paper.Point(start[1], start[0]));
        path.cubicCurveTo(
          new paper.Point(start[1], start[0]),
          new paper.Point(end[1], end[0]),
          new paper.Point(end[1], end[0]),
        );
        fallback.push(path);
      }
      return fallback;
    }

    let cursor = 0;
    while (cursor < spansMeta.length) {
      const meta = spansMeta[cursor];
      if (meta.isLine) {
        const start = meta.subcontour[0];
        const end = meta.subcontour[meta.subcontour.length - 1];
        const path = new paper.Path();
        path.moveTo(new paper.Point(start[1], start[0]));
        path.lineTo(new paper.Point(end[1], end[0]));

        segmentCounter += 1;
        segments.push({
          segmentId: segmentCounter,
          contour,
          startNode: meta.startNode,
          endNode: meta.endNode,
          contourIndices: meta.span,
          kind: "line",
          path,
        });
        cursor += 1;
        continue;
      }

      let runEnd = cursor;
      while (runEnd < spansMeta.length && !spansMeta[runEnd].isLine) {
        runEnd += 1;
      }
      const runMeta = spansMeta.slice(cursor, runEnd);
      const splitAfterIndices = Array.from(
        { length: Math.max(0, runMeta.length - 1) },
        (_, i) => i,
      ).filter((i) => shouldBreakCurveRunAtNode(runMeta[i].endNode));

      let chunkStart = 0;
      for (const splitIx of splitAfterIndices.concat([runMeta.length - 1])) {
        const subrunMeta = runMeta.slice(chunkStart, splitIx + 1);
        const fittedPaths = fitRunMeta(contour.pointsRc, subrunMeta, this);
        for (let i = 0; i < subrunMeta.length; i += 1) {
          segmentCounter += 1;
          segments.push({
            segmentId: segmentCounter,
            contour,
            startNode: subrunMeta[i].startNode,
            endNode: subrunMeta[i].endNode,
            contourIndices: subrunMeta[i].span,
            kind: "bezier",
            path: fittedPaths[i],
          });
        }
        chunkStart = splitIx + 1;
      }

      cursor = runEnd;
    }

    return segments;
  }

  run(): { finalPath: paper.CompoundPath; debugData: DebugResult } {
    ensurePaperScope(this.width, this.height);

    const contours = this.extractContours();
    const nodes: Node[] = [];
    const segments: PathSegment[] = [];
    const contourPaths: paper.Path[] = [];
    const fittedDump: Array<Array<Record<string, unknown>>> = [];

    let nextNodeId = 0;
    let nextSegmentId = 0;

    for (const contour of contours) {
      const contourNodes = this.identifyNodes(contour, nextNodeId);
      if (contourNodes.length > 0) {
        nextNodeId = contourNodes[contourNodes.length - 1].nodeId;
      }
      nodes.push(...contourNodes);

      const contourSegments = this.createPathSegments(
        contour,
        contourNodes,
        nextSegmentId,
      );
      if (contourSegments.length > 0) {
        nextSegmentId = contourSegments[contourSegments.length - 1].segmentId;
      }
      segments.push(...contourSegments);

      annotateTransitions(contourNodes, contourSegments);
      contourPaths.push(composeContourPath(contourSegments));

      const dump: Array<Record<string, unknown>> = [];
      for (const segment of contourSegments) {
        if (segment.kind === "line") {
          const start = segment.path.firstSegment.point;
          const end = segment.path.lastSegment.point;
          dump.push({
            type: "line",
            start: [start.x, start.y],
            end: [end.x, end.y],
          });
          continue;
        }

        for (let i = 1; i < segment.path.segments.length; i += 1) {
          const seg = segment.path.segments[i];
          const prev = segment.path.segments[i - 1];
          dump.push({
            type: "cubic",
            start: [prev.point.x, prev.point.y],
            control1: [
              prev.point.x + prev.handleOut.x,
              prev.point.y + prev.handleOut.y,
            ],
            control2: [
              seg.point.x + seg.handleIn.x,
              seg.point.y + seg.handleIn.y,
            ],
            end: [seg.point.x, seg.point.y],
          });
        }
      }
      fittedDump.push(dump);
    }

    const finalPath = composeFinalPath(contourPaths);

    const debugData: DebugResult = {
      final_path: finalPath.pathData,
      contours: contours.map((contour, ix) => ({
        contour_id: contour.contourId,
        points: contour.pointsRc.map((p) => [p[0], p[1]]),
        fitted: fittedDump[ix],
      })),
      nodes: nodes.map((n) => n.debugEntry()),
      segments: segments.map((segment) => {
        const startRc = segment.contour.pointsRc[segment.contourIndices[0]];
        const endRc =
          segment.contour.pointsRc[
            segment.contourIndices[segment.contourIndices.length - 1]
          ];
        return {
          segment_id: segment.segmentId,
          contour_id: segment.contour.contourId,
          start_node_id: segment.startNode.nodeId,
          end_node_id: segment.endNode.nodeId,
          kind: segment.kind,
          stage: "path_segment",
          num_points: segment.contourIndices.length,
          contour_indices: segment.contourIndices,
          start_point_rc: [startRc[0], startRc[1]],
          end_point_rc: [endRc[0], endRc[1]],
        };
      }),
    };

    return { finalPath, debugData };
  }
}

export { Contour } from "./contour.js";
export { Node } from "./node.js";
export type { DebugResult, PathSegment } from "./types.js";
