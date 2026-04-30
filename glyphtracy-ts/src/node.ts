import { clamp, dot, norm, unit } from "./math.js";
import { Contour } from "./contour.js";
import { AxisExtremaTag, Continuity, NodeTransition } from "./types.js";

export class Node {
  nodeId: number;
  contour: Contour;
  contourIndex: number;
  reasonTags: Set<string>;
  axisExtrema: Set<AxisExtremaTag>;
  transition: NodeTransition | null;
  continuity: Continuity | null;

  constructor(args: {
    nodeId: number;
    contour: Contour;
    contourIndex: number;
    reasonTags: Set<string>;
    axisExtrema?: Set<AxisExtremaTag>;
    transition?: NodeTransition | null;
    continuity?: Continuity | null;
  }) {
    this.nodeId = args.nodeId;
    this.contour = args.contour;
    this.contourIndex = args.contourIndex;
    this.reasonTags = args.reasonTags;
    this.axisExtrema = args.axisExtrema ?? new Set();
    this.transition = args.transition ?? null;
    this.continuity = args.continuity ?? null;
  }

  get pointRc(): number[] {
    return this.contour.pointsRc[this.contourIndex];
  }

  classifyContinuity(
    beforePointsRc: number[][],
    afterPointsRc: number[][],
  ): Continuity {
    if (this.transition !== "curve-curve") {
      return "non-continuous";
    }

    function weightedTangent(polyline: number[][]): number[] {
      if (polyline.length < 2) {
        return [1, 0];
      }
      const dirs: number[][] = [];
      for (let i = 0; i < polyline.length - 1; i += 1) {
        dirs.push(
          unit([
            polyline[i + 1][0] - polyline[i][0],
            polyline[i + 1][1] - polyline[i][1],
          ]),
        );
      }
      let vx = 0;
      let vy = 0;
      for (let i = 0; i < dirs.length; i += 1) {
        const w = 1 + i / Math.max(1, dirs.length - 1);
        vx += dirs[i][0] * w;
        vy += dirs[i][1] * w;
      }
      return unit([vx, vy]);
    }

    function localChain(polyline: number[][], fromEnd: boolean): number[][] {
      if (polyline.length <= 4) {
        return polyline;
      }

      const lengths: number[] = [];
      for (let i = 0; i < polyline.length - 1; i += 1) {
        lengths.push(
          norm([
            polyline[i + 1][0] - polyline[i][0],
            polyline[i + 1][1] - polyline[i][1],
          ]),
        );
      }
      const total = lengths.reduce((a, b) => a + b, 0);
      const target = Math.min(24, Math.max(8, 0.12 * total));

      if (fromEnd) {
        let acc = 0;
        let startIx = polyline.length - 2;
        while (startIx > 0 && acc < target) {
          acc += lengths[startIx];
          startIx -= 1;
        }
        return polyline.slice(startIx);
      }

      let acc = 0;
      let endIx = 1;
      while (endIx < polyline.length - 1 && acc < target) {
        acc += lengths[endIx - 1];
        endIx += 1;
      }
      return polyline.slice(0, endIx);
    }

    function signedCurvature(a: number[], b: number[], c: number[]): number {
      const ab = [b[0] - a[0], b[1] - a[1]];
      const bc = [c[0] - b[0], c[1] - b[1]];
      const ac = [c[0] - a[0], c[1] - a[1]];
      const abLen = norm(ab);
      const bcLen = norm(bc);
      const acLen = norm(ac);
      if (Math.min(abLen, bcLen, acLen) <= 1e-6) {
        return 0;
      }
      const cross = ab[0] * bc[1] - ab[1] * bc[0];
      return (2 * cross) / (abLen * bcLen * acLen);
    }

    if (beforePointsRc.length < 3 || afterPointsRc.length < 3) {
      return "non-continuous";
    }

    const beforeLocal = localChain(beforePointsRc, true);
    const afterLocal = localChain(afterPointsRc, false);
    if (beforeLocal.length < 3 || afterLocal.length < 3) {
      return "non-continuous";
    }

    const tanIn = weightedTangent(beforeLocal);
    const tanOut = weightedTangent(afterLocal);
    const turn = Math.acos(clamp(dot(tanIn, tanOut), -1, 1));
    if (turn > 0.35) {
      return "non-continuous";
    }

    const kIn = signedCurvature(
      beforeLocal[beforeLocal.length - 3],
      beforeLocal[beforeLocal.length - 2],
      beforeLocal[beforeLocal.length - 1],
    );
    const kOut = signedCurvature(afterLocal[0], afterLocal[1], afterLocal[2]);
    const kScale = Math.max(Math.abs(kIn), Math.abs(kOut), 1e-6);
    const kMatch = Math.abs(kIn - kOut) <= 0.06 + 0.35 * kScale;

    if (turn <= 0.14 && kMatch) {
      return "G2";
    }
    return "G1";
  }

  debugEntry(): Record<string, unknown> {
    return {
      node_id: this.nodeId,
      contour_id: this.contour.contourId,
      contour_index: this.contourIndex,
      point_rc: [this.pointRc[0], this.pointRc[1]],
      reasons: [...this.reasonTags].sort(),
      axis_extrema: [...this.axisExtrema].sort(),
      transition: this.transition,
      continuity: this.continuity,
    };
  }
}
