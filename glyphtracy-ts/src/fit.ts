import paper from "paper";
import { clamp, dot, norm, solveLeastSquares, unit } from "./math.js";

interface FittedSegment {
  cubic: Cubic;
  maxError: number;
}

interface Cubic {
  p0: number[];
  p1: number[];
  p2: number[];
  p3: number[];
}

function asXy(pointRc: number[]): number[] {
  return [pointRc[1], pointRc[0]];
}

function signWithDefault(value: number, defaultValue = 1): number {
  if (Math.abs(value) <= 1e-9) {
    return defaultValue;
  }
  return value > 0 ? 1 : -1;
}

function constrainTangentForAxisExtrema(
  tangentXy: number[],
  axisExtrema: Set<string>,
): number[] {
  if (axisExtrema.size === 0) {
    return unit(tangentXy);
  }

  if (axisExtrema.has("x") && !axisExtrema.has("y")) {
    return [0, signWithDefault(tangentXy[1])];
  }
  if (axisExtrema.has("y") && !axisExtrema.has("x")) {
    return [signWithDefault(tangentXy[0]), 0];
  }
  return unit(tangentXy);
}

function chordTValues(pointsXy: number[][]): number[] {
  if (pointsXy.length <= 1) {
    return [0, 1];
  }
  const seg: number[] = [];
  for (let i = 0; i < pointsXy.length - 1; i += 1) {
    seg.push(
      norm([
        pointsXy[i + 1][0] - pointsXy[i][0],
        pointsXy[i + 1][1] - pointsXy[i][1],
      ]),
    );
  }
  const total = seg.reduce((a, b) => a + b, 0);
  if (total <= 1e-9) {
    return Array.from(
      { length: pointsXy.length },
      (_, i) => i / Math.max(1, pointsXy.length - 1),
    );
  }

  const out = [0];
  let acc = 0;
  for (const s of seg) {
    acc += s;
    out.push(acc / total);
  }
  return out;
}

function evalCubic(cubic: Cubic, t: number): number[] {
  const mt = 1 - t;
  const b0 = mt * mt * mt;
  const b1 = 3 * mt * mt * t;
  const b2 = 3 * mt * t * t;
  const b3 = t * t * t;
  return [
    b0 * cubic.p0[0] + b1 * cubic.p1[0] + b2 * cubic.p2[0] + b3 * cubic.p3[0],
    b0 * cubic.p0[1] + b1 * cubic.p1[1] + b2 * cubic.p2[1] + b3 * cubic.p3[1],
  ];
}

function fitSingleCubicWithTangents(
  pointsXy: number[][],
  tanStart: number[],
  tanEnd: number[],
  args: { segmentBalanceWeight: number; handleShrinkWeight: number },
): Cubic {
  const p0 = pointsXy[0];
  const p3 = pointsXy[pointsXy.length - 1];
  const t0 = unit(tanStart);
  const t3 = unit(tanEnd);
  const tValues = chordTValues(pointsXy);

  const aRows: number[][] = [];
  const bRows: number[] = [];

  for (let i = 1; i < pointsXy.length - 1; i += 1) {
    const ti = tValues[i];
    const q = pointsXy[i];
    const mt = 1 - ti;
    const b1 = 3 * mt * mt * ti;
    const b2 = 3 * mt * ti * ti;
    const constX = (mt * mt * mt + b1) * p0[0] + (ti * ti * ti + b2) * p3[0];
    const constY = (mt * mt * mt + b1) * p0[1] + (ti * ti * ti + b2) * p3[1];

    aRows.push([b1 * t0[0], -b2 * t3[0]]);
    bRows.push(q[0] - constX);
    aRows.push([b1 * t0[1], -b2 * t3[1]]);
    bRows.push(q[1] - constY);
  }

  if (args.segmentBalanceWeight > 0) {
    const w = Math.sqrt(args.segmentBalanceWeight);
    aRows.push([w, -w]);
    bRows.push(0);
  }

  if (args.handleShrinkWeight > 0) {
    const w = Math.sqrt(args.handleShrinkWeight);
    aRows.push([w, 0]);
    bRows.push(0);
    aRows.push([0, w]);
    bRows.push(0);
  }

  let alpha = 0;
  let beta = 0;
  if (aRows.length > 0) {
    const x = solveLeastSquares(aRows, bRows);
    alpha = Math.max(0, x[0] ?? 0);
    beta = Math.max(0, x[1] ?? 0);
  }

  const chord = norm([p3[0] - p0[0], p3[1] - p0[1]]);
  const maxLen = Math.max(1, 3 * chord);
  alpha = Math.min(alpha, maxLen);
  beta = Math.min(beta, maxLen);

  const c1 = [p0[0] + alpha * t0[0], p0[1] + alpha * t0[1]];
  const c2 = [p3[0] - beta * t3[0], p3[1] - beta * t3[1]];
  return { p0, p1: c1, p2: c2, p3 };
}

function fitSingleCubicWithPartialConstraints(
  pointsXy: number[][],
  tanStart: number[],
  tanEnd: number[],
  args: {
    freeStartHandle: boolean;
    freeEndHandle: boolean;
    segmentBalanceWeight: number;
    handleShrinkWeight: number;
  },
): Cubic {
  const p0 = pointsXy[0];
  const p3 = pointsXy[pointsXy.length - 1];
  const t0 = unit(tanStart);
  const t3 = unit(tanEnd);
  const tValues = chordTValues(pointsXy);

  let nextVar = 0;
  let iAlpha = -1;
  let iBeta = -1;
  let iC1x = -1;
  let iC1y = -1;
  let iC2x = -1;
  let iC2y = -1;

  if (args.freeStartHandle) {
    iC1x = nextVar;
    iC1y = nextVar + 1;
    nextVar += 2;
  } else {
    iAlpha = nextVar;
    nextVar += 1;
  }

  if (args.freeEndHandle) {
    iC2x = nextVar;
    iC2y = nextVar + 1;
    nextVar += 2;
  } else {
    iBeta = nextVar;
    nextVar += 1;
  }

  const nVars = nextVar;
  const rows: number[][] = [];
  const rhs: number[] = [];

  for (let i = 1; i < pointsXy.length - 1; i += 1) {
    const ti = tValues[i];
    const q = pointsXy[i];
    const mt = 1 - ti;
    const b0 = mt * mt * mt;
    const b1 = 3 * mt * mt * ti;
    const b2 = 3 * mt * ti * ti;
    const b3 = ti * ti * ti;

    let constX = b0 * p0[0] + b3 * p3[0];
    let constY = b0 * p0[1] + b3 * p3[1];
    if (!args.freeStartHandle) {
      constX += b1 * p0[0];
      constY += b1 * p0[1];
    }
    if (!args.freeEndHandle) {
      constX += b2 * p3[0];
      constY += b2 * p3[1];
    }

    const rowX = Array.from({ length: nVars }, () => 0);
    const rowY = Array.from({ length: nVars }, () => 0);

    if (args.freeStartHandle) {
      rowX[iC1x] = b1;
      rowY[iC1y] = b1;
    } else {
      rowX[iAlpha] = b1 * t0[0];
      rowY[iAlpha] = b1 * t0[1];
    }

    if (args.freeEndHandle) {
      rowX[iC2x] = b2;
      rowY[iC2y] = b2;
    } else {
      rowX[iBeta] = -b2 * t3[0];
      rowY[iBeta] = -b2 * t3[1];
    }

    rows.push(rowX);
    rhs.push(q[0] - constX);
    rows.push(rowY);
    rhs.push(q[1] - constY);
  }

  if (
    args.segmentBalanceWeight > 0 &&
    !args.freeStartHandle &&
    !args.freeEndHandle
  ) {
    const w = Math.sqrt(args.segmentBalanceWeight);
    const row = Array.from({ length: nVars }, () => 0);
    row[iAlpha] = w;
    row[iBeta] = -w;
    rows.push(row);
    rhs.push(0);
  }

  if (args.handleShrinkWeight > 0) {
    const w = Math.sqrt(args.handleShrinkWeight);

    if (args.freeStartHandle) {
      const rowX = Array.from({ length: nVars }, () => 0);
      rowX[iC1x] = w;
      rows.push(rowX);
      rhs.push(w * p0[0]);

      const rowY = Array.from({ length: nVars }, () => 0);
      rowY[iC1y] = w;
      rows.push(rowY);
      rhs.push(w * p0[1]);
    } else {
      const row = Array.from({ length: nVars }, () => 0);
      row[iAlpha] = w;
      rows.push(row);
      rhs.push(0);
    }

    if (args.freeEndHandle) {
      const rowX = Array.from({ length: nVars }, () => 0);
      rowX[iC2x] = w;
      rows.push(rowX);
      rhs.push(w * p3[0]);

      const rowY = Array.from({ length: nVars }, () => 0);
      rowY[iC2y] = w;
      rows.push(rowY);
      rhs.push(w * p3[1]);
    } else {
      const row = Array.from({ length: nVars }, () => 0);
      row[iBeta] = w;
      rows.push(row);
      rhs.push(0);
    }
  }

  let c1 = p0.slice();
  let c2 = p3.slice();
  if (rows.length > 0) {
    const x = solveLeastSquares(rows, rhs);

    if (args.freeStartHandle) {
      c1 = [x[iC1x], x[iC1y]];
    } else {
      const alpha = Math.max(0, x[iAlpha] ?? 0);
      c1 = [p0[0] + alpha * t0[0], p0[1] + alpha * t0[1]];
    }

    if (args.freeEndHandle) {
      c2 = [x[iC2x], x[iC2y]];
    } else {
      const beta = Math.max(0, x[iBeta] ?? 0);
      c2 = [p3[0] - beta * t3[0], p3[1] - beta * t3[1]];
    }
  }

  return { p0, p1: c1, p2: c2, p3 };
}

function fitSingleCubicUnconstrained(
  pointsXy: number[][],
  args: { handleShrinkWeight: number },
): Cubic {
  return fitSingleCubicWithPartialConstraints(pointsXy, [1, 0], [1, 0], {
    freeStartHandle: true,
    freeEndHandle: true,
    segmentBalanceWeight: 0,
    handleShrinkWeight: args.handleShrinkWeight,
  });
}

function segmentMaxError(cubic: Cubic, pointsXy: number[][]): [number, number] {
  const tValues = chordTValues(pointsXy);
  let maxError = 0;
  let maxIndex = 0;
  for (let i = 0; i < pointsXy.length; i += 1) {
    const p = evalCubic(cubic, tValues[i]);
    const err = norm([p[0] - pointsXy[i][0], p[1] - pointsXy[i][1]]);
    if (err > maxError) {
      maxError = err;
      maxIndex = i;
    }
  }
  return [maxError, maxIndex];
}

function splitAndRefit(
  pointsXy: number[][],
  tanStart: number[],
  tanEnd: number[],
  args: {
    tolerance: number;
    maxDepth: number;
    segmentBalanceWeight: number;
    handleShrinkWeight: number;
    unconstrainedHandles: boolean;
    freeStartHandle: boolean;
    freeEndHandle: boolean;
  },
): FittedSegment[] {
  let cubic: Cubic;
  if (args.unconstrainedHandles) {
    cubic = fitSingleCubicUnconstrained(pointsXy, {
      handleShrinkWeight: args.handleShrinkWeight,
    });
  } else if (args.freeStartHandle || args.freeEndHandle) {
    cubic = fitSingleCubicWithPartialConstraints(pointsXy, tanStart, tanEnd, {
      freeStartHandle: args.freeStartHandle,
      freeEndHandle: args.freeEndHandle,
      segmentBalanceWeight: args.segmentBalanceWeight,
      handleShrinkWeight: args.handleShrinkWeight,
    });
  } else {
    cubic = fitSingleCubicWithTangents(pointsXy, tanStart, tanEnd, {
      segmentBalanceWeight: args.segmentBalanceWeight,
      handleShrinkWeight: args.handleShrinkWeight,
    });
  }
  const [maxError, splitIxRaw] = segmentMaxError(cubic, pointsXy);

  if (maxError <= args.tolerance || args.maxDepth <= 0 || pointsXy.length < 8) {
    return [{ cubic, maxError }];
  }

  const splitIx = Math.max(2, Math.min(splitIxRaw, pointsXy.length - 3));
  const left = pointsXy.slice(0, splitIx + 1);
  const right = pointsXy.slice(splitIx);
  const splitTangent = unit([
    pointsXy[splitIx + 1][0] - pointsXy[splitIx - 1][0],
    pointsXy[splitIx + 1][1] - pointsXy[splitIx - 1][1],
  ]);

  const leftFitted = splitAndRefit(left, tanStart, splitTangent, {
    ...args,
    maxDepth: args.maxDepth - 1,
    freeEndHandle: false,
  });
  const rightFitted = splitAndRefit(right, splitTangent, tanEnd, {
    ...args,
    maxDepth: args.maxDepth - 1,
    freeStartHandle: false,
  });
  return leftFitted.concat(rightFitted);
}

function globalHandleLengths(
  nodePointsXy: number[][],
  nodeTangents: number[][],
  segmentPointsXy: number[][][],
  continuities: Array<string | null>,
  args: {
    nodeBalanceWeight: number;
    segmentBalanceWeight: number;
    g2Weight: number;
    handleShrinkWeight: number;
  },
): [number[], number[]] {
  const nNodes = nodePointsXy.length;
  if (nNodes < 2) {
    return [[], []];
  }

  const nVars = 2 * nNodes;
  const outOff = 0;
  const inOff = nNodes;
  const rows: number[][] = [];
  const rhs: number[] = [];

  for (let si = 0; si < segmentPointsXy.length; si += 1) {
    const pointsXy = segmentPointsXy[si];
    if (pointsXy.length < 3) {
      continue;
    }

    const p0 = nodePointsXy[si];
    const p3 = nodePointsXy[si + 1];
    const t0 = nodeTangents[si];
    const t3 = nodeTangents[si + 1];
    const tValues = chordTValues(pointsXy);

    for (let i = 1; i < pointsXy.length - 1; i += 1) {
      const ti = tValues[i];
      const q = pointsXy[i];
      const mt = 1 - ti;
      const b1 = 3 * mt * mt * ti;
      const b2 = 3 * mt * ti * ti;
      const constX = (mt * mt * mt + b1) * p0[0] + (ti * ti * ti + b2) * p3[0];
      const constY = (mt * mt * mt + b1) * p0[1] + (ti * ti * ti + b2) * p3[1];

      const rowX = Array.from({ length: nVars }, () => 0);
      const rowY = Array.from({ length: nVars }, () => 0);
      rowX[outOff + si] = b1 * t0[0];
      rowY[outOff + si] = b1 * t0[1];
      rowX[inOff + si + 1] = -b2 * t3[0];
      rowY[inOff + si + 1] = -b2 * t3[1];
      rows.push(rowX);
      rhs.push(q[0] - constX);
      rows.push(rowY);
      rhs.push(q[1] - constY);
    }
  }

  if (args.segmentBalanceWeight > 0) {
    const w = Math.sqrt(args.segmentBalanceWeight);
    for (let si = 0; si < nNodes - 1; si += 1) {
      const row = Array.from({ length: nVars }, () => 0);
      row[outOff + si] = w;
      row[inOff + si + 1] = -w;
      rows.push(row);
      rhs.push(0);
    }
  }

  if (args.nodeBalanceWeight > 0) {
    const w = Math.sqrt(args.nodeBalanceWeight);
    for (let ni = 1; ni < nNodes - 1; ni += 1) {
      const row = Array.from({ length: nVars }, () => 0);
      row[inOff + ni] = w;
      row[outOff + ni] = -w;
      rows.push(row);
      rhs.push(0);
    }
  }

  if (args.g2Weight > 0) {
    const w = Math.sqrt(args.g2Weight);
    for (let ni = 1; ni < nNodes - 1; ni += 1) {
      if (continuities[ni] !== "G2") {
        continue;
      }
      const row = Array.from({ length: nVars }, () => 0);
      row[inOff + ni] = w;
      row[outOff + ni] = -w;
      rows.push(row);
      rhs.push(0);
    }
  }

  if (args.handleShrinkWeight > 0) {
    const w = Math.sqrt(args.handleShrinkWeight);
    for (let vi = 0; vi < nVars; vi += 1) {
      const row = Array.from({ length: nVars }, () => 0);
      row[vi] = w;
      rows.push(row);
      rhs.push(0);
    }
  }

  if (rows.length === 0) {
    return [
      Array.from({ length: nNodes }, () => 0),
      Array.from({ length: nNodes }, () => 0),
    ];
  }

  const x = solveLeastSquares(rows, rhs);
  const outLen = Array.from({ length: nNodes }, (_, i) =>
    Math.max(0, x[outOff + i] ?? 0),
  );
  const inLen = Array.from({ length: nNodes }, (_, i) =>
    Math.max(0, x[inOff + i] ?? 0),
  );

  for (let si = 0; si < nNodes - 1; si += 1) {
    const chord = norm([
      nodePointsXy[si + 1][0] - nodePointsXy[si][0],
      nodePointsXy[si + 1][1] - nodePointsXy[si][1],
    ]);
    const maxLen = Math.max(1, 3 * chord);
    outLen[si] = Math.min(outLen[si], maxLen);
    inLen[si + 1] = Math.min(inLen[si + 1], maxLen);
  }

  return [outLen, inLen];
}

function nodeTangentsFromRun(
  nodePointsXy: number[][],
  continuities: Array<string | null>,
  axisExtremaByNode: Array<Set<string>>,
): number[][] {
  const n = nodePointsXy.length;
  const tangents: number[][] = [];
  for (let i = 0; i < n; i += 1) {
    if (i === 0) {
      tangents.push(
        unit([
          nodePointsXy[1][0] - nodePointsXy[0][0],
          nodePointsXy[1][1] - nodePointsXy[0][1],
        ]),
      );
      continue;
    }
    if (i === n - 1) {
      tangents.push(
        unit([
          nodePointsXy[n - 1][0] - nodePointsXy[n - 2][0],
          nodePointsXy[n - 1][1] - nodePointsXy[n - 2][1],
        ]),
      );
      continue;
    }

    const prevVec = unit([
      nodePointsXy[i][0] - nodePointsXy[i - 1][0],
      nodePointsXy[i][1] - nodePointsXy[i - 1][1],
    ]);
    const nextVec = unit([
      nodePointsXy[i + 1][0] - nodePointsXy[i][0],
      nodePointsXy[i + 1][1] - nodePointsXy[i][1],
    ]);

    const continuity = continuities[i];
    const base =
      continuity === "G1" || continuity === "G2"
        ? unit([prevVec[0] + nextVec[0], prevVec[1] + nextVec[1]])
        : nextVec;
    tangents.push(base);
  }

  return tangents.map((t, i) =>
    constrainTangentForAxisExtrema(t, axisExtremaByNode[i]),
  );
}

function cubicToPath(cubic: Cubic): paper.Path {
  const path = new paper.Path();
  path.moveTo(new paper.Point(cubic.p0[0], cubic.p0[1]));
  path.cubicCurveTo(
    new paper.Point(cubic.p1[0], cubic.p1[1]),
    new paper.Point(cubic.p2[0], cubic.p2[1]),
    new paper.Point(cubic.p3[0], cubic.p3[1]),
  );
  return path;
}

export function fitCurveRun(
  contourPointsRc: number[][],
  runNodes: Array<{
    contourIndex: number;
    continuity: string | null;
    axisExtrema: Set<string>;
    reasonTags?: Set<string>;
  }>,
  runSpans: number[][],
  args: {
    tolerance: number;
    nodeBalanceWeight: number;
    segmentBalanceWeight: number;
    g2Weight: number;
    handleShrinkWeight: number;
    maxSplitDepth: number;
  },
): paper.Path[] {
  if (runNodes.length < 2 || runSpans.length !== runNodes.length - 1) {
    return [];
  }

  function isUnconstrainedBoundaryNode(node: {
    reasonTags?: Set<string>;
    continuity: string | null;
  }): boolean {
    const reasons = new Set(node.reasonTags ?? []);
    return reasons.has("sharp_corner") || node.continuity === "non-continuous";
  }

  const nodePointsXy = runNodes.map((node) =>
    asXy(contourPointsRc[node.contourIndex]),
  );
  const continuities = runNodes.map((node) => node.continuity);
  const axisExtremaByNode = runNodes.map((node) => new Set(node.axisExtrema));
  const nodeTangents = nodeTangentsFromRun(
    nodePointsXy,
    continuities,
    axisExtremaByNode,
  );

  const segmentPointsXy: number[][][] = runSpans.map((span) =>
    span.map((ix) => asXy(contourPointsRc[ix])),
  );
  const [outLen, inLen] = globalHandleLengths(
    nodePointsXy,
    nodeTangents,
    segmentPointsXy,
    continuities,
    {
      nodeBalanceWeight: args.nodeBalanceWeight,
      segmentBalanceWeight: args.segmentBalanceWeight,
      g2Weight: args.g2Weight,
      handleShrinkWeight: args.handleShrinkWeight,
    },
  );

  const paths: paper.Path[] = [];
  const boundaryStartFree = isUnconstrainedBoundaryNode(runNodes[0]);
  const boundaryEndFree = isUnconstrainedBoundaryNode(
    runNodes[runNodes.length - 1],
  );
  const unconstrainedHandles =
    args.nodeBalanceWeight <= 0 &&
    args.segmentBalanceWeight <= 0 &&
    args.g2Weight <= 0;

  for (let si = 0; si < segmentPointsXy.length; si += 1) {
    const pointsXy = segmentPointsXy[si];
    const t0 = nodeTangents[si];
    const t3 = nodeTangents[si + 1];
    const p0 = nodePointsXy[si];
    const p3 = nodePointsXy[si + 1];

    const segmentFreeStart =
      !unconstrainedHandles && si === 0 && boundaryStartFree;
    const segmentFreeEnd =
      !unconstrainedHandles &&
      si === segmentPointsXy.length - 1 &&
      boundaryEndFree;

    let seedCubic: Cubic;
    if (unconstrainedHandles) {
      seedCubic = fitSingleCubicUnconstrained(pointsXy, {
        handleShrinkWeight: args.handleShrinkWeight,
      });
    } else if (segmentFreeStart || segmentFreeEnd) {
      seedCubic = fitSingleCubicWithPartialConstraints(pointsXy, t0, t3, {
        freeStartHandle: segmentFreeStart,
        freeEndHandle: segmentFreeEnd,
        segmentBalanceWeight: args.segmentBalanceWeight,
        handleShrinkWeight: args.handleShrinkWeight,
      });
    } else {
      seedCubic = {
        p0,
        p1: [p0[0] + outLen[si] * t0[0], p0[1] + outLen[si] * t0[1]],
        p2: [p3[0] - inLen[si + 1] * t3[0], p3[1] - inLen[si + 1] * t3[1]],
        p3,
      };
    }

    const [maxError] = segmentMaxError(seedCubic, pointsXy);
    const fitted =
      maxError <= args.tolerance
        ? [{ cubic: seedCubic, maxError }]
        : splitAndRefit(pointsXy, t0, t3, {
            tolerance: args.tolerance,
            maxDepth: args.maxSplitDepth,
            segmentBalanceWeight: args.segmentBalanceWeight,
            handleShrinkWeight: args.handleShrinkWeight,
            unconstrainedHandles,
            freeStartHandle: segmentFreeStart,
            freeEndHandle: segmentFreeEnd,
          });

    const path = new paper.Path();
    const first = fitted[0].cubic;
    path.moveTo(new paper.Point(first.p0[0], first.p0[1]));
    for (const frag of fitted) {
      path.cubicCurveTo(
        new paper.Point(frag.cubic.p1[0], frag.cubic.p1[1]),
        new paper.Point(frag.cubic.p2[0], frag.cubic.p2[1]),
        new paper.Point(frag.cubic.p3[0], frag.cubic.p3[1]),
      );
    }
    paths.push(path);
  }

  return paths;
}
