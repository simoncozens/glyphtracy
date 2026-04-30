import { AxisExtremaTag } from "./types.js";
import { norm } from "./math.js";

export function spanIndicesClosed(
  start: number,
  end: number,
  size: number,
): number[] {
  if (start <= end) {
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }
  const a = Array.from({ length: size - start }, (_, i) => start + i);
  const b = Array.from({ length: end + 1 }, (_, i) => i);
  return a.concat(b);
}

export function spansBetweenAnchorsClosed(
  anchorIndices: number[],
  size: number,
): number[][] {
  if (size < 2) {
    return [];
  }
  const anchors = [...new Set(anchorIndices)].sort((a, b) => a - b);
  if (anchors.length < 2) {
    return [Array.from({ length: size }, (_, i) => i)];
  }

  const spans: number[][] = [];
  for (let i = 0; i < anchors.length; i += 1) {
    const start = anchors[i];
    const end = anchors[(i + 1) % anchors.length];
    const span = spanIndicesClosed(start, end, size);
    if (span.length >= 2) {
      spans.push(span);
    }
  }

  return spans.length > 0 ? spans : [Array.from({ length: size }, (_, i) => i)];
}

export function dedupeClosedContour(contour: number[][]): number[][] {
  if (contour.length > 1) {
    const first = contour[0];
    const last = contour[contour.length - 1];
    if (
      Math.abs(first[0] - last[0]) <= 1e-6 &&
      Math.abs(first[1] - last[1]) <= 1e-6
    ) {
      return contour.slice(0, -1);
    }
  }
  return contour;
}

export function dedupeOpenIndices(
  indices: number[],
  size: number,
  minIndexGap: number,
): number[] {
  if (size < 3) {
    return [];
  }
  const selected: number[] = [];
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  for (const index of sorted) {
    if (index <= 0 || index >= size - 1) {
      continue;
    }
    if (selected.every((x) => Math.abs(x - index) >= minIndexGap)) {
      selected.push(index);
    }
  }
  return selected;
}

export function pruneOpenSplitPositions(
  splitPositions: number[],
  spanSize: number,
  minSpanPoints: number,
): number[] {
  const positions = [...new Set(splitPositions)].sort((a, b) => a - b);
  if (positions.length === 0) {
    return [];
  }

  const minPoints = Math.max(2, Math.floor(minSpanPoints));
  while (positions.length > 0) {
    const starts = [0].concat(positions);
    const ends = positions.concat([spanSize - 1]);
    const lengths = starts.map((start, i) => ends[i] - start + 1);
    const shortest = Math.min(...lengths);
    if (shortest >= minPoints) {
      break;
    }
    const shortestIx = lengths.indexOf(shortest);
    if (shortestIx === 0) {
      positions.shift();
    } else if (shortestIx === lengths.length - 1) {
      positions.pop();
    } else {
      positions.splice(shortestIx, 1);
    }
  }

  return positions;
}

export function splitOpenSpanIndices(
  spanIndices: number[],
  splitPositions: number[],
): number[][] {
  if (splitPositions.length === 0) {
    return [spanIndices];
  }
  const fragments: number[][] = [];
  const positions = [...new Set(splitPositions)].sort((a, b) => a - b);
  let start = 0;
  for (const splitPos of positions) {
    const fragment = spanIndices.slice(start, splitPos + 1);
    if (fragment.length >= 2) {
      fragments.push(fragment);
    }
    start = splitPos;
  }
  const tail = spanIndices.slice(start);
  if (tail.length >= 2) {
    fragments.push(tail);
  }
  return fragments.length > 0 ? fragments : [spanIndices];
}

export function globalAxisExtremaAxes(
  contour: number[][],
  shortPlateauMaxLen = 2,
): Map<number, Set<AxisExtremaTag>> {
  if (contour.length === 0) {
    return new Map();
  }

  function plateauRepresentatives(values: number[], target: number): number[] {
    const span = Math.max(...values) - Math.min(...values);
    const atol = Math.max(1e-9, span * 1e-6);
    const matches: number[] = [];
    for (let i = 0; i < values.length; i += 1) {
      if (Math.abs(values[i] - target) <= atol) {
        matches.push(i);
      }
    }

    if (matches.length === 0) {
      return [];
    }

    const runs: Array<[number, number]> = [];
    let start = matches[0];
    let end = start;
    for (let i = 1; i < matches.length; i += 1) {
      const idx = matches[i];
      if (idx === end + 1) {
        end = idx;
      } else {
        runs.push([start, end]);
        start = idx;
        end = idx;
      }
    }
    runs.push([start, end]);

    const reps: number[] = [];
    for (const [a, b] of runs) {
      const runLen = b - a + 1;
      if (runLen <= shortPlateauMaxLen) {
        reps.push(Math.floor((a + b) / 2));
      } else if (a === b) {
        reps.push(a);
      } else {
        reps.push(a, b);
      }
    }
    return reps;
  }

  const axisMap = new Map<number, Set<AxisExtremaTag>>();
  for (let axis = 0; axis < 2; axis += 1) {
    const axisValues = contour.map((p) => p[axis]);
    const minValue = Math.min(...axisValues);
    const maxValue = Math.max(...axisValues);
    const axisTag: AxisExtremaTag = axis === 0 ? "y" : "x";
    const reps = plateauRepresentatives(axisValues, minValue).concat(
      plateauRepresentatives(axisValues, maxValue),
    );
    for (const i of reps) {
      if (!axisMap.has(i)) {
        axisMap.set(i, new Set());
      }
      axisMap.get(i)?.add(axisTag);
    }
  }

  return axisMap;
}

export function cyclicDistance(a: number, b: number, size: number): number {
  const direct = Math.abs(a - b);
  return Math.min(direct, size - direct);
}

export function dedupeCyclicIndices(
  indices: number[],
  size: number,
  minIndexGap: number,
): number[] {
  if (indices.length === 0) {
    return [];
  }
  const selected: number[] = [];
  const sorted = [...new Set(indices)].sort((x, y) => x - y);
  for (const index of sorted) {
    const okay = selected.every(
      (existing) => cyclicDistance(index, existing, size) >= minIndexGap,
    );
    if (okay) {
      selected.push(index);
    }
  }

  while (
    selected.length > 1 &&
    cyclicDistance(selected[0], selected[selected.length - 1], size) <
      minIndexGap
  ) {
    selected.pop();
  }
  return selected;
}

export function normalized(v: number[]): number[] {
  const n = norm(v);
  if (n <= 1e-9) {
    return [1, 0];
  }
  return [v[0] / n, v[1] / n];
}

export function smoothCyclicSignal(values: number[], window: number): number[] {
  if (window <= 1 || values.length < 3) {
    return values.slice();
  }
  let w = window;
  if (w % 2 === 0) {
    w += 1;
  }
  const radius = Math.floor(w / 2);
  const kernel = Array.from({ length: w }, () => 1 / w);
  const padded: number[] = [];
  for (let i = values.length - radius; i < values.length; i += 1) {
    padded.push(values[(i + values.length) % values.length]);
  }
  padded.push(...values);
  for (let i = 0; i < radius; i += 1) {
    padded.push(values[i % values.length]);
  }

  const out: number[] = [];
  for (let i = 0; i + w <= padded.length; i += 1) {
    let acc = 0;
    for (let j = 0; j < w; j += 1) {
      acc += padded[i + j] * kernel[j];
    }
    out.push(acc);
  }
  return out;
}

export function smoothSignal(
  values: number[],
  window: number,
  closed: boolean,
): number[] {
  if (closed) {
    return smoothCyclicSignal(values, window);
  }
  if (window <= 1 || values.length < 3) {
    return values.slice();
  }
  let w = window;
  if (w % 2 === 0) {
    w += 1;
  }

  const radius = Math.floor(w / 2);
  const padded: number[] = [];
  for (let i = 0; i < radius; i += 1) {
    padded.push(values[0]);
  }
  padded.push(...values);
  for (let i = 0; i < radius; i += 1) {
    padded.push(values[values.length - 1]);
  }

  const out: number[] = [];
  for (let i = 0; i + w <= padded.length; i += 1) {
    let acc = 0;
    for (let j = 0; j < w; j += 1) {
      acc += padded[i + j] / w;
    }
    out.push(acc);
  }

  return out;
}

export function axisProminentExtremaIndices(
  values: number[],
  radius: number,
  minProminence: number,
  closed = true,
): number[] {
  const n = values.length;
  if (n < 3) {
    return Array.from({ length: n }, (_, i) => i);
  }
  const r = Math.max(1, Math.min(radius, Math.max(1, Math.floor(n / 4))));
  const indices: number[] = [];
  for (let i = 0; i < n; i += 1) {
    let local: number[];
    if (closed) {
      local = [];
      for (let off = -r; off <= r; off += 1) {
        local.push(values[(i + off + n) % n]);
      }
    } else {
      const lo = Math.max(0, i - r);
      const hi = Math.min(n - 1, i + r);
      local = values.slice(lo, hi + 1);
    }

    const v = values[i];
    const localMin = Math.min(...local);
    const localMax = Math.max(...local);
    const isLocalMax = local.every((x) => v >= x);
    const isLocalMin = local.every((x) => v <= x);

    const prominence = isLocalMax
      ? v - localMin
      : isLocalMin
        ? localMax - v
        : 0;

    if ((isLocalMax || isLocalMin) && prominence >= minProminence) {
      indices.push(i);
    }
  }
  return indices;
}

export function isBasicallyALine(
  contour: number[][],
  pixelTolerance = 1.5,
): boolean {
  if (contour.length < 2) {
    return true;
  }
  const start = contour[0];
  const end = contour[contour.length - 1];
  const vec = [end[0] - start[0], end[1] - start[1]];
  const n = norm(vec);
  if (n <= 1e-9) {
    return true;
  }

  const unitVec = [vec[0] / n, vec[1] / n];
  for (let i = 1; i < contour.length - 1; i += 1) {
    const pointVec = [contour[i][0] - start[0], contour[i][1] - start[1]];
    const dist = Math.abs(pointVec[0] * unitVec[1] - pointVec[1] * unitVec[0]);
    if (dist > pixelTolerance) {
      return false;
    }
  }
  return true;
}
