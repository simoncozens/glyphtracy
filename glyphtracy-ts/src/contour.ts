import { AxisExtremaTag } from "./types.js";
import {
  axisProminentExtremaIndices,
  dedupeCyclicIndices,
  dedupeOpenIndices,
  globalAxisExtremaAxes,
  isBasicallyALine,
  normalized,
  pruneOpenSplitPositions,
  smoothSignal,
  spanIndicesClosed,
  splitOpenSpanIndices,
} from "./utils.js";

export class Contour {
  contourId: number;
  pointsRc: number[][];
  closed: boolean;

  constructor(contourId: number, pointsRc: number[][], closed = true) {
    this.contourId = contourId;
    this.pointsRc = pointsRc;
    this.closed = closed;
  }

  get size(): number {
    return this.pointsRc.length;
  }

  pointsForIndices(indices: number[]): number[][] {
    return indices.map((i) => this.pointsRc[i]);
  }

  cornerAngle(i: number, stride = 1): number {
    const n = this.size;
    if (n < 3) {
      return 0;
    }

    const prevI = this.closed ? (i - stride + n) % n : Math.max(0, i - stride);
    const nextI = this.closed ? (i + stride) % n : Math.min(n - 1, i + stride);
    if (prevI === i || nextI === i) {
      return 0;
    }

    const pPrev = this.pointsRc[prevI];
    const pCur = this.pointsRc[i];
    const pNext = this.pointsRc[nextI];

    const vin = normalized([pCur[0] - pPrev[0], pCur[1] - pPrev[1]]);
    const vout = normalized([pNext[0] - pCur[0], pNext[1] - pCur[1]]);
    const d = Math.max(-1, Math.min(1, vin[0] * vout[0] + vin[1] * vout[1]));
    return Math.acos(d);
  }

  iterativeAxisExtremaAxes(args: {
    minIndexGap?: number;
    minSpanPoints?: number;
    maxIterations?: number;
  }): Map<number, Set<AxisExtremaTag>> {
    const minIndexGap = args.minIndexGap ?? 3;
    const minSpanPoints = args.minSpanPoints ?? 8;
    const maxIterations = args.maxIterations ?? 8;

    const n = this.size;
    if (n < 3) {
      return new Map();
    }

    const seedMap = globalAxisExtremaAxes(
      this.pointsRc,
      Math.max(2, Math.floor(n / 80)),
    );
    const seedIndices = new Set<number>(seedMap.keys());

    for (let axis = 0; axis < 2; axis += 1) {
      const smoothed = smoothSignal(
        this.pointsRc.map((p) => p[axis]),
        9,
        true,
      );
      const axisTag: AxisExtremaTag = axis === 0 ? "y" : "x";
      const axisExtrema = axisProminentExtremaIndices(
        smoothed,
        Math.max(2, Math.min(8, Math.floor(n / 30))),
        1.5,
        true,
      );
      for (const idx of axisExtrema) {
        seedIndices.add(idx);
        if (!seedMap.has(idx)) {
          seedMap.set(idx, new Set());
        }
        seedMap.get(idx)?.add(axisTag);
      }
    }

    const splitIndices = dedupeCyclicIndices(
      [...seedIndices],
      n,
      Math.max(2, minIndexGap),
    );
    const discovered = new Map<number, Set<AxisExtremaTag>>();
    for (const idx of splitIndices) {
      discovered.set(idx, new Set(seedMap.get(idx) ?? []));
    }

    let spans: number[][];
    if (splitIndices.length < 2) {
      spans = [Array.from({ length: n }, (_, i) => i)];
    } else {
      spans = [];
      for (let i = 0; i < splitIndices.length; i += 1) {
        const start = splitIndices[i];
        const end = splitIndices[(i + 1) % splitIndices.length];
        const span = spanIndicesClosed(start, end, n);
        if (span.length >= 2) {
          spans.push(span);
        }
      }
      if (spans.length === 0) {
        spans = [Array.from({ length: n }, (_, i) => i)];
      }
    }

    const minSpan = Math.max(4, Math.floor(minSpanPoints));
    for (let it = 0; it < maxIterations; it += 1) {
      let didSplit = false;
      const nextSpans: number[][] = [];

      for (const spanIndices of spans) {
        if (spanIndices.length < Math.max(2 * minSpan, 6)) {
          nextSpans.push(spanIndices);
          continue;
        }

        const subcontour = this.pointsForIndices(spanIndices);
        const localCandidates = globalAxisExtremaAxes(
          subcontour,
          Math.max(2, minIndexGap),
        );

        for (let axis = 0; axis < 2; axis += 1) {
          const smoothed = smoothSignal(
            subcontour.map((p) => p[axis]),
            5,
            false,
          );
          const axisTag: AxisExtremaTag = axis === 0 ? "y" : "x";
          const axisExtrema = axisProminentExtremaIndices(
            smoothed,
            2,
            0.75,
            false,
          );
          for (const localIx of axisExtrema) {
            if (!localCandidates.has(localIx)) {
              localCandidates.set(localIx, new Set());
            }
            localCandidates.get(localIx)?.add(axisTag);
          }
        }

        let splitPositions = dedupeOpenIndices(
          [...localCandidates.keys()],
          spanIndices.length,
          Math.max(2, minIndexGap - 1),
        );
        splitPositions = pruneOpenSplitPositions(
          splitPositions,
          spanIndices.length,
          minSpan,
        );

        if (splitPositions.length === 0) {
          nextSpans.push(spanIndices);
          continue;
        }

        didSplit = true;
        for (const localIx of splitPositions) {
          const globalIx = spanIndices[localIx];
          if (!discovered.has(globalIx)) {
            discovered.set(globalIx, new Set());
          }
          const tags =
            localCandidates.get(localIx) ?? new Set<AxisExtremaTag>();
          for (const tag of tags) {
            discovered.get(globalIx)?.add(tag);
          }
        }

        nextSpans.push(...splitOpenSpanIndices(spanIndices, splitPositions));
      }

      spans = nextSpans;
      if (!didSplit) {
        break;
      }
    }

    return discovered;
  }

  iterativeAxisExtremaAxesOnOpenSpan(
    spanIndices: number[],
    args: {
      minIndexGap?: number;
      minSpanPoints?: number;
      maxIterations?: number;
    },
  ): Map<number, Set<AxisExtremaTag>> {
    const minIndexGap = args.minIndexGap ?? 3;
    const minSpanPoints = args.minSpanPoints ?? 8;
    const maxIterations = args.maxIterations ?? 8;

    if (spanIndices.length < 3) {
      return new Map();
    }

    const spanPoints = this.pointsForIndices(spanIndices);
    if (isBasicallyALine(spanPoints, 1.5)) {
      return new Map();
    }

    const localAll = Array.from({ length: spanIndices.length }, (_, i) => i);
    const discovered = new Map<number, Set<AxisExtremaTag>>();
    let spansLocal: number[][] = [localAll];
    const minSpan = Math.max(4, Math.floor(minSpanPoints));

    for (let it = 0; it < maxIterations; it += 1) {
      let didSplit = false;
      const nextLocalSpans: number[][] = [];

      for (const localSpan of spansLocal) {
        if (localSpan.length < Math.max(2 * minSpan, 6)) {
          nextLocalSpans.push(localSpan);
          continue;
        }

        const globalSpan = localSpan.map((ix) => spanIndices[ix]);
        const subcontour = this.pointsForIndices(globalSpan);
        const localCandidates = globalAxisExtremaAxes(
          subcontour,
          Math.max(2, minIndexGap),
        );

        for (let axis = 0; axis < 2; axis += 1) {
          const smoothed = smoothSignal(
            subcontour.map((p) => p[axis]),
            5,
            false,
          );
          const axisTag: AxisExtremaTag = axis === 0 ? "y" : "x";
          const axisExtrema = axisProminentExtremaIndices(
            smoothed,
            2,
            0.75,
            false,
          );
          for (const localIx of axisExtrema) {
            if (!localCandidates.has(localIx)) {
              localCandidates.set(localIx, new Set());
            }
            localCandidates.get(localIx)?.add(axisTag);
          }
        }

        let splitPositions = dedupeOpenIndices(
          [...localCandidates.keys()],
          localSpan.length,
          Math.max(2, minIndexGap - 1),
        );
        splitPositions = pruneOpenSplitPositions(
          splitPositions,
          localSpan.length,
          minSpan,
        );

        if (splitPositions.length === 0) {
          nextLocalSpans.push(localSpan);
          continue;
        }

        didSplit = true;
        for (const localIx of splitPositions) {
          const globalIx = spanIndices[localSpan[localIx]];
          if (!discovered.has(globalIx)) {
            discovered.set(globalIx, new Set());
          }
          const tags =
            localCandidates.get(localIx) ?? new Set<AxisExtremaTag>();
          for (const tag of tags) {
            discovered.get(globalIx)?.add(tag);
          }
        }

        nextLocalSpans.push(...splitOpenSpanIndices(localSpan, splitPositions));
      }

      spansLocal = nextLocalSpans;
      if (!didSplit) {
        break;
      }
    }

    return discovered;
  }
}
