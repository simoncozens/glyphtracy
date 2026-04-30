export function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1];
}

export function norm(v: number[]): number {
  return Math.hypot(v[0], v[1]);
}

export function unit(v: number[]): number[] {
  const n = norm(v);
  if (n <= 1e-9) {
    return [1, 0];
  }
  return [v[0] / n, v[1] / n];
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const b = rhs.slice();

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
        pivot = row;
      }
    }

    if (Math.abs(a[pivot][col]) <= 1e-12) {
      continue;
    }

    if (pivot !== col) {
      [a[pivot], a[col]] = [a[col], a[pivot]];
      [b[pivot], b[col]] = [b[col], b[pivot]];
    }

    const div = a[col][col];
    for (let j = col; j < n; j += 1) {
      a[col][j] /= div;
    }
    b[col] /= div;

    for (let row = 0; row < n; row += 1) {
      if (row === col) {
        continue;
      }
      const factor = a[row][col];
      if (Math.abs(factor) <= 1e-12) {
        continue;
      }
      for (let j = col; j < n; j += 1) {
        a[row][j] -= factor * a[col][j];
      }
      b[row] -= factor * b[col];
    }
  }

  return b;
}

export function solveLeastSquares(a: number[][], b: number[]): number[] {
  if (a.length === 0) {
    return [];
  }
  const cols = a[0].length;
  const ata: number[][] = Array.from({ length: cols }, () =>
    Array.from({ length: cols }, () => 0),
  );
  const atb: number[] = Array.from({ length: cols }, () => 0);

  for (let r = 0; r < a.length; r += 1) {
    for (let i = 0; i < cols; i += 1) {
      atb[i] += a[r][i] * b[r];
      for (let j = 0; j < cols; j += 1) {
        ata[i][j] += a[r][i] * a[r][j];
      }
    }
  }

  for (let i = 0; i < cols; i += 1) {
    ata[i][i] += 1e-9;
  }

  return solveLinearSystem(ata, atb);
}
