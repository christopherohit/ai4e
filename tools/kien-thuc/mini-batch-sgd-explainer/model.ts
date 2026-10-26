// Batch / mini-batch / stochastic gradient descent lab model: the same MSE problem and update rule, differing only in
// how many samples feed each gradient estimate. Runs are seeded so the article can describe what readers will see.
import { N, gradient, lossPerEpoch, train, type Point, type Run } from '../../_shared/optimization-lab/model';

export const BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64] as const;
export const START: Point = [-0.5, 2.6];

export type Variant = 'batch' | 'mini' | 'sgd';
export const batchSizeOf = (variant: Variant, mini: number) => (variant === 'batch' ? N : variant === 'sgd' ? 1 : mini);
export const updatesPerEpoch = (batchSize: number) => Math.ceil(N / batchSize);

export function run(variant: Variant, { lr, epochs, mini, seed }: { lr: number; epochs: number; mini: number; seed: number }): Run {
  return train({ batchSize: batchSizeOf(variant, mini), lr, epochs, start: START, seed });
}

export interface Comparison { variant: Variant; batchSize: number; run: Run; perEpoch: number[] }
export function compare(options: { lr: number; epochs: number; mini: number; seed: number }): Comparison[] {
  return (['batch', 'mini', 'sgd'] as const).map(variant => {
    const result = run(variant, options);
    return { variant, batchSize: batchSizeOf(variant, options.mini), run: result, perEpoch: lossPerEpoch(result, options.epochs) };
  });
}

/** Angle in degrees between a mini-batch gradient and the full gradient (0° = same direction). */
export function angle(estimate: Point, full: Point) {
  const norms = Math.hypot(...estimate) * Math.hypot(...full);
  if (norms < 1e-12) return 0;
  return Math.acos(Math.max(-1, Math.min(1, (estimate[0] * full[0] + estimate[1] * full[1]) / norms))) * 180 / Math.PI;
}

/** Share of mini-batch gradients at θ that point downhill (positive dot product with the full gradient). */
export function downhillShare(samples: Point[], theta: Point) {
  const full = gradient(theta);
  return samples.filter(g => g[0] * full[0] + g[1] * full[1] > 0).length / samples.length;
}
