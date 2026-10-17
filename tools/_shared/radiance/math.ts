export type RGB = [number, number, number];

export function composite(samples: { alpha: number; color: RGB }[], background: RGB = [0, 0, 0]) {
  let transmission = 1;
  const color: RGB = [0, 0, 0];
  const weights = samples.map(sample => {
    const weight = transmission * Math.max(0, Math.min(1, sample.alpha));
    color.forEach((_, channel) => { color[channel] += weight * sample.color[channel]; });
    transmission -= weight;
    return weight;
  });
  color.forEach((_, channel) => { color[channel] += transmission * background[channel]; });
  return { color, weights, transmission };
}

export function volumeAlpha(density: number, distance: number) {
  return -Math.expm1(-Math.max(0, density) * Math.max(0, distance));
}

export function projectedCovariance(point: number[], covariance: number[][], focal: number) {
  const [horizontal, vertical, depth] = point;
  if (depth <= 0) throw new Error('Point must be in front of the camera');
  const jacobian = [[focal / depth, 0, -focal * horizontal / depth ** 2], [0, focal / depth, -focal * vertical / depth ** 2]];
  return jacobian.map(row => jacobian.map(column => row.reduce((sum, value, index) => sum + value * covariance[index].reduce((inner, entry, axis) => inner + entry * column[axis], 0), 0)));
}

export function gaussianAlpha(offset: number[], covariance: number[][], opacity: number) {
  const [[first, cross], [, second]] = covariance;
  const determinant = first * second - cross * cross;
  if (determinant <= 0) return 0;
  const [horizontal, vertical] = offset;
  return Math.min(0.99, Math.max(0, opacity) * Math.exp(-0.5 * (second * horizontal ** 2 - 2 * cross * horizontal * vertical + first * vertical ** 2) / determinant));
}
