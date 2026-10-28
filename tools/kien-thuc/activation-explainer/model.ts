// Data and maths for the activation lab: two-moons points, networks trained offline (scripts/ml-datasets/
// prepare_nn_pca.py), affine maps and their composition, ReLU folding and the hand-written XOR network.
import data from './data/activation.json';

export type Vec = [number, number];
export type Mat = [[number, number], [number, number]];
export interface Affine { A: Mat; b: Vec }
export interface Net { act: 'tanh' | 'relu' | 'none'; hidden: number; W1: number[][]; b1: number[]; W2: number[][]; b2: number[]; testAccuracy: number; trainAccuracy: number; loss: number[]; snapshots?: (Omit<Net, 'act' | 'hidden' | 'testAccuracy' | 'trainAccuracy' | 'loss'> & { epoch: number; loss: number })[] }

export const POINTS = data.x as Vec[];
export const LABELS = data.y as number[];
export const LINEAR = data.linear as { w: Vec; b: number; testAccuracy: number; trainAccuracy: number };
export const NETS = data.models as unknown as Record<'tanh3' | 'relu8' | 'none8' | 'relu2', Net>;
export const SOURCE = data.source;

export const IDENTITY: Affine = { A: [[1, 0], [0, 1]], b: [0, 0] };
export const apply = ({ A, b }: Affine, [x, y]: Vec): Vec => [A[0][0] * x + A[0][1] * y + b[0], A[1][0] * x + A[1][1] * y + b[1]];
export const mix = (p: Affine, q: Affine, t: number): Affine => ({
  A: [[p.A[0][0] + (q.A[0][0] - p.A[0][0]) * t, p.A[0][1] + (q.A[0][1] - p.A[0][1]) * t], [p.A[1][0] + (q.A[1][0] - p.A[1][0]) * t, p.A[1][1] + (q.A[1][1] - p.A[1][1]) * t]],
  b: [p.b[0] + (q.b[0] - p.b[0]) * t, p.b[1] + (q.b[1] - p.b[1]) * t],
});
/** q ∘ p: first p, then q. */
export const compose = (q: Affine, p: Affine): Affine => ({
  A: [[q.A[0][0] * p.A[0][0] + q.A[0][1] * p.A[1][0], q.A[0][0] * p.A[0][1] + q.A[0][1] * p.A[1][1]], [q.A[1][0] * p.A[0][0] + q.A[1][1] * p.A[1][0], q.A[1][0] * p.A[0][1] + q.A[1][1] * p.A[1][1]]],
  b: apply({ A: q.A, b: q.b }, p.b),
});
export const rotation = (deg: number, scale: Vec = [1, 1], b: Vec = [0, 0]): Affine => {
  const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180);
  return { A: [[c * scale[0], -s * scale[1]], [s * scale[0], c * scale[1]]], b };
};
export const shear = (k: number, b: Vec = [0, 0]): Affine => ({ A: [[1, k], [0, 1]], b });

// Chapter 1: three affine steps applied in sequence.
export const STEPS: Affine[] = [rotation(35), compose(shear(0.7), rotation(35)), compose({ A: [[1.25, 0], [0, 0.6]], b: [0.25, -0.15] }, compose(shear(0.7), rotation(35)))];
// Chapter 2: two layers and their single-layer equivalent.
export const LAYER1: Affine = rotation(-28, [1.15, 0.8], [0.2, 0.1]);
export const LAYER2: Affine = { A: [[1, 0.9], [-0.35, 1]], b: [-0.3, 0.15] };
export const MERGED = compose(LAYER2, LAYER1);

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const relu = (z: number) => Math.max(0, z);
const act = (kind: Net['act'], z: number) => (kind === 'tanh' ? Math.tanh(z) : kind === 'relu' ? relu(z) : z);

export function hidden(net: Pick<Net, 'W1' | 'b1'> & { act: Net['act'] }, [x, y]: Vec) {
  return net.W1.map((w, i) => act(net.act, w[0] * x + w[1] * y + net.b1[i]));
}
export function logit(net: Pick<Net, 'W1' | 'b1' | 'W2' | 'b2'> & { act: Net['act'] }, p: Vec) {
  return hidden(net, p).reduce((s, h, i) => s + net.W2[0][i] * h, net.b2[0]);
}
export const linearLogit = ([x, y]: Vec) => LINEAR.w[0] * x + LINEAR.w[1] * y + LINEAR.b;
export function accuracy(score: (p: Vec) => number) {
  return POINTS.reduce((s, p, i) => s + ((score(p) > 0 ? 1 : 0) === LABELS[i] ? 1 : 0), 0) / POINTS.length;
}

/** A network with no activation is one affine map: W2 (W1 x + b1) + b2. */
export function collapse(net: Net) {
  const w: Vec = [0, 0];
  let b = net.b2[0];
  net.W1.forEach((row, i) => { w[0] += net.W2[0][i] * row[0]; w[1] += net.W2[0][i] * row[1]; b += net.W2[0][i] * net.b1[i]; });
  return { w, b };
}

// XOR with two ReLU units: h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 − 1), y = h1 − 2 h2.
export const XOR: { x: Vec; y: number }[] = [{ x: [0, 0], y: 0 }, { x: [1, 0], y: 1 }, { x: [0, 1], y: 1 }, { x: [1, 1], y: 0 }];
export const xorHidden = ([a, b]: Vec): Vec => [relu(a + b), relu(a + b - 1)];
export const xorOut = (h: Vec) => h[0] - 2 * h[1];

export type ModelKey = 'linear' | 'none8' | 'relu2' | 'relu8' | 'tanh3';
export const MODELS: { key: ModelKey; label: string; note: string }[] = [
  { key: 'linear', label: 'Hồi quy logistic', note: 'một đường thẳng' },
  { key: 'none8', label: '8 nơron, không kích hoạt', note: 'gộp lại thành một đường thẳng' },
  { key: 'relu2', label: '2 nơron ReLU', note: 'gập hai lần' },
  { key: 'relu8', label: '8 nơron ReLU', note: 'nhiều nếp gập' },
  { key: 'tanh3', label: '3 nơron tanh', note: 'uốn mềm' },
];
export const scoreOf = (key: ModelKey) => (key === 'linear' ? linearLogit : (p: Vec) => logit(NETS[key], p));
export const testAccuracyOf = (key: ModelKey) => (key === 'linear' ? LINEAR.testAccuracy : NETS[key].testAccuracy);
