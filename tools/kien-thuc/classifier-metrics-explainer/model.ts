// Classifier-evaluation lab: held-out logistic-regression scores on Breast Cancer Wisconsin (Diagnostic), baked by
// scripts/ml-datasets/prepare_stats.py. Every metric is recomputed here from the scores and labels.
import data from './data/metrics.json';

export const D = data as { source: string; positive: string; negative: string; scores: number[]; labels: number[]; rocAuc: number; prAuc: number; trainSize: number; testSize: number };
export const POS = D.labels.filter(y => y === 1).length;
export const NEG = D.labels.length - POS;

export function confusion(threshold: number, scores = D.scores, labels = D.labels) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  scores.forEach((s, i) => { const p = s >= threshold; if (p && labels[i]) tp++; else if (p) fp++; else if (labels[i]) fn++; else tn++; });
  return { tp, fp, tn, fn };
}
export function metrics(threshold: number) {
  const { tp, fp, tn, fn } = confusion(threshold);
  const precision = tp + fp ? tp / (tp + fp) : 1, recall = tp / (tp + fn), specificity = tn / (tn + fp);
  return { tp, fp, tn, fn, accuracy: (tp + tn) / (tp + fp + tn + fn), precision, recall, specificity, fpr: 1 - specificity, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 };
}
/** ROC points sorted by decreasing threshold, and the trapezoid AUC. */
export function roc() {
  const order = D.scores.map((s, i) => [s, D.labels[i]] as const).sort((a, b) => b[0] - a[0]);
  const pts: [number, number][] = [[0, 0]];
  let tp = 0, fp = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i][1]) tp++; else fp++;
    if (i === order.length - 1 || order[i + 1][0] !== order[i][0]) pts.push([fp / NEG, tp / POS]);
  }
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2;
  return { pts, auc };
}
/** Precision–recall points and average precision (step-wise, as scikit-learn computes it). */
export function pr() {
  const order = D.scores.map((s, i) => [s, D.labels[i]] as const).sort((a, b) => b[0] - a[0]);
  const pts: [number, number][] = [];
  let tp = 0, fp = 0, ap = 0, prevRecall = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i][1]) tp++; else fp++;
    if (i === order.length - 1 || order[i + 1][0] !== order[i][0]) {
      const r = tp / POS, p = tp / (tp + fp);
      ap += (r - prevRecall) * p; prevRecall = r;
      pts.push([r, p]);
    }
  }
  return { pts, ap };
}
/** Precision when the same sensitivity/specificity meet a different prevalence (Bayes' rule). */
export const precisionAt = (recall: number, specificity: number, prevalence: number) => recall * prevalence / (recall * prevalence + (1 - specificity) * (1 - prevalence));
