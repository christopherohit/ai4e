// Multinomial Naive Bayes trained on the SMS Spam Collection (80% train split, seeded and stratified). Only word counts
// are stored; test-set metrics for each α come from scripts/ml-datasets/prepare.py and match scikit-learn exactly.
import data from './data/sms-spam.json';

export const SOURCE = data.source;
export const TRAIN = data.train, TEST = data.test;
export const METRICS = data.metrics;
export const CALIBRATION = data.calibration;
export const EXTREME = data.extreme;
export const VOCAB = data.vocab as [string, number, number][];
const INDEX = new Map(VOCAB.map(([word], i) => [word, i]));
export const TOTAL = { spam: VOCAB.reduce((s, v) => s + v[1], 0), ham: VOCAB.reduce((s, v) => s + v[2], 0) };
export const V = VOCAB.length;
export const PRIOR_SPAM = TRAIN.spam / (TRAIN.spam + TRAIN.ham);

const TOKEN = new RegExp(data.token, 'g');
export const tokenize = (text: string) => text.toLowerCase().match(TOKEN) ?? [];
export const known = (word: string) => INDEX.has(word);
export const counts = (word: string): [number, number] => { const i = INDEX.get(word); return i === undefined ? [0, 0] : [VOCAB[i][1], VOCAB[i][2]]; };

/** P(w | c) = (count + α) / (total words of c + α·|V|). */
export function likelihood(word: string, alpha: number): [number, number] {
  const [s, h] = counts(word);
  return [(s + alpha) / (TOTAL.spam + alpha * V), (h + alpha) / (TOTAL.ham + alpha * V)];
}
/** log P(w | spam) − log P(w | thường): the evidence one occurrence adds to the log-odds. */
export const logRatio = (word: string, alpha: number) => { const [s, h] = likelihood(word, alpha); return Math.log(s) - Math.log(h); };

export interface Step { word: string; known: boolean; delta: number; total: number }
/** Log-odds after the prior and after each token, in reading order. Unknown tokens add nothing. */
export function evidence(text: string, alpha: number) {
  const prior = Math.log(PRIOR_SPAM / (1 - PRIOR_SPAM));
  let total = prior;
  const steps: Step[] = tokenize(text).map(word => { const k = known(word), delta = k ? logRatio(word, alpha) : 0; total += delta; return { word, known: k, delta, total }; });
  return { prior, steps, logOdds: total, pSpam: 1 / (1 + Math.exp(-total)) };
}

/** “lt”, “gt”, “amp” come from HTML entities (&lt;#&gt;) left in the corpus, not from words people typed. */
const ARTIFACTS = new Set(['lt', 'gt', 'amp']);
/** Words that best separate the classes: largest |log ratio| among words seen at least `min` times. */
export function strongest(alpha: number, side: 'spam' | 'ham', count: number, min = 15) {
  return VOCAB.filter(([word, s, h]) => s + h >= min && !ARTIFACTS.has(word)).map(([word, s, h]) => ({ word, spam: s, ham: h, ratio: logRatio(word, alpha) }))
    .sort((a, b) => (side === 'spam' ? b.ratio - a.ratio : a.ratio - b.ratio)).slice(0, count);
}

export const EXAMPLES = [
  { name: 'Trúng thưởng', text: 'WINNER! You have won a guaranteed £1000 cash prize. To claim call now' },
  { name: 'Nhạc chuông', text: 'Free ringtone waiting for you, txt STOP to end' },
  { name: 'Hẹn về nhà', text: 'Ok lor, I will call you later when I get home' },
  { name: 'Lẫn lộn', text: 'Did you claim the prize at the office party? Call me later lol' },
];
