// Multinomial Naive Bayes spam filter on 16 hand-written Vietnamese emails and a 10-word vocabulary. Small enough that
// every count, smoothed probability and log-score in the article can be checked by hand.
export const CLASSES = ['spam', 'thường'] as const;
export type Label = 0 | 1;

export const VOCAB = ['miễn phí', 'trúng', 'thưởng', 'khuyến mãi', 'click', 'chuyển khoản', 'họp', 'báo cáo', 'dự án', 'cảm ơn'] as const;
export type Word = typeof VOCAB[number];

export interface Email { text: string; words: Word[]; label: Label }
const mail = (label: Label, text: string, ...words: Word[]): Email => ({ text, words, label });

export const TRAIN: Email[] = [
  mail(0, 'Miễn phí! Click nhận quà khuyến mãi', 'miễn phí', 'click', 'khuyến mãi'),
  mail(0, 'Bạn đã trúng thưởng, click để nhận', 'trúng', 'thưởng', 'click'),
  mail(0, 'Trúng thưởng 100 triệu, chuyển khoản phí để nhận', 'trúng', 'thưởng', 'chuyển khoản'),
  mail(0, 'Khuyến mãi miễn phí vận chuyển, click ngay', 'khuyến mãi', 'miễn phí', 'click'),
  mail(0, 'Chúc mừng trúng thưởng iPhone miễn phí', 'trúng', 'thưởng', 'miễn phí'),
  mail(0, 'Chuyển khoản ngay để giữ khuyến mãi', 'chuyển khoản', 'khuyến mãi'),
  mail(1, 'Mời họp dự án lúc 9h', 'họp', 'dự án'),
  mail(1, 'Gửi báo cáo dự án tuần này', 'báo cáo', 'dự án'),
  mail(1, 'Cảm ơn anh đã gửi báo cáo', 'cảm ơn', 'báo cáo'),
  mail(1, 'Lịch họp: báo cáo tiến độ dự án', 'họp', 'báo cáo', 'dự án'),
  mail(1, 'Đã chuyển khoản tiền ăn trưa, cảm ơn', 'chuyển khoản', 'cảm ơn'),
  mail(1, 'Link tài liệu dự án, click để xem', 'dự án', 'click'),
  mail(1, 'Cảm ơn mọi người đã dự họp', 'cảm ơn', 'họp'),
  mail(1, 'Khuyến mãi căng tin: cơm miễn phí cho đội dự án', 'khuyến mãi', 'miễn phí', 'dự án'),
  mail(1, 'Báo cáo cuộc họp và kế hoạch dự án', 'báo cáo', 'họp', 'dự án'),
  mail(1, 'Cảm ơn, mình đã nhận được báo cáo', 'cảm ơn', 'báo cáo'),
];

/** A message to classify: how many times each vocabulary word occurs. */
export type Counts = Partial<Record<Word, number>>;

export function classCounts() {
  return [0, 1].map(label => TRAIN.filter(email => email.label === label).length) as [number, number];
}
export const prior = (label: Label) => classCounts()[label] / TRAIN.length;

/** count(w, c): occurrences of each word over all emails of class c. */
export function wordCounts(label: Label): Record<Word, number> {
  const counts = Object.fromEntries(VOCAB.map(word => [word, 0])) as Record<Word, number>;
  for (const email of TRAIN) if (email.label === label) for (const word of email.words) counts[word]++;
  return counts;
}
export const totalWords = (label: Label) => Object.values(wordCounts(label)).reduce((sum, value) => sum + value, 0);

/** P(w | c) = (count(w, c) + α) / (Σ_w count(w, c) + α·|V|): Laplace (add-α) smoothing. */
export function likelihood(word: Word, label: Label, alpha: number) {
  return (wordCounts(label)[word] + alpha) / (totalWords(label) + alpha * VOCAB.length);
}

export interface Term { word: Word; count: number; logs: [number, number] }
export interface Score { logPrior: [number, number]; terms: Term[]; logScore: [number, number]; posterior: [number, number] }

/** log P(c) + Σ_w n_w log P(w | c), then normalise. With α = 0 an unseen word gives −∞ for that class. */
export function classify(message: Counts, alpha: number): Score {
  const logPrior = [Math.log(prior(0)), Math.log(prior(1))] as [number, number];
  const terms: Term[] = VOCAB.filter(word => (message[word] ?? 0) > 0).map(word => ({ word, count: message[word]!, logs: [0, 1].map(label => message[word]! * Math.log(likelihood(word, label as Label, alpha))) as [number, number] }));
  const logScore = [0, 1].map(label => logPrior[label] + terms.reduce((sum, term) => sum + term.logs[label], 0)) as [number, number];
  return { logPrior, terms, logScore, posterior: normalise(logScore) };
}

/** Softmax of two log-scores. Both −∞ (each class has an impossible word) is undefined: returned as NaN. */
export function normalise([a, b]: [number, number]): [number, number] {
  if (a === -Infinity && b === -Infinity) return [NaN, NaN];
  const top = Math.max(a, b);
  const [ea, eb] = [Math.exp(a - top), Math.exp(b - top)];
  return [ea / (ea + eb), eb / (ea + eb)];
}

/** P(spam) after the prior and after each word occurrence, in vocabulary order: the evidence piling up. */
export function evidence(message: Counts, alpha: number) {
  const score = classify(message, alpha);
  const steps: { label: string; pSpam: number }[] = [{ label: 'prior', pSpam: normalise(score.logPrior)[0] }];
  let running = [...score.logPrior] as [number, number];
  for (const term of score.terms) for (let k = 0; k < term.count; k++) {
    running = [running[0] + term.logs[0] / term.count, running[1] + term.logs[1] / term.count];
    steps.push({ label: term.word, pSpam: normalise(running)[0] });
  }
  return steps;
}

/**
 * "trúng" and "thưởng" always appear together in the training spam, so they carry one piece of evidence. Treating them
 * as one feature "trúng thưởng" shows how much the independence assumption double-counts.
 */
export function classifyMerged(message: Counts, alpha: number) {
  const pair = Math.max(message['trúng'] ?? 0, message['thưởng'] ?? 0);
  const rest = { ...message, 'trúng': 0, 'thưởng': 0 };
  const base = classify(rest, alpha);
  const vocab = VOCAB.length - 1;
  const pairLog = [0, 1].map(label => {
    const counts = wordCounts(label as Label);
    const total = totalWords(label as Label) - counts['thưởng'];
    return pair ? pair * Math.log((counts['trúng'] + alpha) / (total + alpha * vocab)) : 0;
  }) as [number, number];
  // The other words are rescored with the merged vocabulary so both views use a proper distribution.
  const restLog = [0, 1].map(label => {
    const counts = wordCounts(label as Label);
    const total = totalWords(label as Label) - counts['thưởng'];
    return base.terms.reduce((sum, term) => sum + term.count * Math.log((counts[term.word] + alpha) / (total + alpha * vocab)), 0);
  });
  const logScore = [base.logPrior[0] + restLog[0] + pairLog[0], base.logPrior[1] + restLog[1] + pairLog[1]] as [number, number];
  return { pairLog, logScore, posterior: normalise(logScore) };
}

export const EXAMPLES: { name: string; counts: Counts }[] = [
  { name: 'Trúng thưởng, click', counts: { 'trúng': 1, 'thưởng': 1, 'click': 1 } },
  { name: 'Báo cáo dự án', counts: { 'báo cáo': 1, 'dự án': 1 } },
  { name: 'Khuyến mãi miễn phí cho dự án', counts: { 'khuyến mãi': 1, 'miễn phí': 1, 'dự án': 1 } },
  { name: 'Trúng thưởng… mời họp', counts: { 'trúng': 1, 'thưởng': 1, 'họp': 1 } },
];
