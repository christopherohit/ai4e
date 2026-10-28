// Gradient descent lab model on Auto MPG: predict fuel economy (mpg) from car weight. The same data in two units: raw
// weight in thousands of pounds (a long, narrow valley) and standardized weight (a round bowl).
import data from '../../_shared/glass-lab/data/auto-mpg.json';
import { LinearProblem, standardize, type Theta } from '../../_shared/glass-lab/regression';

export const CARS = data.name.map((name, i) => ({ name, mpg: data.mpg[i], weight: data.weight[i], year: data.year[i], origin: data.origin[i] }));
export const SOURCE = data.source;
const kilo = data.weight.map(w => w / 1000);
const scaled = standardize(kilo);
export const WEIGHT_MEAN = scaled.mean, WEIGHT_SD = scaled.sd;

/** ŷ = w · (weight / 1000 lb) + b. */
export const raw = new LinearProblem(kilo, data.mpg);
/** ŷ = w · z + b with z = (weight − mean) / sd. */
export const std = new LinearProblem(scaled.z, data.mpg);

/** The same straight line expressed in the other parametrisation. */
export const rawToStd = ([w, b]: Theta): Theta => [w * WEIGHT_SD, b + w * WEIGHT_MEAN];
export const stdToRaw = ([w, b]: Theta): Theta => [w / WEIGHT_SD, b - (w / WEIGHT_SD) * WEIGHT_MEAN];

/** A deliberately bad first guess: a line that rises with weight. */
export const START_RAW: Theta = [2, 20];
export const START_STD = rawToStd(START_RAW);

export const VIEW_RAW = { w: [-16, 4] as const, b: [10, 76] as const };
export const VIEW_STD = { w: [-17, 4] as const, b: [13, 34] as const };

export const LR_PRESETS = [0.01, 0.05, 0.09, 0.1] as const;
