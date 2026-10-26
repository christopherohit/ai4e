# optimization-lab

Shared code for the foundational machine-learning explainers: gradient-descent, mini-batch-sgd, naive-bayes, svm,
bias-variance and regularization.

- model.ts: 64 synthetic samples, MSE loss and gradient, closed-form optimum, level sets, seeded mini-batch training and gradient-noise formula (used by the two optimization labs).
- plots.ts / plots.css: SVG frame, contour map, data scatter, curves, arrows, class markers, bands and decision fields, styled with the sequence-lab colour roles.

The lab chrome (steps, step card, controls, chart) comes from tools/_shared/sequence-lab/.
