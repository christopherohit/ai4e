const terms = new Set([
  'Attention', 'Self-Attention', 'Multi-head Self Attention', 'Embedding', 'Text embedding',
  'Text Encoder', 'Image Encoder', 'Image Decoder', 'Latent representation', 'Noise',
  'Random noise', 'Seed', 'Random seed', 'Guidance scale', 'Prompt', 'Timestep', 'Temperature', 'Sampling',
  'Generator', 'Discriminator', 'Learning rate', 'Batch size', 'Loss', 'Noise dimension',
  'Features', 'Accuracy', 'Precision', 'Recall', 'Confusion Matrix', 'Neuron', 'Neurons',
  'Activation threshold', 'Layer', 'Head', 'Heads', 'Attention heads', 'Epoch',
  'Data Distribution', 'Layered Distributions', 'Gradients', 'Activation', 'Denoising',
  'Text Representation', 'Image Representation', 'Language Model', 'Tokenize',
  'Attention Head', 'Attention Heads', 'Probabilities', 'Residual', 'Key', 'Query', 'Value',
  'Activation Score Threshold', 'True Positive Rate', 'False Positive Rate',
  'Feature Distribution', 'Attention Flow', 'Saliency', 'Dependency', 'Instance',
  'Text Representation Generator', 'Image Representation Refiner', 'Representation',
  'Saliency View', 'Dependency List', 'Dependency Tree', 'Syntactic Relations',
  'syntactic correlations', 'semantic correlations', 'importance scores',
  'Attention Wights', 'Attention Weights', 'Semantic Attention Graph', 'Force Layout',
  'Radial Layout', 'Grid Layout', 'Edge Force', 'ManyBody', 'Node Distance', 'Edge Threshold',
  'Semantic Saliency Score', 'Attention Head Overview', 'Log loss', 'LeastSq loss',
  'Optimizer', 'hidden layer(s)', 'neuron(s)', '1D Uniform', '1D Gaussian',
  '2D Uniform', '2D Gaussian', 'Tokenizer', 'Hybrid Text-Image Space',
  'Dropout', 'Layer Normalization', 'Dot product', 'Scaling · Mask',
  'Discriminator Loss', 'Generator Loss', 'Convolution', 'Pooling', 'Kernel', 'Bias',
  'Padding', 'Kernel Size', 'Stride', 'conv', 'max_pool', 'flatten', 'weight',
  'Normalize', 'normalization', 'class probabilities',
].map(term => term.toLowerCase()));

export function isTechnicalTerm(value: string): boolean {
  return terms.has(value.trim().replace(/\s+/g, ' ').replace(/:$/, '').toLowerCase());
}
