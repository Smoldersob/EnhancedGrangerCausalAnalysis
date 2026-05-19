# Evaluating Granger Analysis Results

Granger analysis results are usually evaluated with a mix of statistical and practical checks.

The main indicators are:

- **p-value**: the primary decision rule for declaring causality. Smaller p-values indicate stronger evidence against the null hypothesis.
- **F-statistic**: used together with the p-value to measure how much the reference model improves over the base model.
- **Prediction error**: compare base and reference errors; a meaningful drop in error supports the causality claim.
- **Signed result**: the sign matrix can show whether the strongest contribution is positive or negative.
- **Ground truth comparison**: if a labeled causality matrix exists, evaluate precision, recall, F1-score, and accuracy against it.

In practice, a result is most reliable when low p-values are consistent with improved error metrics and, when available, with known ground truth.