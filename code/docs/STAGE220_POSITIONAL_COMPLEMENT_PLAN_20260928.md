# Stage220: is the frozen-expert gain concentrated at window starts?

The protected Stage143 model scores 1.399686162 full-validation BPB. A
frozen Stage143/Stage155 50:50 probability mixture scores 1.376182664 but
cannot be deployed under the current CPU/asset envelope. Stage162 showed
that selecting only some whole windows loses most of the gain. A distinct
possibility is that the gain concentrates in the early positions of *every*
independent 256-token window, where a short-prefix helper might be cheap.

Use only the existing Stage162 validation true-target log-probability
streams, SHA-256 `75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3`
for Stage143 and `0cbe8ee4aec517ba756fc5e167a95aea2b82e088b5f870e29f77337a9a499603`
for Stage155. Require 376,599 finite targets and reproduce Stage143,
Stage155 and their uniform mixture BPB before analysis. No new model is
trained; do not open test data or refit weights.

Predeclare eight disjoint position bins `[0,32),...,[224,256)` and report
each bin's target count, share of the total mixture NLL gain, and the
mean per-target gain. Also report fixed input-position-only policies that
use the 50:50 mixture for only the first 32, 64 or 128 positions and
Stage143 elsewhere. These policies are causal but still *over budget*;
the true-target arrays are analysis inputs, never inference features.

Advance to engineering a new short-context helper only if the first 64
positions contribute at least half of the complete frozen-mixture gain
**and** the first-64 policy gains at least 0.012 BPB over Stage143. Both
conditions are required. Passing would license only a separately planned
train-only model and exact CPU/assets check, not a claim of <1.35 or a
test score. If either fails, stop this route. The objection is that the
known ensemble complement may be broadly distributed across all positions;
this fixed analysis directly tests that objection.
