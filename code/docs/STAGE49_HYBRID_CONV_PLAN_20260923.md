# Stage49: alternating attention and gated causal convolution

Stage45 showed that simply widening every attention block to 288 exceeds the
CPU limit, while Stage48 showed that narrowing every active FFN for sparse MoE
hurts quality. Stage49 tests a different compute allocation: keep four global
self-attention blocks and replace layers 2/4/6/8 with gated causal depthwise
convolution. The saved token-mixing cost supports width 288 and SwiGLU720 in all
eight blocks.

Each convolution block applies pre-norm, a width-to-2width gate/value projection,
left-padded depthwise kernel-7 convolution, gated activation and output
projection, followed by the standard SwiGLU residual. It has no future access,
cross-window state or external data. Alternating global and local mixing aims to
retain long-range access while giving local syntax a cheaper inductive bias.

Before training, a random-weight full graph including prefix copy and the fixed
collapsed MKN expert at weight .125 must pass three fresh-process measurements:
5x baseline CPU, 4 GiB peak RSS and 64 MiB inference assets. Tests require exact
causality, normalized probabilities, finite gradients and training/export graph
equivalence. Passing permits one matched Stage26-budget run; it is not a quality
claim. No test split is scored.
