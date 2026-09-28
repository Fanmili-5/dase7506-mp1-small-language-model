# Post-Stage223 method and literature triage (no new experiment)

This is a decision record, not a model claim or release authorization. The
course fixes supplied training text, BPE-2048 tokenizer, independent causal
256-token evaluation windows, and CPU-FP32 limits of 5x baseline time,
4 GiB RAM, and 64 MiB uncompressed assets. Published word-level perplexity,
WikiText-103, and translation BLEU are not this assignment's BPB.

The protected Stage143 scorer is 1.399686162 validation / 1.415657617 test
BPB. Its within-window prefix-copy, train-only MKN count expert, and neural
backbone already cover several generic literature ideas. The Stage218 fixed
three-family mixture reached 1.371031 validation BPB but exceeded resources
and still missed 1.35; it is a diagnostic, not a submission predictor. The
Stage204/205/223 adjacent-token-input pilots all missed their continuation
gates. These local outcomes motivate stopping near-duplicate trials, not a
claim that the design space is exhausted.

| Published method | Fit to this assignment and local evidence | Decision |
| --- | --- | --- |
| Li, Povey & Khudanpur (2020), [implicit cache pointers](https://arxiv.org/abs/2009.13774) | Adds history-word output symbols and altered supervision; paper reports WikiText-2 word-perplexity benefits, not fixed-BPE BPB. A strictly within-window version may be legal, but Stage143 already copies causal prefix tokens and Stage43/132/136 found only small or negative nearby gains. | Mechanistically relevant, but no evidence of the roughly 0.05 validation gain needed. Do not implement merely because the paper improves another metric. |
| Provilkov, Emelianenko & Voita (2020), [BPE-Dropout](https://arxiv.org/abs/1910.13267) | Stochastic training segmentation with unchanged vocabulary and deterministic inference; original evidence is machine-translation BLEU. The guide says keep the tokenizer unchanged, so permission for changing its training-time application is uncertain. | Ask instructor if considering it; no run or claim of legality/benefit now. |
| Krause et al. (2017), [dynamic evaluation](https://arxiv.org/abs/1709.07432) | Adapts model to recent evaluation history; its main mechanism conflicts with independent windows/no cross-window state and likely adds CPU overhead. | Exclude as published, stateful method. |
| Dai et al. (2019), [Transformer-XL](https://arxiv.org/abs/1901.02860) | Segment recurrence supplies longer context, unavailable to the fixed independent 256-token evaluator. The relative-position component alone is not the paper's full gain and a nearby Stage216 relative bias missed its gate. | Exclude cross-window recurrence; no nearby positional sweep. |
| Khandelwal et al. (2020), [kNN-LM](https://arxiv.org/abs/1911.00172) | Requires a train-derived datastore and retrieval; Stage163's compact permitted-version mixtures all worsened full validation, while asset/CPU budgets are tight. | Do not revisit this local route without new independent evidence. |

The remaining *research question* is not “which famous architecture has a
better published score?” but which distinct, causal, train-only mechanism can
improve this model's distributed out-of-sample errors by enough while fitting
the CPU/asset envelope. The reviewed papers do not currently identify one.
Before any further run, require (1) a mechanism not already represented by
local negative trials, (2) a clear course-legality argument, (3) a quantified
route to the missing BPB margin, and (4) a predeclared equal-target and
resource gate. Until then, preserve Stage143 and prioritize the submission
deadline decision with the student.

The hypothesis/rival and fixed-endpoint design record used procedural
guidance from Kassis et al. (2026), [*Scientific Agent Skills: A Library of
Procedural Knowledge for Research Agents*](https://doi.org/10.48550/arXiv.2609.00065).
This citation acknowledges planning assistance, not evidence of model quality.
