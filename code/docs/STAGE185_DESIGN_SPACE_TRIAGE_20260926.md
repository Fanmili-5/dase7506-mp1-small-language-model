# Stage185: evidence-bounded design-space and release decision

This is a decision record, **not** a new score or trained candidate. The
assignment permits new architectures and unlimited training from the supplied
training text, but fixes BPE-2048, independent causal 256-token windows, CPU
FP32 scoring, <=5x baseline time, <=4 GiB evaluation RAM, <=64 MiB inference
assets, and no test-set selection. Stage143 is 1.399686162 complete-validation
BPB, leaving 0.049686162 to the aspirational 1.35 target. Its local Windows
resource ratio is 3.617702x, but an independent Linux one-thread host measured
5.502555x. Any replacement needs both a material quality gain and a measured
resource pass; an input-only preflight alone is not enough.

## Divergent hypotheses, then evidence filter

The list was generated using failure analysis (where Stage143 loses),
composition/decomposition (neural, copy and count roles), and the quality vs
CPU/asset tension. Entries are retained even when rejected so that an old
negative result is not mistaken for an untested idea.

| Candidate mechanism | Current relevant evidence | Decision before new spend |
| --- | --- | --- |
| Wider Transformer | Stage147 first-2400 gain 0.01135 BPB; Stage155 full larger pure-neural average 1.40988 | No nearby width sweep |
| Deeper Transformer | Stage150 first-2400 gain 0.01151; Stage154 upper branch gain 0.00600 | No nearby depth sweep |
| Parallel global/local block | Stage177 input-only feature time 1.43857x current graph, above its 1.25 gate | Reject before training |
| Two independent neural experts | Stage91 over-budget mixture 1.38162 BPB, still above 1.35 | Cannot deploy directly |
| Compress the two-expert teacher | Stage158 1.40321; Stage169 only 0.000246 gain over Stage143 | No repeated distillation run |
| Stronger train-only regularization | Stage179 dropout 0.20 and Stage184 drop path 0.10 both regressed | No rate/seed search |
| Optimizer replacement | Stage168 SAM and Stage180 Muon failed matched early gates | No optimizer sweep |
| Byte/character spelling head | Stage160 residual worsened; Stage174 auxiliary loss did not improve | No near-duplicate head |
| Frequency-targeted training | Stage161 first-2400 endpoint was worse than control | Reject this fixed weighting |
| Longer exact history / cache | Stage146/148 target-only gain <=0.00304 BPB | Too small for current gap |
| Distant co-occurrence or compact kNN | Stage149 and Stage163 nonzero mixtures worsened full validation | Reject these fixed retrieval routes |
| Rich causal neural/count gate | Stage175 hindsight oracle 1.30019, but Stage176 causal-feature crossfit 1.40004 | Oracle gap is not deployable evidence |
| Geometric expert agreement | Stage183 best diagnostic gain 0.00144, below 0.015 gate | No deployment |
| Word-boundary/lexical count expert | Not directly tested; may pool word histories, but likely overlaps existing order-six MKN and must fit ~8 MiB remaining assets | Only a bounded full-distribution diagnostic, no immediate build |
| Alternative count smoothing (e.g. escape/backoff) | MKN is already train-only and useful; no same-table alternative has established a large gain | Only normalized diagnostic with exact cost estimate |

The strongest remaining *unmeasured* directions are a word-boundary expert
and a different count estimator. Neither presently has evidence for a
~0.05-BPB gain, and both add CPU/asset cost to a candidate with a known
Linux-portability risk. A target-only probability improvement would not
establish a normalized model score. Therefore do **not** start another GPU
training run or post-hoc validation grid merely to keep searching. If new
evidence or a concrete student-supplied mechanism changes this ranking,
predeclare one mechanism, a complete-validation quality gate of at least
0.015 BPB, a CPU/asset projection and a matched control before execution.

## Immediate release path and explicit dependencies

1. Protect Stage143 checkpoint/graph hashes and the clean-extract record.
   Current read-only preflight says `eligible_for_freeze_only`, not frozen.
2. Obtain the student's choice of freeze date and student ID. The guide asks
   for a score issue **before 29 September** and final public immutable code
   plus matching checkpoint links by 30 September (UTC+8).
3. Only after explicit method freeze, run the SHA-bound full CPU FP32 test
   once, produce a report matching that exact candidate, and audit its <=10
   pages, links, bundle and course-site issue. The historical `REPORT.pdf`
   must be replaced; it describes a different Stage10 checkpoint.

This prioritization is not a claim that 1.35 is mathematically impossible.
It is a deadline-aware comparison of measured routes and their residual
risks. It does not authorize test scoring or publishing the private repo.
