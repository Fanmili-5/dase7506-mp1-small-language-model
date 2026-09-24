# Stage110: bounded low-cost gate ablation

Stages103–105 reproduce 1.399686 BPB but fail the 5× CPU limit. A one-batch
CPU profile attributes measurable cost to the gate's top-2 operation and
sparse maximum-mass reduction, while neural matrix multiplication dominates
overall. Stage110 keeps the Stage100 coefficients, means, scales, Stage94
calibration, Stage92 neural and Stage73 order-6 count statistics fixed. It
screens eight prespecified feature subsets at slope scales
0.2/0.3/0.4/0.5/0.6, including the exact Stage102 reference. Validation
selects only a feature subset and scale; no coefficient receives validation
gradients. The purpose is to identify a sub-1.4 gate with fewer expensive
features for subsequent exact export and independent resource qualification.
Test remains untouched.

## Result

The exact four-feature scale-0.5 reference reproduced at 1.3996861632 BPB.
Among the predeclared rows, the full gate at scale 0.4 reached 1.399612395,
but does not reduce inference work. Removing neural margin yielded
1.399966472 at scale 0.4, only 0.000033528 below the target; removing the
neural maximum or sparse maximum mass gave best scores 1.400203248 and
1.400928806 respectively. Count-only and neural-only subsets scored at best
1.400825714 and 1.400877463. Thus no low-cost variant has enough measured
quality margin or a credible estimated time saving to be promoted without a
new exact export and resource test. Raw all-row results are in
`code/results/stage110-evidence/result.json`. Test was not scored.
