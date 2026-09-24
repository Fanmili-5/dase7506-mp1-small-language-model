# Stage136: train-only repair of a single successor-copy head

Stage135 changed Stage115's content-copy values to successor-copy values
without adapting weights and validation jumped from 1.400225 to 1.559768
BPB. This establishes a severe parameter/semantics mismatch, not a bound on
trained successor copying. Stage40's independently trained successor route
was slightly better than content-copy on an older model, and unlike Stage133's
added cache this replacement preserves one attention route.

Start from the exact Stage92 averaged neural checkpoint. Freeze all eight
backbone blocks, token/head weights, vocabulary bias and count tables. Change
only the copy value alignment to earlier-context successors, and train only
the existing 64-dimensional copy query/key and scalar gate (no new weights).
At position zero the gate is forced to zero. The objective is true next-token
NLL of the fixed Stage94-calibrated neural probability mixed with the frozen
train-only order-five MKN target probability at weight 0.0625. Random 257-
token crops come solely from supplied training text; MKN statistics and the
unigram prior are frozen train-derived assets. The static-mixture objective
is a deliberate first gate and does not assume Stage100's content-copy
confidence-gate coefficients transfer to successor semantics.

One seed **136017**, batch **24**, **2,400** optimizer updates, AdamW peak LR
`5e-4`, 100-step warmup, minimum LR 10%, weight decay 0.01, and no backbone
dropout (the frozen feature extractor stays in eval mode) are fixed. This
presents 14,745,600 training targets. Score complete validation at
0/300/600/900/1200/1500/1800/2100/2400 using the unchanged static mixture;
save those checkpoints. Use the minimum of the prespecified checkpoints as
the exploratory selection, while reporting the entire trajectory and a fixed
average of 1500/1800/2100/2400. No source, seed, learning rate, or checkpoint
is altered after monitoring. If neither selected endpoint nor fixed average
beats the original Stage92 static 1.401708 BPB, stop. A gain that crosses
1.4 still needs an exact deployable fused export and three-repeat official
CPU/RAM/asset qualification. Stage85 remains the qualified fallback.

The strongest objection is that the frozen representation may lack features
useful for matching prior contexts, or training may merely drive copy-gate
weight toward zero. Report validation BPB and mean gate activation, so an
apparent recovery cannot be mistaken for successful successor retrieval.
Only train text generates gradients; test stays untouched.

## Completed result

The Windows RTX 3070 Ti run completed all 2,400 updates and 14,745,600
training targets. The fixed static-mixture validation trajectory at steps
0/300/600/900/1200/1500/1800/2100/2400 was
**1.560908/1.437538/1.411317/1.410222/1.409999/1.410026/1.409914/
1.409918/1.409885 BPB** over every one of 376,599 targets. The selected
last endpoint was 1.409885016 BPB. The prespecified four-checkpoint average
(SHA-256 `5cc7e6957ec03aae9d9acaa28bb5681b3ecc2b50aa284bf7f497ca99b3f5d4fc`)
scored **1.409892011 BPB**. Mean validation copy-gate activation first fell
from 0.11590 to 0.03965 and then recovered to 0.08270 at the endpoint;
the route did not simply shut off.

This is a large recovery from the untrained swap but still 0.00818 BPB worse
than the unchanged Stage92 static control. The predeclared quality gate fails,
so this frozen-backbone head-only recipe is not exported or resource-audited.
It does not rule out jointly adapting the backbone and head, which is a
different, more expensive experiment. Raw run, complete trajectory and
average evidence are in `../results/stage136-evidence/`. No test scoring.
