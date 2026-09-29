# Stage47: training-only R-Drop consistency

The Stage26 curve still has a train/validation generalization gap, while adding
untied output capacity did not help. Stage47 therefore changes optimization,
not inference capacity. For every sampled causal window it performs two
independent dropout forwards, averages the unchanged primary/deep/future losses,
and adds symmetric KL between the two final prefix-copy distributions at fixed
weight 0.5.

The two passes expose the same 58,982,400 unique primary target positions; all
117,964,800 stochastic primary presentations and doubled auxiliary
presentations will be disclosed. Training compute is approximately doubled.
The seed, sampled windows, 7,200 updates, optimizer, learning-rate schedule,
architecture, checkpoint averaging and supplied training text remain fixed.

R-Drop and auxiliary modules are removed at export, which must reproduce the
ordinary `student_structured` graph exactly. Consequently inference CPU, RAM
and assets are unchanged for a given exported state. Advancement requires at
least 0.003 BPB improvement over Stage26 before count mixing and a fresh final
resource gate. This is one fixed coefficient, not a validation coefficient
sweep. No test split is scored.

## Result

The complete run finished all 7,200 updates. Its endpoint validation BPB was
1.4522765848, versus 1.4687773150 for the matched Stage26 control. The fixed
average of checkpoints 6000, 6300, 6600, 6900 and 7200 was exported to the
unchanged `student_structured` inference graph and independently scored on CPU
FP32 at **1.4506130032 BPB**. This improves the matched Stage26 average,
1.4649939083, by **0.0143809051 BPB**, exceeding the preregistered 0.003 gate.

The exported checkpoint SHA-256 is
`4266b9c559da7ce62703c8762ea3b7deb5d0feaa4fc0eff234a3fa6c123f01c4`.
Training took about 1,754 seconds excluding validation and reported peak CUDA
allocation/reservation of 4.165/4.333 GB. Stage47 is accepted as a neural
ablation; hybrid calibration and the complete CPU/RAM/asset resource audit are
handled separately by Stage52. Test remains untouched.
