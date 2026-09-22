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
