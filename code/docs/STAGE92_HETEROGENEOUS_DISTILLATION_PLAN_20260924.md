# Stage92: train-only heterogeneous distillation

Stage91 establishes a 1.38320-BPB neural-only ceiling from two independently
trained architectures, but deploying both violates the runtime and asset
budgets. Stage92 distills the fixed 0.55 calibrated-Stage71 plus 0.45
Stage76 probability teacher into a single Stage71-capacity student initialized
from the exact Stage71 checkpoint.

Only supplied training prefixes produce gradients. The frozen teacher uses no
next-token labels; the student objective is 0.75 full-distribution teacher
cross-entropy plus 0.25 ordinary training next-token NLL. One fixed trajectory
uses seed 92017, batch 24, 1,800 updates, peak learning rate 1e-5, and a
prespecified average of updates 900/1200/1500/1800. Validation is monitoring and
does not alter the run. The averaged single model is scored through the frozen
Stage79 calibration and count weight 0.075. Only a material gain advances to a
fresh MKN/calibration scan and exact fused resource qualification. Test remains
untouched.

## Result

The initial point reproduced 1.4030241601 BPB. Validation improved monotonically
through 300/600/900/1200/1500/1800 updates to 1.4023756121, 1.4021399883,
1.4019946837, 1.4017754084, 1.4017648005 and **1.4017357355**. The fixed
four-checkpoint average scored **1.4017970557 BPB** (SHA-256
`5e4ca423aeca652f5193f6b6b93eafe2a940ba6bab08689cb56b09c5c0b4c417`).
This is a real 0.00122708 gain but remains above 1.4. Test was not scored.
