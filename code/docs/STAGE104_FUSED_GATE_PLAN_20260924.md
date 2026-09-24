# Stage104: fused gated inference

Stage103 exports the exact Stage102 predictor, but the direct implementation
creates full dense copy and count distributions. Stage104 preserves all
weights, count tables and gate scalars while scattering prefix-copy mass into
the vocabulary tensor and adding collapsed MKN mass with a per-prefix gate
weight. It compares against the exact Stage103 checkpoint on deterministic
windows, then runs full CPU FP32 validation and three alternating CPU timing
runs. Peak RSS and inference asset bytes are independently checked. Test
remains untouched.

## Result

The fused checkpoint reproduced Stage103 at **1.399686163170 validation BPB**.
Its three-repeat Windows median was 5.484419× the fixed baseline, improved
from Stage103's 6.064540× but still above the 5× limit. Peak RSS was
2,063,765,504 bytes and conservative inference assets 53,264,132 bytes,
within their respective limits. The checkpoint SHA-256 is
`56a210fc59bf8faff3bcf5971d63a003a0ca987a4b09a492c686dd3e29cbdf1b`.
Raw evidence is in `code/results/stage104-evidence/`. It is not a qualified
submission candidate, and test was not scored.
