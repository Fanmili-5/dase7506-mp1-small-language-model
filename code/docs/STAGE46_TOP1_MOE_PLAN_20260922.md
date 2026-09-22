# Stage46: sparse conditional-capacity preflight

The remaining gap to the requested 1.4 BPB is too large for another scalar or
seed search. Stage46 tests a structural hypothesis: the dense model may need
more conditional capacity, not more active computation at every position.

Every dense SwiGLU683 FFN is replaced by two top-1 routed SwiGLU512 experts. A
token executes exactly one expert. Relative to the dense FFN, active projection
weights fall from about 525k to 393k per layer, while stored FFN capacity rises
to about 786k. The router is causal and stateless. Training will use a standard
load-balancing term and router z-loss; both disappear from the inference loss.
The gate multiplier starts near one (`2 * p_selected`) rather than halving every
new residual branch at initialization.

Sparse dispatch can be slower than its FLOP count suggests on CPU. Therefore
this commit permits only a random-weight full-hybrid preflight with the fixed
collapsed modified-Kneser-Ney expert at weight .125. It must pass the unchanged
5x CPU, 4 GiB RSS and 64 MiB asset limits before training is implemented or
launched. Passing is only feasibility evidence, not a quality claim. Test is
not scored.

## Preflight result

The complete random-weight MoE plus fixed collapsed MKN expert passed all three
gates. Across three fresh CPU processes its median candidate-to-baseline ratio
was **4.872452227x**; maximum peak RSS was 2,052,231,168 bytes and conservative
inference assets were 52,622,669 bytes. The neural graph contains 8,954,369
parameters. This admits one matched-target training run but does not establish
quality, and the final trained/exported checkpoint must repeat the exact gate.
No test split was scored.
