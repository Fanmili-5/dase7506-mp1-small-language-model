# Stage51: reallocate prefix-copy compute to residual width

Stage45 kept per-block projection parameters nearly fixed while widening the
residual stream from 256 to 288 and narrowing SwiGLU from 683 to 528. Its full
hybrid inference missed the CPU gate narrowly at 5.08012x. That experiment also
retained a 64-dimensional prefix-copy head, whose quadratic position-to-position
score is separate from the Transformer blocks.

Stage51 tests a mechanism-level reallocation: retain Stage45's wider residual
and attention stream but reduce the prefix-copy projection from 64 to 32. This
halves the copy query/key dimensions and their causal attention arithmetic. The
copy mechanism itself, eight Transformer blocks, modified-KN expert and fixed
0.125 mixture remain present. Training-only deep supervision and future-token
auxiliary heads match Stage26.

The first action is a random-weight full-hybrid resource preflight. It must pass
the unchanged three-repeat 5x CPU, 4 GiB peak RSS and 64 MiB asset gates before
any gradient update. A pass is only feasibility evidence; it makes no quality
claim and does not permit test scoring.
