# Stage75: balanced-hybrid MKN qualification

Stage74 beats the matched Stage54 architecture control by 0.0033153 BPB with a
different resource-admitted global/local layer allocation.  Stage75 keeps the
exact Stage74 average and frozen train-only Stage25 modified Kneser--Ney expert,
scans the existing fixed count-weight grid 0.0000--0.2000 in increments of
0.0125 on validation, serializes the selected collapsed predictor, and performs
independent CPU-FP32 scoring plus three-repeat CPU/RAM/asset qualification.

No gradient training, new statistics, grid expansion or test scoring occurs.
The branch advances to low-learning-rate continuation only if the collapsed
predictor improves the Stage74 neural average and passes CPU <=5x baseline,
RSS <=4 GiB and uncompressed inference assets <=64 MiB.
