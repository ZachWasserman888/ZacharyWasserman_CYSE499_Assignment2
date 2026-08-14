# Stage 2 Hidden Test Evaluation

## Hidden Test Result

The exact Stage 1 checkpoint was reloaded and used for inference only. No retraining, fine-tuning, or checkpoint modification was performed.

- Hidden-test examples: 600
- Hidden-test accuracy: 0.6183
- True negatives: 158
- False positives: 142
- False negatives: 87
- True positives: 213

## Public vs. Hidden Test

The public-test accuracy was 0.6000, while the hidden-test accuracy was 0.6183. The hidden result differed from the public result by +0.0183. Both evaluation sets are balanced, so their total accuracies can be compared directly.

## What I Would Try Next

If I had more time or compute, I would compare several pretrained sentence encoders and use repeated stratified validation on the Stage 1 training set. I would also compare the frozen-embedding approach with a carefully regularized lightweight fine-tuning approach. These would be future experiments only; the Stage 1 checkpoint used for this evaluation was not changed.
