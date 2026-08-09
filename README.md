# Sentiment Classification with Pretrained Embeddings — Stage 1

This repository contains the Stage 1 sentiment-classification submission.

## Task

Predict movie-review sentiment:

- `0` = negative
- `1` = positive

The released training set contains 240 reviews and is intentionally imbalanced:

- 180 positive
- 60 negative

The released public test set contains 400 balanced reviews:

- 200 positive
- 200 negative

## Model

The final model uses:

1. a frozen pretrained Sentence Transformer,
2. normalized sentence embeddings,
3. class-weighted logistic regression.

The pretrained encoder is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This design was chosen because the training set is very small. Freezing the pretrained encoder reduces overfitting risk, while the pretrained representations can still handle words and phrases that were not present in the 240 released training reviews.

The logistic-regression classifier uses balanced class weights to address the 180/60 class imbalance.

## Repository Files

```text
stage1_notebook.ipynb
README.md
requirements.txt
train.csv
public_test.csv
model_checkpoint/
public_test_predictions.csv
predict.py
```

`model_checkpoint/` and `public_test_predictions.csv` are created when the notebook is run.

## Setup

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Stage 1

Open:

```text
stage1_notebook.ipynb
```

Run all cells from top to bottom.

The notebook will:

1. load and validate the released CSV files,
2. verify class distributions,
3. compute frozen pretrained sentence embeddings,
4. create a stratified validation split,
5. compare several logistic-regression regularization values,
6. select the best value using validation macro F1,
7. retrain the classifier on all 240 training reviews,
8. evaluate on `public_test.csv`,
9. print total public-test accuracy,
10. print and plot the public-test confusion matrix,
11. save a complete checkpoint,
12. create `public_test_predictions.csv`,
13. reload the checkpoint and confirm predictions match without retraining.

## Checkpoint

The notebook saves:

```text
model_checkpoint/
├── classifier.joblib
├── metadata.json
└── embedding_model/
```

The local `embedding_model/` directory contains the pretrained encoder files required for later inference.

## Prediction File

The notebook creates:

```text
public_test_predictions.csv
```

with exactly:

```text
id,predicted_label
```

where `predicted_label` is `0` or `1`.

## Reload Without Retraining

After the notebook creates the checkpoint, inference can also be run from the command line:

```bash
python predict.py public_test.csv public_test_predictions_reloaded.csv
```

This loads the saved Stage 1 checkpoint and does not retrain the model.

## Important Stage 1 Rule

Do not train on `public_test.csv`. It is used only for local evaluation and prediction generation.
