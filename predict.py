# predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate sentiment predictions from the saved Stage 1 checkpoint."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_checkpoint"),
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_dir: Path):
    """Load the saved encoder, classifier, and metadata."""
    encoder_path = checkpoint_dir / "embedding_model"
    classifier_path = checkpoint_dir / "classifier.joblib"
    metadata_path = checkpoint_dir / "metadata.json"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing embedding model: {encoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier: {classifier_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    encoder = SentenceTransformer(str(encoder_path))
    classifier = joblib.load(classifier_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return encoder, classifier, metadata


def main() -> None:
    """Generate predictions without retraining."""
    args = parse_args()

    data = pd.read_csv(args.input_csv)

    required_columns = {"id", "text"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing_columns)}"
        )

    encoder, classifier, metadata = load_checkpoint(args.checkpoint)

    embeddings = encoder.encode(
        data["text"].astype(str).tolist(),
        batch_size=int(metadata["encoding_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    predictions = classifier.predict(embeddings).astype(int)

    output = pd.DataFrame(
        {
            "id": data["id"].to_numpy(),
            "predicted_label": predictions,
        }
    )

    output.to_csv(args.output_csv, index=False)

    print(f"Saved {len(output)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
