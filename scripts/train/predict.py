import argparse

import pandas as pd

from scripts.data.preprocess import Preprocessor
from scripts.train.utils import get_model_paths, get_models


def predict_texts(model_name, texts, model_type="tuned"):
    models = get_models()

    if model_name not in models:
        sub = model_name
        msg = f"Unknown model name: {sub}"
        raise ValueError(msg)

    model_path, vectoriser_path, _ = get_model_paths(
        model_name=model_name,
        model_type=model_type,
    )

    if not model_path.exists() or not vectoriser_path.exists():
        sub = model_name
        msg = f"Model/vectoriser files not found in {sub}"
        raise FileNotFoundError(msg)

    classifier_class = models[model_name]

    model = classifier_class()
    model.load_model(model_path, vectoriser_path)

    data = pd.DataFrame({"text": texts})

    preprocessor = Preprocessor()
    data = preprocessor.preprocess(data)

    predictions = model.predict(data["processed_text"])

    return pd.DataFrame(
        {
            "text": texts,
            "prediction": predictions,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Predict spam/ham text.")

    parser.add_argument(
        "--model",
        choices=["naive_bayes", "logistic_regression", "svm"],
        default="logistic_regression",
        help="Model to use for prediction.",
    )

    parser.add_argument(
        "--model-type",
        choices=["baseline", "tuned"],
        default="tuned",
        help="Use baseline or tuned model.",
    )

    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Text values to classify.",
    )

    args = parser.parse_args()

    results = predict_texts(
        model_name=args.model,
        texts=args.text,
        model_type=args.model_type,
    )

    print(results)


if __name__ == "__main__":
    main()
