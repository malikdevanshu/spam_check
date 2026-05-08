import pandas as pd

from scripts.train.utils import (
    get_config_values,
    get_models,
    load_and_prepare_data,
    get_model_paths,
)


def train_models():
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    models = get_models()

    results = []

    for model_name, classifier_class in models.items():
        print(f"\nTraining baseline model: {model_name}")

        model = classifier_class()
        model.train(X_train, y_train)

        model_path, vectoriser_path, _ = get_model_paths(
            model_name=model_name,
            model_type="baseline",
        )

        model.save_model(model_path, vectoriser_path)

        print(f"Saved model to: {model_path}")
        print(f"Saved vectoriser to: {vectoriser_path}")

        results.append({
            "model_name": model_name,
            "stage": "baseline_training",
            "model_path": str(model_path),
            "vectoriser_path": str(vectoriser_path),
        })

    results_df = pd.DataFrame(results)

    results_path = model_dir / "baseline_training_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nBaseline training complete.")
    print(f"Results saved to: {results_path}")

    return results_df


if __name__ == "__main__":
    train_models()