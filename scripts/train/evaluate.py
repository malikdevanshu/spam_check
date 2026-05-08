import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from scripts.train.utils import (
    get_config_values,
    get_models,
    load_and_prepare_data,
    get_model_paths,
)


def evaluate_models(model_type="tuned"):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    models = get_models()
    results = []

    for model_name, classifier_class in models.items():
        print(f"\nEvaluating {model_type} model: {model_name}")

        model_path, vectoriser_path, _ = get_model_paths(
            model_name=model_name,
            model_type=model_type,
        )

        if not model_path.exists() or not vectoriser_path.exists():
            print(f"Skipping {model_name}. Saved files not found.")
            continue

        model = classifier_class()
        model.load_model(model_path, vectoriser_path)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"{model_name} accuracy: {accuracy:.4f}")
        print(report)

        results.append({
            "model_name": model_name,
            "stage": f"{model_type}_evaluation",
            "accuracy": accuracy,
            "model_path": str(model_path),
            "vectoriser_path": str(vectoriser_path),
        })

    results_df = pd.DataFrame(results)

    results_path = model_dir / f"{model_type}_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nEvaluation complete.")
    print(f"Results saved to: {results_path}")

    return results_df


if __name__ == "__main__":
    evaluate_models(model_type="tuned")