import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV

from scripts.train.utils import (
    get_config_values,
    get_models,
    get_param_grids,
    load_and_prepare_data,
    build_pipeline_from_classifier,
    get_model_paths,
)


def tune_models(cv=5, scoring="accuracy"):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    models = get_models()
    param_grids = get_param_grids()

    results = []

    for model_name, classifier_class in models.items():
        print(f"\nTuning model: {model_name}")

        pipeline = build_pipeline_from_classifier(classifier_class)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_

        model_path, vectoriser_path, pipeline_path = get_model_paths(
            model_name=model_name,
            model_type="tuned",
        )

        joblib.dump(best_pipeline.named_steps["classifier"], model_path)
        joblib.dump(best_pipeline.named_steps["vectoriser"], vectoriser_path)
        joblib.dump(best_pipeline, pipeline_path)

        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        print(f"Saved tuned model to: {model_path}")
        print(f"Saved tuned vectoriser to: {vectoriser_path}")
        print(f"Saved tuned pipeline to: {pipeline_path}")

        results.append({
            "model_name": model_name,
            "stage": "grid_search_cv",
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "model_path": str(model_path),
            "vectoriser_path": str(vectoriser_path),
            "pipeline_path": str(pipeline_path),
        })

    results_df = pd.DataFrame(results)

    results_path = model_dir / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nTuning complete.")
    print(f"Grid search results saved to: {results_path}")

    return results_df


if __name__ == "__main__":
    tune_models()