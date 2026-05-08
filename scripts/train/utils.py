from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scripts.classifiers.logistic_regression import LogisticRegressionClassifier
from scripts.classifiers.naive_bayes import NaiveBayesClassifier
from scripts.classifiers.svm import SVMClassifier
from scripts.config.config import load_config
from scripts.data.ingestion import load_data
from scripts.data.preprocess import Preprocessor


def get_config_values():
    config = load_config()

    return {
        "raw_data_path": Path(config["paths"]["raw_data_path"]),
        "model_dir": Path(config["model_path"]["path"]),
        "processed_data_path": Path(config["paths"]["processed_data_path"]),
        "test_size": config["test_size"]["size"],
        "random_state": config["random_state"]["state"],
    }


def get_models():
    return {
        "naive_bayes": NaiveBayesClassifier,
        "logistic_regression": LogisticRegressionClassifier,
        "svm": SVMClassifier,
    }


def get_param_grids():
    return {
        "naive_bayes": {
            "vectoriser__max_features": [1000, 1500, 2000],
            "vectoriser__min_df": [2, 5],
            "vectoriser__max_df": [0.7, 0.9],
            "classifier__alpha": [0.1, 0.5, 1.0],
        },
        "logistic_regression": {
            "vectoriser__max_features": [1000, 1500, 2000],
            "vectoriser__min_df": [2, 5],
            "vectoriser__max_df": [0.7, 0.9],
            "vectoriser__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["liblinear", "lbfgs"],
        },
        "svm": {
            "vectoriser__max_features": [1000, 1500, 2000],
            "vectoriser__min_df": [2, 5],
            "vectoriser__max_df": [0.7, 0.9],
            "vectoriser__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
        },
    }


def load_and_prepare_data():
    config_values = get_config_values()

    data = load_data(config_values["raw_data_path"])

    if "text" not in data.columns:
        raise ValueError("Dataset must contain a 'text' column")

    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column")

    preprocessor = Preprocessor()
    data = preprocessor.preprocess(data)

    x = data["processed_text"]
    y = data["label"]

    return train_test_split(
        x,
        y,
        test_size=config_values["test_size"],
        random_state=config_values["random_state"],
        stratify=y,
    )


def build_pipeline_from_classifier(classifier_class):
    model = classifier_class()

    return Pipeline(
        [
            ("vectoriser", model.vectoriser),
            ("classifier", model.classifier),
        ]
    )


def get_model_paths(model_name, model_type):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]

    model_path = model_dir / f"{model_name}_{model_type}_model.joblib"
    vectoriser_path = model_dir / f"{model_name}_{model_type}_vectoriser.joblib"
    pipeline_path = model_dir / f"{model_name}_{model_type}_pipeline.joblib"

    return model_path, vectoriser_path, pipeline_path
