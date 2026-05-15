import pytest

from scripts.classifiers.logistic_regression import (
    LogisticRegressionClassifier,
)
from scripts.classifiers.naive_bayes import NaiveBayesClassifier
from scripts.classifiers.svm import SVMClassifier
from scripts.classifiers.decision_tree import Decision_classifier

CLASSIFIERS = [
    NaiveBayesClassifier,
    LogisticRegressionClassifier,
    SVMClassifier,
    Decision_classifier,
]


@pytest.mark.parametrize("classifier_class", CLASSIFIERS)
def test_classifier_can_train_and_predict(classifier_class):
    x_train = [
        "free money win prize",
        "limited offer claim cash",
        "team meeting tomorrow",
        "project update schedule",
        "win cash bonus now",
        "normal work email",
        "team meeting project schedule update tomorrow",
        "project meeting agenda schedule update attached",
        "team project discussion schedule update today",
        "meeting agenda project update notes attached",
        "team schedule project meeting tomorrow update",
        "project update meeting schedule agenda tomorrow",
        "free money prize winner claim offer now",
        "free cash prize winner claim offer today",
        "free money bonus winner claim offer now",
        "free prize cash winner claim offer urgent",
        "free money prize claim offer limited",
        "free cash bonus prize winner claim now",
    ]
    y_train = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = classifier_class()
    model.train(x_train, y_train)

    predictions = model.predict(["free prize", "team meeting"])

    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize("classifier_class", CLASSIFIERS)
def test_classifier_save_and_load(tmp_path, classifier_class):
    x_train = [
        "free money win prize",
        "limited offer claim cash",
        "team meeting tomorrow",
        "project update schedule",
        "win cash bonus now",
        "normal work email",
        "team meeting project schedule update tomorrow",
        "project meeting agenda schedule update attached",
        "team project discussion schedule update today",
        "meeting agenda project update notes attached",
        "team schedule project meeting tomorrow update",
        "project update meeting schedule agenda tomorrow",
        "free money prize winner claim offer now",
        "free cash prize winner claim offer today",
        "free money bonus winner claim offer now",
        "free prize cash winner claim offer urgent",
        "free money prize claim offer limited",
        "free cash bonus prize winner claim now",
    ]
    y_train = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = classifier_class()
    model.train(x_train, y_train)

    model_path = tmp_path / "model.joblib"
    vectoriser_path = tmp_path / "vectoriser.joblib"

    model.save_model(model_path, vectoriser_path)

    original_prediction = model.predict(["free money"])

    loaded_model = classifier_class()
    loaded_model.load_model(model_path, vectoriser_path)

    loaded_prediction = loaded_model.predict(["free money"])

    assert len(loaded_prediction) == 1
    assert loaded_prediction[0] in {0, 1}
    assert loaded_prediction[0] == original_prediction[0]
