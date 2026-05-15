from abc import ABC, abstractmethod

from joblib import dump, load


class BaseClassifier(ABC):
    VECTORIZER_PARAMS = {  # noqa: RUF012
        "max_features": 1500,
        "min_df": 5,
        "max_df": 0.7,
    }

    def __init__(self):
        self.classifier = None
        self.vectoriser = None

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    def save_model(self, model_path, vectoriser_path):
        dump(self.classifier, model_path)
        dump(self.vectoriser, vectoriser_path)

    def load_model(self, model_path, vectoriser_path):
        self.classifier = load(model_path)
        self.vectoriser = load(vectoriser_path)
