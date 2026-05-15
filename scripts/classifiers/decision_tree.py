from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_classifier import BaseClassifier


class Decision_classifier(BaseClassifier):  # noqa: N801
    def __init__(self):
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.classifier = DecisionTreeClassifier()

    def train(self, x_train, y_train):
        x_train_vectorized = self.vectoriser.fit_transform(x_train)
        self.classifier.fit(x_train_vectorized, y_train)

    def predict(self, x_test):
        x_test_vectorized = self.vectoriser.transform(x_test)
        return self.classifier.predict(x_test_vectorized)
