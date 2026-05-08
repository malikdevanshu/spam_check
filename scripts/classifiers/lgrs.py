from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.classifier = LogisticRegression(max_iter=1000)

    def train(self, x_train, y_train):
        x_train_vectorized = self.vectoriser.fit_transform(x_train)
        self.classifier.fit(x_train_vectorized, y_train)

    def predict(self, x_test):
        x_test_vectorized = self.vectoriser.transform(x_test)
        return self.classifier.predict(x_test_vectorized)
