from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.VECTORIZER_PARAMS = BaseClassifier.VECTORIZER_PARAMS
        self.vectoriser = CountVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.classifier = MultinomialNB()

    def train(self, x_train, y_train):
        x_train_vectorized = self.vectoriser.fit_transform(x_train)
        self.classifier.fit(x_train_vectorized, y_train)

    def predict(self, x_test):
        x_test_vectorized = self.vectoriser.transform(x_test)
        return self.classifier.predict(x_test_vectorized)
