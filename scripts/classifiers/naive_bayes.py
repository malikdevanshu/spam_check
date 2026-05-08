from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.vectoriser = CountVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.classifier = MultinomialNB()

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train).toarray()
        self.classifier.fit(X_train_vectorized, y_train)

    def predict(self, X_test):
        X_test_vectorized = self.vectoriser.transform(X_test)
        return self.classifier.predict(X_test_vectorized)    