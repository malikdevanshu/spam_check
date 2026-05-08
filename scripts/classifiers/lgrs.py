from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
        self.classifier = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)

    def predict(self, X_test):
        X_test_vectorized = self.vectoriser.transform(X_test)
        return self.classifier.predict(X_test_vectorized)    
       

