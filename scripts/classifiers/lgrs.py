from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        self.vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectoriser.fit_transform(X_train)
        self.classifer = LogisticRegression()
        self.classifer.fit(X_train_vectorized, y_train)
       

