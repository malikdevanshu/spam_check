from sklearn.model_selection import train_test_split

from scripts.data.Ingestion import load_data
from scripts.data.validation import feature_target_split
from scripts.classifiers.lgrs import LogisticRegressionClassifier
from scripts.data.preprocess import Preprocessor
import pandas as pd
def main():
    print("Hello from spam-check!")

    """res = class_distribution()
    print(res)
    length = avg_text_length()
    print(length)
    split = feature_target_split()
    print(split)"""
    data = load_data()
    X, y = feature_target_split(data)
    pre = Preprocessor()
    X_processed = pre.preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.33, random_state=42)
    model = LogisticRegressionClassifier()
    res = model.train(X_train, y_train, X_test, y_test)
    print(res)
    
    


if __name__ == "__main__":
    main()
