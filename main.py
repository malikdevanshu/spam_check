from scripts.data.Ingestion import load_data
from scripts.data.validation import class_distribution, avg_text_length, feature_target_split
from scripts.data.preprocess import Preprocessor
import pandas as pd
def main():
    print("Hello from spam-check!")

    res = class_distribution()
    print(res)
    length = avg_text_length()
    print(length)
    split = feature_target_split()
    print(split)
    
    
    


if __name__ == "__main__":
    main()
