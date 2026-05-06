import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
class Preprocessor:
    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))
        self.url_pattern = re.compile(r"http\S+|www\.\S+")
        self.email_pattern = re.compile(r"\S+@\S+")
        self.money_pattern = re.compile(r"[$€£]\s*\d+|\d+\s*[$€£]")
        self.number_pattern = re.compile(r"\b\d+\b")
        self.space_pattern = re.compile(r"\s+")
        

    def preprocess_text(self, text):
        text = str(text)
        text = text.lower()
        text = self.url_pattern.sub(" URLTOKEN ", text)
        text = self.email_pattern.sub(" EMAILTOKEN ", text)
        text = self.money_pattern.sub(" MONEYTOKEN ", text)
        text = self.number_pattern.sub(" NUMTOKEN ", text)

        text = re.sub(r"([!?$€£%])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z!?$€£%]+", " ", text)

        text = text.split()

        text = [self.lemmatiser.lemmatize(word) for word in text if word not in self.stopwords_set or word in {"!", "?", "$", "€", "£", "%"}]
        text = ' '.join(text)
        return text
    
    def preprocess(self, data):
        if 'text' not in data.columns:
            raise ValueError("data must contain the text column")
        data["processed_text"] = data["text"].apply(self.preprocess_text)
        return data
    
 
    
