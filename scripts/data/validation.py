from scripts.data.Ingestion import load_data
def class_distribution():
    data = load_data()
    target = data["label"].value_counts()
    print(f"Number of Non Spam Emails : {target.get(0, 0)}")
    print(f"Number of Spam Emails: {target.get(1, 0)}")

def avg_text_length():
    data = load_data()
    data["text_len"] = data["text"].apply(len)
    avg_spam = data[data["label"] == 1]["text_len"].mean()
    print(f"average length of spam emails: {avg_spam}")
    avg_nospam =   data[data["label"] == 0]["text_len"].mean()
    print(f"Average length of no spam emails: {avg_nospam} ")

def feature_target_split():
    data = load_data()
    X = data["text"]
    print(f"length of feature X: {X.shape}")
    y = data["label"]
    print(f"length of target y : {y.shape}")


    

