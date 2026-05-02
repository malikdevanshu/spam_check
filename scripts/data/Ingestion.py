from scripts.configure.config import load_config
from email import policy
from email.parser import BytesParser
import pandas as pd
from pathlib import Path


def text_from_email(path):
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

  
    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    parts.append(part.get_content())
                except Exception:
                    pass
    else:
        try:
            parts.append(msg.get_content())
        except Exception:
            pass

    return "\n".join(parts)

def load_data():
    config = load_config()
    data_dir = Path(config["paths"]["raw_data_path"])

    folders = {
        "easy_ham": 0,
        "spam": 1
    }

    rows = []

    for folder_name, label in folders.items():
        folder_path = data_dir / folder_name

        for file_path in folder_path.iterdir():
            if file_path.is_file():
                text = text_from_email(file_path)

                rows.append({
                    "filename": str(file_path),
                    "label": label,
                    "text": text
                })

    df = pd.DataFrame(rows)
    df = df[df["text"].str.strip() != ""]


    return df