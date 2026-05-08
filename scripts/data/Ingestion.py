import contextlib
import pathlib

from email import policy
from email.parser import BytesParser

import pandas as pd


def text_from_email(path):
    with pathlib.Path(path).open("rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                with contextlib.suppress(Exception):
                    parts.append(part.get_content())

    else:
        with contextlib.suppress(Exception):
            parts.append(msg.get_content())

    return "\n".join(parts)


def load_data(data_path):

    folders = {"easy_ham": 0, "spam": 1}

    rows = []

    for folder_name, label in folders.items():
        folder_path = data_path / folder_name

        for file_path in folder_path.iterdir():
            if file_path.is_file():
                text = text_from_email(file_path)

                rows.append(
                    {"filename": str(file_path), "label": label, "text": text}
                )

    df = pd.DataFrame(rows)

    return df[df["text"].str.strip() != ""]
