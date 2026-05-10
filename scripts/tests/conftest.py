import pandas as pd
import pytest


@pytest.fixture
def sample_texts():
    return [
        "Win $1000 now!!! Visit https://spam.com",
        "Hello, can we meet tomorrow?",
        "Contact me at test@example.com",
        "Your invoice is 123 euros",
        "FREE prize claim now",
        "Project meeting schedule attached",
        "tHIS I SJUST AN important reminder for dinner tonight "
        "maybe the invoice is not correct"
        "Free iphone in the walmart just login",
    ]


@pytest.fixture
def sample_labels():
    return [1, 0, 0, 0, 1, 0]


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "text": [
                "Win $1000 now!!! Visit https://spam.com",
                "Hello, can we meet tomorrow?",
            ],
            "label": [1, 0],
        }
    )
