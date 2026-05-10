import pandas as pd
import pytest

from scripts.data.preprocess import Preprocessor


def test_preprocess_adds_processed_text_column(sample_dataframe):
    preprocessor = Preprocessor()

    result = preprocessor.preprocess(sample_dataframe)

    assert "processed_text" in result.columns
    assert len(result) == len(sample_dataframe)


def test_preprocess_text_lowercases_text():
    preprocessor = Preprocessor()

    result = preprocessor.preprocess_text("HELLO WORLD")

    assert result == "hello world"


def test_preprocess_text_replaces_url():
    preprocessor = Preprocessor()

    result = preprocessor.preprocess_text("Visit https://example.com now")

    assert "URLTOKEN" in result


def test_preprocess_text_replaces_email():
    preprocessor = Preprocessor()

    result = preprocessor.preprocess_text("Email me at person@example.com")

    assert "EMAILTOKEN" in result


def test_preprocess_text_replaces_money():
    preprocessor = Preprocessor()

    result = preprocessor.preprocess_text("Win $1000 today")

    assert "MONEYTOKEN" in result


def test_preprocess_text_replaces_numbers():
    preprocessor = Preprocessor()

    result = preprocessor.preprocess_text("Call me at 12345")

    assert "NUMTOKEN" in result


def test_preprocess_raises_error_without_text_column():
    preprocessor = Preprocessor()
    data = pd.DataFrame({"message": ["hello"]})

    with pytest.raises(ValueError, match="text column"):
        preprocessor.preprocess(data)
