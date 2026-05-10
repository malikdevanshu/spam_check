from pathlib import Path

from scripts.data.Ingestion import load_data, text_from_email


def test_text_from_email_reads_plain_text(tmp_path):
    email_file = tmp_path / "email.txt"
    email_file.write_text(
        "From: test@example.com\n" "Subject: Hello\n" "\n" "This is a test email.",
        encoding="utf-8",
    )

    result = text_from_email(email_file)

    assert "This is a test email." in result


def test_load_data_reads_easy_ham_and_spam(tmp_path):
    easy_ham = tmp_path / "easy_ham"
    spam = tmp_path / "spam"
    easy_ham.mkdir()
    spam.mkdir()

    (easy_ham / "ham1.txt").write_text(
        "From: a@example.com\n\nNormal message",
        encoding="utf-8",
    )
    (spam / "spam1.txt").write_text(
        "From: b@example.com\n\nWin money now",
        encoding="utf-8",
    )

    df = load_data(tmp_path)

    assert len(df) == 2
    assert set(df["label"]) == {0, 1}
    assert set(df.columns) == {"filename", "label", "text"}
