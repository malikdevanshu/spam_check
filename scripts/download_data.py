from pathlib import Path
import urllib.request
import tarfile

data_dir = Path(__file__).resolve().parent / "data" / "raw" 
data_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://spamassassin.apache.org/old/publiccorpus/"

files = [
    "20030228_easy_ham.tar.bz2",
    "20030228_spam.tar.bz2"
]

for file in files:
    url = base_url + file
    download_path = data_dir / file

    urllib.request.urlretrieve(url, download_path)

    with tarfile.open(download_path, "r:bz2") as tar:
        tar.extractall(data_dir)

