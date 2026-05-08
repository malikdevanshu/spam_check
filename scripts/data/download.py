import io
import tarfile

from pathlib import Path
from urllib.parse import urljoin

import requests

from bs4 import BeautifulSoup


def download_and_extract_files(page_url, path):
    Path(path).mkdir(exist_ok=True, parents=True)

    response = requests.get(page_url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    tar_links = [
        urljoin(page_url, link["href"])
        for link in soup.find_all("a", href=True)
        if link["href"].lower().endswith(".tar.bz2")
    ]

    for tar_url in tar_links:
        tar_response = requests.get(tar_url, timeout=10)
        tar_response.raise_for_status()

        zip_data = io.BytesIO(tar_response.content)

        with tarfile.open(fileobj=zip_data, mode="r:bz2") as tar_ref:
            tar_ref.extractall(path, filter="data")

    print("Done!")
