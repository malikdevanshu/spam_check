import os
from pathlib import Path
import requests
import tarfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io

def download_and_extract_files(page_url, path):
    os.makedirs(path, exist_ok=True )

    response = requests.get(page_url)
    response.raise_for_status

    soup = BeautifulSoup(response.text, "html.parser")

    zip_links = [
        urljoin(page_url, link["href"])
        for link in soup.find_all("a", href=True)
        if link["href"].lower().endswith(".tar.bz2")
    ]
        
    for zip_url in zip_links:

        zip_response = requests.get(zip_url)
        zip_response.raise_for_status()

        zip_data = io.BytesIO(zip_response.content)

        with tarfile.open(fileobj=zip_data, mode="r:bz2") as zip_ref:
            zip_ref.extractall(path)
    print("Done!")


         







