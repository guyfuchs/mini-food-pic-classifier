import os
import zipfile

from pathlib import Path

import requests

def download_data(project_path: str,
                  data_subpath: str,
                  file_name: str,
                  source_url: str,
                  data_sesc: str):

    # Setup path to data folder
    data_path = Path(os.path.join(project_path, "data/"))
    image_path = data_path / data_subpath

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / file_name, "wb") as f:
        request = requests.get(source_url)
        print(f"Downloading {data_sesc}...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / file_name, "r") as zip_ref:
        print(f"Unzipping {data_sesc}...")
        zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(data_path / file_name)
    
    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    return train_dir, test_dir
