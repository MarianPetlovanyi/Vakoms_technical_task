import opendatasets
import pandas

"""
This script is created to download dataset for project. 
If you have kaggle.json file with credentials(Kaggle API token) 
you can copy it to this directory (./src/) and data will be passed automatically.
"""


dataset_url = "https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/data"
download_path = "../data/raw/"

opendatasets.download(dataset_url, data_dir=download_path)

