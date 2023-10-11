import opendatasets
import pandas

dataset_url = "https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/data"
download_path = "../data/raw/"

opendatasets.download(dataset_url, data_dir=download_path)

