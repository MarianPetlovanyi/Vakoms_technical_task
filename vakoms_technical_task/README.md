# Vakoms Technical Task - Machine Learning Project

## Introduction

This repository contains the code and resources for the "vakoms_technical_task" machine learning project. The project aims to create machine learning model to detect obejct based on dataset provided by Vakoms. The goal is implement application with model that will classify object and predict bounding boxes.  


## Setting up a Python Virtual Environment

To create a virtual environment for this project, follow these steps:

1. Open your terminal or command prompt.

2. Navigate to the project's root directory:

   ```bash
   cd /path/to/vakoms_technical_task
   python -m venv venv
   ```
3. Activate virtual environment
   
   Windows
   ```bash
   .\venv\Scripts\activate
   ```
   Linux/macOS
    ```bash
    source venv/bin/activate
   ```

## Installation

To set up the project environment, you can use the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Downloading the Dataset

To obtain the dataset necessary for this project, follow these steps:

1. Open your terminal or command prompt.

2. Navigate to the project's root directory:

   ```bash
   cd /path/to/vakoms_technical_task/src
   python download_dataset.py
   ```

## Processing Dataset

To preprocess dataset, follow these steps:

1. Open your terminal or command prompt.

2. Navigate to the project's root directory:

   ```bash
   cd /path/to/vakoms_technical_task/src
   python process.py
   ```


## Usage

To preprocess dataset, follow these steps:

1. Open your terminal or command prompt.

2. Navigate to the project's root directory:

   ```bash
   cd /path/to/vakoms_technical_task/
   python detect.py demo.jpg       #replace demo.jpg
   ```