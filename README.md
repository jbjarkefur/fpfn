# README

## Overview

FPFN.ai is a tool for getting performance insights of computer vision models. Our mission is to make it much easier and faster to improve computer vision models and to empower more people to help out with data science work.

## Installation

The installation instructions assume that you have conda installed (e.g. miniconda). You can install without conda by manually installing packages according to the contents of environment.yml

## Starting the app

Perform the following steps to start the web app with a default "mock" Fracture Detection dataset (random values):

1. Clone this repository
2. Manually copy the folder "test_data_images" from Google Drive (ask the owner of this repository for access) to this repository folder. Note: this step is not needed if you use a custom dataset, see sub-chapter below.
3. `conda env create -f environment.yml`
4. `conda activate fpfn_env`
5. Start the backend: `uvicorn api:app --reload`
6. Open a new terminal and start the frontend: `streamlit run frontend.py`

### Custom dataset

To start the app with a custom dataset, just update the first lines of api.py: set load_custom_dataset to True and point to your ground truth and prediction files. Image ids and study ids can be either int or str.
