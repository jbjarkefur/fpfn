# README

## Overview

FPFN.ai is a tool for getting performance insights of computer vision models. Our mission is to make it easier and faster to improve computer vision models and to empower more people to help out with data science work.

## Installation

The installation instructions assume that you have conda installed (e.g. miniconda). You can install without conda by manually installing packages according to the contents of environment.yml

Perform the following steps to start the web app with a default "mock" Fracture Detection dataset (random values):

1. Clone this repository
2. `conda env create -f environment.yml`
3. `conda activate fpfn_env`
4. Start the backend: `uvicorn api:app --reload`
5. Start the frontend (in a new terminal): `streamlit run frontend.py`

## Other

The backend can be tested separately by navigating to http://localhost:8000/docs
