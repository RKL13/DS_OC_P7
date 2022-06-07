#!/bin/bash

# Set up and activate a virtual environment

python3 -m venv venv
source venv/bin/activate

# Install necessary packages

pip install -r requirements.txt

# Launch the server

streamlit run index.py