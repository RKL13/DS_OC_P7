#!/bin/bash

# Set up and activate a virtual environment

python3 -m venv venv
source venv/bin/activate

# Install necessary packages

cat requirements.txt | xargs -n 1 pip install

# Launch the client

streamlit run index.py