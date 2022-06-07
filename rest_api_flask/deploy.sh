#!/bin/bash

tmux new -s session_1

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python controller.py
