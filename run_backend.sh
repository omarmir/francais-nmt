#!/bin/bash
# Linux shell script to run FastAPI backend

# Go to the script's directory
cd "$(dirname "$0")"

# Run the FastAPI app
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
