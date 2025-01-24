#!/bin/bash

# Ensure the PORT environment variable is set (default to 8000 if not set)
PORT=${PORT:-8000}

# Start the Gunicorn server with Uvicorn worker
exec gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
