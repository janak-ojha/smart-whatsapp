#!/bin/bash

# Apply database migrations using Alembic
echo "Applying database migrations..."
alembic upgrade head

# Start the FastAPI application with uvicorn
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload