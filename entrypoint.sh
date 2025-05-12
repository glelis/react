#!/bin/bash

# Create necessary directories
mkdir -p /app/data
mkdir -p /app/chroma_db

# Check execution mode
if [ "$1" = "flask" ] || [ "$1" = "api" ]; then
    echo "Starting Flask API server..."
    python /app/src/api/app.py
elif [ "$1" = "streamlit" ] || [ "$1" = "web" ]; then
    echo "Starting Streamlit web interface..."
    streamlit run /app/src/web/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
elif [ "$1" = "chatbot" ]; then
    echo "Starting chatbot in console mode..."
    python /app/main.py
elif [ -n "$1" ]; then
    # If a command is provided, execute it
    exec "$@"
else
    # Default behavior: start both services in parallel
    echo "Starting all services..."
    python /app/src/api/app.py &
    streamlit run /app/src/web/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
fi