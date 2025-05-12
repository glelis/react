#!/usr/bin/env python3
"""
Script to run the application components locally.
"""
import os
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path

# Add the root directory to the Python PATH
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def debug_paths():
    """Displays diagnostic information about paths used."""
    from src.config.settings import DB_PATH, VECTOR_STORE_DIR
    
    print("=== DIAGNOSTIC INFORMATION ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print(f"Python path: {sys.path}")
    print(f"Configured DB_PATH: {DB_PATH}")
    print(f"Configured VECTOR_STORE_DIR: {VECTOR_STORE_DIR}")
    print(f"DB_PATH exists: {os.path.exists(DB_PATH)}")
    print(f"DB_PATH parent directory exists: {os.path.exists(os.path.dirname(DB_PATH))}")
    print(f"VECTOR_STORE_DIR exists: {os.path.exists(VECTOR_STORE_DIR)}")
    print("================================")

def ensure_directories():
    """Ensures necessary directories exist."""
    print("Checking and creating necessary directories...")
    # Create directory for SQLite database
    data_dir = os.path.join(str(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create directory for ChromaDB
    chroma_dir = os.path.join(str(current_dir), "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    
    print(f"Directories created/verified: {data_dir}, {chroma_dir}")

def run_flask():
    """Runs the Flask server."""
    print("Starting Flask API server at http://localhost:5000")
    
    # Configure environment variables with absolute paths
    os.environ["FLASK_PORT"] = "5000"
    os.environ["FLASK_DEBUG"] = "True"
    
    # Use explicit paths
    db_path = os.path.join(str(current_dir), "data", "chat.db")
    vector_store_dir = os.path.join(str(current_dir), "chroma_db")
    os.environ["DB_PATH"] = db_path
    os.environ["VECTOR_STORE_DIR"] = vector_store_dir
    
    print(f"DB_PATH set to: {db_path}")
    print(f"VECTOR_STORE_DIR set to: {vector_store_dir}")
    
    # Run Flask in debug mode to see detailed error messages
    flask_script = os.path.join(str(current_dir), "src", "api", "app.py")
    subprocess.run([sys.executable, flask_script], check=True)

def run_streamlit():
    """Runs the Streamlit interface."""
    print("Starting Streamlit interface at http://localhost:8501")
    os.environ["API_URL"] = "http://localhost:5000"
    
    streamlit_script = os.path.join(str(current_dir), "src", "web", "streamlit_app.py")
    subprocess.run([
        "streamlit", "run", streamlit_script, 
        "--server.port=8501", 
        "--server.address=0.0.0.0"
    ], check=True)

def run_chatbot():
    """Runs the chatbot in console mode."""
    print("Starting chatbot in console mode")
    main_script = os.path.join(str(current_dir), "main.py")
    
    # Use explicit paths
    db_path = os.path.join(str(current_dir), "data", "chat.db")
    vector_store_dir = os.path.join(str(current_dir), "chroma_db")
    os.environ["DB_PATH"] = db_path
    os.environ["VECTOR_STORE_DIR"] = vector_store_dir
    
    subprocess.run([sys.executable, main_script], check=True)

def run_all():
    """Runs all components."""
    # Display diagnostic information
    debug_paths()
    
    # Ensure directories exist
    ensure_directories()
    
    # Start Flask API in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Wait a moment for the API to start
    time.sleep(2)
    
    # Start Streamlit
    run_streamlit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the application components locally.")
    parser.add_argument("component", choices=["api", "web", "chatbot", "all"], 
                        default="all", nargs="?",
                        help="Component to run (api, web, chatbot, or all)")
    
    args = parser.parse_args()
    
    # Display diagnostic information
    debug_paths()
    
    # Always ensure directories exist
    ensure_directories()
    
    if args.component == "api":
        run_flask()
    elif args.component == "web":
        run_streamlit()
    elif args.component == "chatbot":
        run_chatbot()
    else:
        run_all()