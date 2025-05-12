"""
Configuration settings for the chatbot application.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Determine if we are in a Docker environment or local development
IN_DOCKER = os.environ.get('IN_DOCKER', '0') == '1'

# Add the root directory to Python PATH to allow relative imports
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load environment variables
load_dotenv()

# Vector store settings - use local paths when not in Docker
if IN_DOCKER:
    DEFAULT_VECTOR_STORE_DIR = "/app/chroma_db"
    DEFAULT_DB_PATH = "/app/data/chat.db"
else:
    DEFAULT_VECTOR_STORE_DIR = os.path.join(str(root_dir), "chroma_db")
    DEFAULT_DB_PATH = os.path.join(str(root_dir), "data", "chat.db")

print('DEFAULT_DB_PATH', DEFAULT_DB_PATH)
print('DEFAULT_VECTOR_STORE_DIR', DEFAULT_VECTOR_STORE_DIR)

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", DEFAULT_VECTOR_STORE_DIR)
#VECTOR_STORE_DIR = DEFAULT_VECTOR_STORE_DIR

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # Default dimension for text-embedding-3-small

# Database settings
DB_PATH = os.getenv("DB_PATH", DEFAULT_DB_PATH)

# Search settings
DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", "8"))