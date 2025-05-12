# ReAct RAG Chatbot

A conversational ReAct (Reasoning and Action) chatbot powered by LangGraph with semantic search capabilities over a vector knowledge base. It includes a Streamlit web interface and a Flask API, with support for multiple concurrent users.

## Overview

This project implements a ReAct-based conversational agent using the LangGraph library alongside OpenAI models. The chatbot offers the following features:

- Interactive conversation using GPT-4o with ReAct capabilities  
- Semantic search over a vector knowledge base (ChromaDB)  
- Automatic summarization of long conversations  
- Persistent state between user interactions via SQLite  
- User-friendly web interface with Streamlit  
- REST API built with Flask for system integration  
- Multi-user support with isolated session contexts  
- Full document processing pipeline for embedding generation  

## Project Structure

```
.
├── src/                          # Main source code
│   ├── api/                      # Flask API
│   │   └── app.py                # API server implementation
│   ├── chatbot/                  # Chatbot module
│   │   ├── __init__.py           # Chatbot components export
│   │   └── agent.py              # ReAct agent implementation
│   ├── config/                   # Configuration settings
│   │   └── settings.py           # Centralized settings
│   ├── data_processing/          # Document processing module
│   │   ├── document_processor.py # Processing of various document formats
│   │   ├── json_serializer.py    # Document and embedding serialization
│   │   ├── main.py               # Main processing script
│   │   └── query_tool.py         # Query tool for the vector store
│   ├── database/                 # Database module
│   │   ├── __init__.py
│   │   └── vector_store.py       # Vector DB interface
│   └── web/                      # Streamlit web app
│       └── streamlit_app.py      # Web interface implementation
├── main.py                       # Console chatbot entry point
├── run_local.py                  # Local launcher script
├── requirements.txt              # Project dependencies
├── langgraph.json                # LangGraph configuration
├── .env.example                  # Environment variable example
├── Dockerfile                    # Container build configuration
├── entrypoint.sh                 # Docker entrypoint script
├── docker-compose.yml            # Docker Compose configuration
└── README.md                     # Project documentation
```

## Requirements

- Python 3.12
- OpenAI API key
- Optional: Docker and Docker Compose for containerization

## Installation

### Local Environment

1. Clone the repository  
2. Create a virtual environment:
   ```bash
   python -m venv venv_react
   source venv_react/bin/activate  # Linux/Mac
   venv_react\Scripts\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and update the values as needed:
   ```bash
   cp .env.example .env  # Linux/Mac
   copy .env.example .env  # Windows
   ```

### Docker Environment

1. Build and run the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```
2. Access the application:
   - Streamlit interface: [http://localhost:8501](http://localhost:8501)
   - Flask API: [http://localhost:5000](http://localhost:5000)

## Configuration

Set the following environment variables in the `.env` file:

```
# OpenAI configuration
OPENAI_API_KEY=your_api_key_here

# Vector database settings
VECTOR_STORE_DIR=/app/chroma_db

# LLM model settings
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# SQLite DB path
DB_PATH=/app/data/chat.db

# Search configuration
DEFAULT_SEARCH_RESULTS=8
```

## Usage

### Running Locally

You can launch the application in several modes using the `run_local.py` script:

1. **Run all components** (Flask API + Streamlit interface):
   ```bash
   python run_local.py
   ```

2. **Run only the Flask API**:
   ```bash
   python run_local.py api
   ```

3. **Run only the Streamlit web app** (requires the API to be running):
   ```bash
   python run_local.py web
   ```

4. **Run the chatbot in console mode**:
   ```bash
   python run_local.py chatbot
   # or
   python main.py
   ```

### Streamlit Interface

1. Open the Streamlit interface in your browser: [http://localhost:8501](http://localhost:8501)
2. Interact with the chatbot by typing your messages in the input box.

### Flask API

1. Send requests to the Flask API at [http://localhost:5000](http://localhost:5000).
2. Example request using `curl`:
   ```bash
   curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "Hello!"}'
   ```

### Document Processing & Embedding Generation

The project includes a complete pipeline for document processing and vector database creation:

1. **Basic document processing**:
   ```bash
   python src/data_processing/main.py --dir data_raw --chunks_dir chunks --embeddings_dir embeddings --db_dir chroma_db
   ```

2. **Processing specific file types**:
   ```bash
   python src/data_processing/main.py --dir data_raw --extensions txt,pdf,html
   ```

3. **Test queries after processing**:
   ```bash
   python src/data_processing/main.py --dir data_raw --query "your test query here"
   ```

4. **Direct query to the vector store**:
   ```bash
   python src/data_processing/query_tool.py --db_dir chroma_db --query "your query" --k 5 --with_score
   ```

The pipeline performs:
1. Text chunking of documents  
2. Embedding generation per chunk  
3. Storage in ChromaDB vector store  

Supported formats: TXT, HTML, PDF (non-OCR)

### Running with Docker

To run the system using Docker, you can:

1. **Run all services**:
   ```bash
   docker compose up --build
   ```

2. **Run only the API service**:
   ```bash
   docker compose up api
   ```

3. **Run only the web interface**:
   ```bash
   docker compose up web
   ```

Once running, access:
- **REST API**: http://localhost:5000  
- **Streamlit Web Interface**: http://localhost:8501  

## Key Components

### 1. ReAct Chatbot Agent (`src/chatbot/agent.py`)

Implements the ReAct agent using LangGraph with:
- Reasoning and action loop for query handling  
- Conversation summarization system  
- Integrated search tools  
- State management for ongoing chats  

### 2. Vector Store Manager (`src/database/vector_store.py` & `src/data_processing/vector_store.py`)

Handles vector database operations with ChromaDB:
- Document storage and retrieval  
- Semantic similarity search  
- Embedding management  

### 3. Document Processor (`src/data_processing/document_processor.py`)

Handles various document formats:
- Text extraction from PDF, HTML, TXT  
- Chunking into appropriate sizes  
- Metadata enrichment  

### 4. JSON Serializer (`src/data_processing/json_serializer.py`)

Manages serialization and storage of:
- Processed document chunks  
- Corresponding embeddings  

### 5. Flask API (`src/api/app.py`)

Implements a RESTful API to:
- Send messages to the chatbot  
- Receive responses  
- Manage sessions using `thread_id`  
- Support multiple concurrent users  

### 6. Streamlit Interface (`src/web/streamlit_app.py`)

Provides a web UI to:
- Chat with the agent  
- View message history  
- Start new conversations  
- Configure API URL  

### 7. Settings (`src/config/settings.py`)

Centralized configuration, including:
- Directory paths  
- Model settings  
- Search parameters  

## Multi-user Support

The system supports multiple users simultaneously via:

- Unique `thread_id`-based session management  
- Context isolation across conversations  
- Persistent state per user  
- Concurrency control for API access  

## Docker

The project includes full Docker support:

- **Dockerfile**: Defines the app execution environment  
- **docker-compose.yml**: Orchestrates services (API & Web UI) and volumes  
- **entrypoint.sh**: Container startup script  

Persistent volumes are used for:
- `./chroma_db`: Vector database  
- `./data`: SQLite database  
- `./chunks`: Processed document chunks  
- `./embeddings`: Generated embeddings  

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
