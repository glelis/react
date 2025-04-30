#!/usr/bin/env python3
"""
Script para executar os componentes da aplicação localmente.
"""
import os
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path

# Adicionar o diretório raiz ao PATH do Python
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def debug_paths():
    """Mostra informações de diagnóstico sobre caminhos usados."""
    from src.config.settings import DB_PATH, VECTOR_STORE_DIR
    
    print("=== INFORMAÇÕES DE DIAGNÓSTICO ===")
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Diretório do script: {current_dir}")
    print(f"Python path: {sys.path}")
    print(f"DB_PATH configurado: {DB_PATH}")
    print(f"VECTOR_STORE_DIR configurado: {VECTOR_STORE_DIR}")
    print(f"DB_PATH existe: {os.path.exists(DB_PATH)}")
    print(f"DB_PATH diretório pai existe: {os.path.exists(os.path.dirname(DB_PATH))}")
    print(f"VECTOR_STORE_DIR existe: {os.path.exists(VECTOR_STORE_DIR)}")
    print("================================")

def ensure_directories():
    """Cria os diretórios necessários para o funcionamento da aplicação."""
    print("Verificando e criando diretórios necessários...")
    # Criar diretório para o banco de dados SQLite
    data_dir = os.path.join(str(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Criar diretório para o ChromaDB
    chroma_dir = os.path.join(str(current_dir), "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    
    print(f"Diretórios criados/verificados: {data_dir}, {chroma_dir}")

def run_flask():
    """Executa o servidor Flask API."""
    print("Iniciando servidor Flask API em http://localhost:5000")
    
    # Configurar variáveis de ambiente com caminhos absolutos
    os.environ["FLASK_PORT"] = "5000"
    os.environ["FLASK_DEBUG"] = "True"
    
    # Usar caminhos explícitos
    db_path = os.path.join(str(current_dir), "data", "chat.db")
    vector_store_dir = os.path.join(str(current_dir), "chroma_db")
    os.environ["DB_PATH"] = db_path
    os.environ["VECTOR_STORE_DIR"] = vector_store_dir
    
    print(f"DB_PATH definido como: {db_path}")
    print(f"VECTOR_STORE_DIR definido como: {vector_store_dir}")
    
    # Executar o Flask em modo debug para ver mensagens de erro detalhadas
    flask_script = os.path.join(str(current_dir), "src", "api", "app.py")
    subprocess.run([sys.executable, flask_script], check=True)

def run_streamlit():
    """Executa a interface web Streamlit."""
    print("Iniciando interface Streamlit em http://localhost:8501")
    os.environ["API_URL"] = "http://localhost:5000"
    
    streamlit_script = os.path.join(str(current_dir), "src", "web", "streamlit_app.py")
    subprocess.run([
        "streamlit", "run", streamlit_script, 
        "--server.port=8501", 
        "--server.address=0.0.0.0"
    ], check=True)

def run_chatbot():
    """Executa o chatbot em modo console."""
    print("Iniciando chatbot em modo console")
    main_script = os.path.join(str(current_dir), "main.py")
    
    # Usar caminhos explícitos
    db_path = os.path.join(str(current_dir), "data", "chat.db")
    vector_store_dir = os.path.join(str(current_dir), "chroma_db")
    os.environ["DB_PATH"] = db_path
    os.environ["VECTOR_STORE_DIR"] = vector_store_dir
    
    subprocess.run([sys.executable, main_script], check=True)

def run_all():
    """Executa todos os componentes em threads separadas."""
    # Mostrar informações de diagnóstico
    debug_paths()
    
    # Garantir que os diretórios existam
    ensure_directories()
    
    # Iniciar API Flask em uma thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Aguardar um momento para que a API seja iniciada
    time.sleep(2)
    
    # Iniciar Streamlit
    run_streamlit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa os componentes da aplicação localmente.")
    parser.add_argument("component", choices=["api", "web", "chatbot", "all"], 
                        default="all", nargs="?",
                        help="Componente para executar (api, web, chatbot ou all)")
    
    args = parser.parse_args()
    
    # Mostrar informações de diagnóstico
    debug_paths()
    
    # Sempre garantir que os diretórios existam
    ensure_directories()
    
    if args.component == "api":
        run_flask()
    elif args.component == "web":
        run_streamlit()
    elif args.component == "chatbot":
        run_chatbot()
    else:
        run_all()