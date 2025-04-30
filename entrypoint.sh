#!/bin/bash

# Criar diretórios necessários
mkdir -p /app/data
mkdir -p /app/chroma_db

# Verificar o modo de execução
if [ "$1" = "flask" ] || [ "$1" = "api" ]; then
    echo "Iniciando servidor API Flask..."
    python /app/src/api/app.py
elif [ "$1" = "streamlit" ] || [ "$1" = "web" ]; then
    echo "Iniciando interface web com Streamlit..."
    streamlit run /app/src/web/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
elif [ "$1" = "chatbot" ]; then
    echo "Iniciando chatbot em modo console..."
    python /app/main.py
elif [ -n "$1" ]; then
    # Se um comando for fornecido, execute-o
    exec "$@"
else
    # Comportamento padrão: iniciar ambos os serviços em paralelo
    echo "Iniciando todos os serviços..."
    python /app/src/api/app.py &
    streamlit run /app/src/web/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
fi