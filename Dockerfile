FROM python:3.12-slim

WORKDIR /app

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código fonte
COPY . .

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV IN_DOCKER=1
ENV FLASK_DEBUG=False
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Tornar o script de entrypoint executável
RUN chmod +x /app/entrypoint.sh

# Expor portas para Flask e Streamlit
EXPOSE 5000 8501

# Usar o script de entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]