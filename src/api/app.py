"""
API Flask para interagir com o chatbot LangGraph.
"""
import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify

# Adicionar o diretório raiz ao PATH do Python
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config.settings import DB_PATH
from src.chatbot.agent import graph_with_memory
from langchain_core.messages import HumanMessage

# Garantir que o diretório do banco de dados exista
db_dir = os.path.dirname(DB_PATH)
os.makedirs(db_dir, exist_ok=True)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar se a API está funcionando."""
    return jsonify({"status": "ok", "message": "API do chatbot está funcionando"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint para enviar mensagens ao chatbot e receber respostas.
    
    Espera um JSON com o seguinte formato:
    {
        "message": "Mensagem do usuário",
        "thread_id": "ID da conversa (opcional)"
    }
    """
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "A mensagem é obrigatória"}), 400
    
    user_message = data["message"]
    thread_id = data.get("thread_id", "20")  # Usar "20" como ID padrão se não for fornecido
    
    # Configurar o identificador da conversa
    config = {"configurable": {"thread_id": thread_id}}
    
    # Criar mensagem do usuário e invocar o grafo
    messages = [HumanMessage(content=user_message)]
    
    try:
        output = graph_with_memory.invoke({"messages": messages}, config)
        app.logger.info(f"Output type: {type(output)}")
        app.logger.info(f"Output content: {output}")
        
        # Extrair o conteúdo das mensagens de resposta
        response_messages = []
        
        # Adicionar a mensagem do usuário atual ao início da resposta
        response_messages.append({
            "content": user_message,
            "type": "human"
        })
        
        # Adicionar apenas a mensagem de resposta do AI (última mensagem)
        if output and "messages" in output and len(output["messages"]) > 0:
            # A última mensagem é a resposta do AI para esta entrada
            last_message = output["messages"][-1]
            response_messages.append({
                "content": last_message.content,
                "type": last_message.type
            })
        
        response_data = {
            "response": response_messages,
            "thread_id": thread_id
        }
        
        app.logger.info(f"Final response: {json.dumps(response_data)}")
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Erro ao processar mensagem: {str(e)}")
        return jsonify({"error": f"Erro ao processar sua mensagem: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    app.run(host="0.0.0.0", port=port, debug=debug)