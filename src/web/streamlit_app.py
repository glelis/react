"""
Interface Streamlit para o chatbot LangGraph.
"""
import os
import sys
import json
import uuid
import requests
from pathlib import Path
import streamlit as st

# Adicionar o diretório raiz ao PATH do Python
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Importar a função clear_agent_memory para limpar o banco de dados
from src.database.chat_memory import clear_agent_memory

# Configurar a página Streamlit
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Verificar se API_URL está nas configurações ou usar padrão
def get_api_url():
    return os.environ.get("API_URL", "http://localhost:5000")

# Inicializar variáveis de estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# Título do aplicativo
st.title("🤖 LangGraph Chatbot")

# Botão para testar conexão com API
with st.sidebar:
    if st.button("Testar Conexão com API"):
        try:
            response = requests.get(f"{get_api_url()}/health", timeout=5)
            if response.status_code == 200:
                st.success(f"API disponível: {response.json().get('message', '')}")
            else:
                st.error(f"API retornou erro: {response.status_code}")
        except requests.RequestException as e:
            st.error(f"Não foi possível conectar à API: {e}")

# Exibir mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
user_input = st.chat_input("Digite sua mensagem...")

if user_input:
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Exibir mensagem do usuário
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Preparar mensagem para o chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")
        
        try:
            # Enviar mensagem para a API
            api_url = f"{get_api_url()}/chat"
            
            # Depuração - exibir o que está sendo enviado para a API
            st.sidebar.write("Enviando para API:", {"message": user_input, "thread_id": st.session_state.thread_id})
            
            response = requests.post(
                api_url,
                json={"message": user_input, "thread_id": st.session_state.thread_id},
                timeout=60
            )
            
            # Depuração - exibir resposta bruta da API
            st.sidebar.write("Código de status da API:", response.status_code)
            st.sidebar.write("Resposta da API (raw):", response.text)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Depuração - exibir dados estruturados
                    st.sidebar.write("Dados da API (parsed):", data)
                    
                    # Atualizar thread_id se fornecido
                    if "thread_id" in data:
                        st.session_state.thread_id = data["thread_id"]
                    
                    # Processar resposta
                    if "response" in data and isinstance(data["response"], list) and len(data["response"]) > 0:
                        # Procurar apenas pela resposta do assistente (tipo "ai")
                        ai_responses = [msg for msg in data["response"] if isinstance(msg, dict) and msg.get("type") == "ai"]
                        
                        if ai_responses:
                            # Usar apenas a última resposta do assistente
                            bot_message = ai_responses[-1]["content"]
                        else:
                            # Se não houver resposta do tipo "ai", usar a última resposta
                            bot_response = data["response"][-1]
                            if isinstance(bot_response, dict) and "content" in bot_response:
                                bot_message = bot_response["content"]
                            else:
                                bot_message = str(bot_response)
                                st.sidebar.warning("Formato de resposta inesperado")
                    else:
                        bot_message = "Não recebi uma resposta clara. Tente novamente."
                    
                    # Mostrar a resposta e atualizar o histórico
                    message_placeholder.markdown(bot_message)
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})
                    
                except Exception as e:
                    st.sidebar.error(f"Erro ao processar resposta JSON: {e}")
                    message_placeholder.markdown(f"❌ Erro ao processar resposta: {str(e)}")
            else:
                error_msg = f"Erro na API: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except:
                        error_msg = f"Erro na API: {response.text}"
                
                message_placeholder.markdown(f"❌ {error_msg}")
        
        except requests.exceptions.RequestException as e:
            message_placeholder.markdown(f"❌ Erro de conexão: {str(e)}")
            st.error(f"Não foi possível conectar à API em {get_api_url()}. Verifique se o servidor está em execução.")

# Barra lateral com informações
with st.sidebar:
    st.subheader("Sobre o Chatbot")
    st.write("Este chatbot é construído com LangGraph e utiliza um modelo de linguagem da OpenAI para fornecer respostas em tempo real.")
    
    st.subheader("Estado da Conversa")
    st.write(f"ID da conversa: {st.session_state.thread_id}")
    
    if st.button("Nova Conversa"):
        # Gerar um novo ID para a conversa
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()
    
    st.subheader("Configurações")
    api_url = st.text_input("URL da API:", value=get_api_url())
    if api_url and api_url != get_api_url():
        os.environ["API_URL"] = api_url
        st.success(f"URL da API atualizada para: {api_url}")
        st.rerun()
    
    # Limpar estado da sessão e memória do agente se necessário
    if st.button("Limpar Cache da Sessão"):
        # Salvar thread_id antes de limpar o cache para poder usá-lo para limpar a memória do agente
        thread_id = st.session_state.thread_id if "thread_id" in st.session_state else None
        
        # Limpar o cache da sessão
        #for key in list(st.session_state.keys()):
        #    del st.session_state[key]
            
        # Limpar a memória do agente no banco de dados
        #if thread_id:
        records_removed = clear_agent_memory()
        st.success(f"Cache da sessão e memória do agente limpos! ({records_removed} registros removidos do banco de dados)")
        #else:
        #    st.success("Cache da sessão limpo!")
            
        st.rerun()