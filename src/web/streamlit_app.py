"""
Streamlit interface for the LangGraph chatbot.
"""
import os
import sys
import json
import uuid
import requests
from pathlib import Path
import streamlit as st

# Add the root directory to the Python PATH
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import the clear_agent_memory function to clear the database
from src.database.chat_memory import clear_agent_memory

# Configure the Streamlit page
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon=":fox_face:",
    layout="centered"
)

# Check if API_URL is in the settings or use the default
def get_api_url():
    return os.environ.get("API_URL", "http://localhost:5000")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# Application title
st.title(":fox_face: ReAct RAG Chatbot")

# Button to test API connection
with st.sidebar:
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{get_api_url()}/health", timeout=5)
            if response.status_code == 200:
                st.success(f"API available: {response.json().get('message', '')}")
            else:
                st.error(f"API returned error: {response.status_code}")
        except requests.RequestException as e:
            st.error(f"Unable to connect to the API: {e}")

# Display messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to the history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user", avatar='🐸'):
        st.markdown(user_input)
    
    # Prepare message for the chatbot
    with st.chat_message("assistant", avatar='🦊'):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Send message to the API
            api_url = f"{get_api_url()}/chat"
            
            response = requests.post(
                api_url,
                json={"message": user_input, "thread_id": st.session_state.thread_id},
                timeout=60
            )
            
            # Debugging - display raw API response
            st.sidebar.write("API status code:", response.status_code)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Update thread_id if provided
                    if "thread_id" in data:
                        st.session_state.thread_id = data["thread_id"]
                    
                    # Process response
                    if "response" in data and isinstance(data["response"], list) and len(data["response"]) > 0:
                        # Look only for assistant responses (type "ai")
                        ai_responses = [msg for msg in data["response"] if isinstance(msg, dict) and msg.get("type") == "ai"]
                        
                        if ai_responses:
                            # Use only the last assistant response
                            bot_message = ai_responses[-1]["content"]
                        else:
                            # If no response of type "ai", use the last response
                            bot_response = data["response"][-1]
                            if isinstance(bot_response, dict) and "content" in bot_response:
                                bot_message = bot_response["content"]
                            else:
                                bot_message = str(bot_response)
                                st.sidebar.warning("Unexpected response format")
                    else:
                        bot_message = "I didn't receive a clear response. Please try again."
                    
                    # Show the response and update the history
                    message_placeholder.markdown(bot_message)
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing JSON response: {e}")
                    message_placeholder.markdown(f"❌ Error processing response: {str(e)}")
            else:
                error_msg = f"API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except:
                        error_msg = f"API error: {response.text}"
                
                message_placeholder.markdown(f"❌ {error_msg}")
        
        except requests.exceptions.RequestException as e:
            message_placeholder.markdown(f"❌ Connection error: {str(e)}")
            st.error(f"Unable to connect to the API at {get_api_url()}. Please check if the server is running.")

# Sidebar with information
with st.sidebar:
    
    st.subheader("Conversation State")
    st.write(f"Conversation ID: {st.session_state.thread_id}")
    
    if st.button("New Conversation"):
        # Generate a new conversation ID
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()
    
    st.subheader("Settings")
    api_url = st.text_input("API URL:", value=get_api_url())
    if api_url and api_url != get_api_url():
        os.environ["API_URL"] = api_url
        st.success(f"API URL updated to: {api_url}")
        st.rerun()
    
    # Clear session state and agent memory if necessary
    if st.button("Clear Session Cache"):
        # Save thread_id before clearing the cache to use it for clearing agent memory
        thread_id = st.session_state.thread_id if "thread_id" in st.session_state else None

        records_removed = clear_agent_memory()
        st.success(f"Session cache and agent memory cleared! ({records_removed} records removed from the database)")

            
        st.rerun()