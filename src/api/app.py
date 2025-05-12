"""
Flask API to interact with the LangGraph chatbot.
"""
import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify

# Add the root directory to the Python PATH
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config.settings import DB_PATH
from src.chatbot.agent import graph_with_memory
from langchain_core.messages import HumanMessage

# Ensure the database directory exists
db_dir = os.path.dirname(DB_PATH)
os.makedirs(db_dir, exist_ok=True)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint to check if the API is running."""
    return jsonify({"status": "ok", "message": "Chatbot API is running"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to send messages to the chatbot and receive responses.
    
    Expects a JSON with the following format:
    {
        "message": "User's message",
        "thread_id": "Conversation ID (optional)"
    }
    """
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data["message"]
    thread_id = data.get("thread_id", "20")  # Use "20" as the default ID if not provided
    
    # Configure the conversation identifier
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create user message and invoke the graph
    messages = [HumanMessage(content=user_message)]
    
    try:
        output = graph_with_memory.invoke({"messages": messages}, config)
        app.logger.info(f"Output type: {type(output)}")
        app.logger.info(f"Output content: {output}")
        
        # Extract the content of the response messages
        response_messages = []
        
        # Add the current user message at the beginning of the response
        response_messages.append({
            "content": user_message,
            "type": "human"
        })
        
        # Add only the AI response message (last message)
        if output and "messages" in output and len(output["messages"]) > 0:
            # The last message is the AI's response to this input
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
        app.logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": f"Error processing your message: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    app.run(host="0.0.0.0", port=port, debug=debug)