"""
Implementation of the LangGraph-based conversational agent.
"""
import os
import sys
from pathlib import Path

# Add the root directory to the Python PATH to allow relative imports when the script is executed directly
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from src.database import VectorStoreManager
from src.config.settings import LLM_MODEL, DB_PATH, DEFAULT_SEARCH_RESULTS

# Ensure the database directory exists
db_dir = os.path.dirname(DB_PATH)
print(f"DB_DIR: {db_dir}")
print(f"DB_PATH: {DB_PATH}")
#os.makedirs(db_dir, exist_ok=True)

# Initialize the database connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Initialize the memory checkpointer
memory = SqliteSaver(conn)

# Initialize the LLM
model = ChatOpenAI(model=LLM_MODEL, temperature=0)

# State class to store messages and summary
class State(MessagesState):
    summary: str

def call_model(state: State):
    """
    Calls the LLM with the current state and returns the response.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with LLM response
    """
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it to messages
    if summary:
        # Add summary to system message
        system_message = f"You are a helpful Intelligent Contract Template Selector assistant specializing in Non-disclosure agreement (NDA). Use the search_documents tool to retrieve relevant information about NDA contracts. Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

def retrieve_search_results(query: str) -> dict:
    """
    Search for examples of Non-disclosure agreement contracts.
    It retrieves the top-k results along with their similarity scores and returns them in a
    structured format with metadata.

    Args:
        query (str): The search query string.

    Returns:
        dict: A dictionary containing formatted search results with content, filename, 
              page information and similarity scores.
    """
    vector_db = VectorStoreManager()
    results = vector_db.search_with_score(query, k=DEFAULT_SEARCH_RESULTS)
    
    formatted_results = []
    
    for doc, score in results:
        result = {
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", "Unknown"),
            "score": score
        }
        
        # Add page label if available
        if "page_label" in doc.metadata:
            result["page_label"] = doc.metadata["page_label"]
        
        # Add page number if available
        if "page" in doc.metadata:
            result["page"] = doc.metadata["page"]
            
        formatted_results.append(result)
    
    return {
        "search_results": formatted_results,
        "query": query
    }

def should_continue(state: State):
    """
    Determines whether to end or summarize the conversation.
    
    Args:
        state: Current conversation state
        
    Returns:
        Next node to execute
    """
    messages = state["messages"]
    
    # If there are more than twelve messages, then we summarize the conversation
    if len(messages) > 12:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

def summarize_conversation(state: State):
    """
    Summarizes the conversation to reduce token usage.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with summary and reduced message history
    """
    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages and add our summary to the state 
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Define the tools
tools = [retrieve_search_results]
model = model.bind_tools(tools)

# Define the graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges('conversation', tools_condition)
workflow.add_edge("tools", "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile the graph
graph = workflow.compile()
graph_with_memory = workflow.compile(checkpointer=memory)

def chat():
    """
    Interactive chat loop for the conversational agent.
    """
    config = {"configurable": {"thread_id": "2"}}  # Use a thread ID for persistent memory
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Create a HumanMessage and invoke the graph
        messages = [HumanMessage(content=user_input)]
        output = graph_with_memory.invoke({"messages": messages}, config)
        
        # Print the chatbot's response
        for m in output['messages']:
            m.pretty_print()

# Allow the script to be executed directly
if __name__ == "__main__":
    chat()