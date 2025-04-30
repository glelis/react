"""
Vector store implementation for semantic search capabilities.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Adicionar o diretório raiz ao PATH do Python para permitir importações relativas
# quando o script é executado diretamente
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from src.config.settings import VECTOR_STORE_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

class VectorStoreManager:
    def __init__(self, persist_directory: str = VECTOR_STORE_DIR):
        """
        Initializes the vector database manager.
        
        Args:
            persist_directory: Directory where the Chroma database will be stored
        """
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Configure OpenAI embeddings using the specified model
        self.embedding_function = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        # Initialize the Chroma database
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the configured embedding function.
        
        Args:
            texts: List of strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_function.embed_documents(texts)
    
    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the vector database.
        
        Args:
            documents: List of documents to add to the database
        """
        # Add documents to Chroma
        self.vector_store.add_documents(documents)
        
        # Persist the database
        self.vector_store.persist()
    
    def add_documents_with_embeddings(self, texts: List[str], embeddings: List[List[float]], 
                                     metadatas: List[Dict[str, Any]]):
        """
        Adds documents with pre-calculated embeddings to the vector database.
        
        Args:
            texts: List of document texts
            embeddings: List of pre-calculated embedding vectors
            metadatas: List of metadata associated with the documents
        """
        # Add to Chroma using pre-calculated embeddings
        self.vector_store.add_embeddings(texts, embeddings, metadatas)
        
        # Persist the database
        self.vector_store.persist()
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a semantic search in the database.
        
        Args:
            query: Text query
            k: Number of documents to return
            
        Returns:
            List of the most similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Performs a semantic search and returns documents with their scores.
        
        Args:
            query: Text query
            k: Number of documents to return
            
        Returns:
            List of tuples (document, score)
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        collection = self.vector_store._collection
        return {
            "count": collection.count(),
            "collection_name": collection.name
        }