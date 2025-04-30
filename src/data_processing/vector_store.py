import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Inicializa o gerenciador do banco de dados vetorial.
        
        Args:
            persist_directory: Diretório onde o banco de dados Chroma será armazenado
        """
        load_dotenv()
        
        self.persist_directory = persist_directory
        
        # Certificar que o diretório existe
        os.makedirs(persist_directory, exist_ok=True)
        
        # Configuração dos embeddings da OpenAI usando o modelo text-embedding-3-small
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536  # Dimensão padrão do modelo text-embedding-3-small
        )
        
        # Inicializar o banco de dados Chroma
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos usando a função de embedding configurada.
        
        Args:
            texts: Lista de strings para gerar embeddings
            
        Returns:
            Lista de vetores de embedding
        """
        return self.embedding_function.embed_documents(texts)
    
    def add_documents(self, documents: List[Document]):
        """
        Adiciona documentos ao banco de dados vetorial.
        
        Args:
            documents: Lista de documentos para adicionar ao banco de dados
        """
        # Adicionar os documentos ao Chroma
        self.vector_store.add_documents(documents)
        
        # Persistir o banco de dados
        self.vector_store.persist()
    
    def add_documents_with_embeddings(self, texts: List[str], embeddings: List[List[float]], 
                                      metadatas: List[Dict[str, Any]]):
        """
        Adiciona documentos com embeddings pré-calculados ao banco de dados vetorial.
        
        Args:
            texts: Lista de textos dos documentos
            embeddings: Lista de vetores de embedding pré-calculados
            metadatas: Lista de metadados associados aos documentos
        """
        # Adicionar ao Chroma usando o método correto
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        # Persistir o banco de dados
        self.vector_store.persist()
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Realiza uma busca semântica no banco de dados.
        
        Args:
            query: Consulta de texto
            k: Número de documentos a retornar
            
        Returns:
            Lista dos documentos mais similares
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Realiza uma busca semântica e retorna os documentos com suas pontuações.
        
        Args:
            query: Consulta de texto
            k: Número de documentos a retornar
            
        Returns:
            Lista de tuplas (documento, pontuação)
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre o banco de dados.
        
        Returns:
            Dicionário com estatísticas do banco de dados
        """
        collection = self.vector_store._collection
        return {
            "count": collection.count(),
            "collection_name": collection.name
        }