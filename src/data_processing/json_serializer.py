import os
import json
from typing import List, Dict, Any
from datetime import datetime
from langchain.docstore.document import Document
import hashlib
import numpy as np

class JsonSerializer:
    """Classe para serializar e armazenar chunks e embeddings em arquivos JSON."""
    
    def __init__(self, chunks_dir="chunks", embeddings_dir="embeddings"):
        """
        Inicializa o serializador JSON.
        
        Args:
            chunks_dir: Diretório onde os chunks serão armazenados
            embeddings_dir: Diretório onde os embeddings serão armazenados
        """
        self.chunks_dir = chunks_dir
        self.embeddings_dir = embeddings_dir
        
        # Criar diretórios se não existirem
        os.makedirs(chunks_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
    
    def _generate_document_id(self, file_path: str, content_hash: str = None) -> str:
        """
        Gera um ID único para um documento baseado em seu caminho e conteúdo.
        
        Args:
            file_path: Caminho do arquivo
            content_hash: Hash opcional do conteúdo para garantir unicidade
            
        Returns:
            ID único para o documento
        """
        base_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if content_hash is None:
            # Se não tiver hash do conteúdo, usa apenas o nome e timestamp
            document_id = f"{base_name.split('.')[0]}_{timestamp}"
        else:
            # Se tiver hash do conteúdo, usa para garantir unicidade
            document_id = f"{base_name.split('.')[0]}_{content_hash[:8]}"
            
        return document_id
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calcula o hash SHA-256 do conteúdo."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _document_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Converte um objeto Document para dicionário."""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    
    def _dict_to_document(self, doc_dict: Dict[str, Any]) -> Document:
        """Converte um dicionário para objeto Document."""
        return Document(
            page_content=doc_dict["page_content"],
            metadata=doc_dict["metadata"]
        )
    
    def save_chunks(self, documents: List[Document], file_path: str) -> str:
        """
        Salva os chunks de um documento em um arquivo JSON.
        
        Args:
            documents: Lista de documentos (chunks) a serem salvos
            file_path: Caminho do arquivo original
            
        Returns:
            ID do documento usado para salvar os chunks
        """
        if not documents:
            return None
            
        # Gerar um ID único para o documento baseado no conteúdo do primeiro chunk
        content_hash = self._calculate_content_hash(documents[0].page_content)
        document_id = self._generate_document_id(file_path, content_hash)
        
        # Caminho do arquivo JSON de saída
        json_file_path = os.path.join(self.chunks_dir, f"{document_id}.json")
        
        # Converter documentos para dicionários
        doc_dicts = [self._document_to_dict(doc) for doc in documents]
        
        # Adicionar metadados de processamento
        output_data = {
            "document_id": document_id,
            "original_file": file_path,
            "chunk_count": len(documents),
            "processed_at": datetime.now().isoformat(),
            "chunks": doc_dicts
        }
        
        # Salvar em JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return document_id
    
    def load_chunks(self, document_id: str) -> List[Document]:
        """
        Carrega chunks de um arquivo JSON.
        
        Args:
            document_id: ID do documento
            
        Returns:
            Lista de objetos Document
        """
        json_file_path = os.path.join(self.chunks_dir, f"{document_id}.json")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Arquivo de chunks não encontrado: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [self._dict_to_document(chunk) for chunk in data["chunks"]]
    
    def save_embeddings(self, document_id: str, embeddings: List[List[float]], texts: List[str], 
                       metadatas: List[Dict[str, Any]]) -> str:
        """
        Salva embeddings em um arquivo JSON.
        
        Args:
            document_id: ID do documento
            embeddings: Lista de embeddings (vetores)
            texts: Lista de textos correspondentes
            metadatas: Lista de metadados correspondentes
            
        Returns:
            Caminho do arquivo JSON de embeddings
        """
        # Caminho do arquivo JSON de saída
        json_file_path = os.path.join(self.embeddings_dir, f"{document_id}_embeddings.json")
        
        # Preparar dados para salvar
        output_data = {
            "document_id": document_id,
            "embedding_count": len(embeddings),
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "created_at": datetime.now().isoformat(),
            "items": [
                {
                    "text": text[:200] + ("..." if len(text) > 200 else ""),  # Texto truncado para economizar espaço
                    "metadata": metadata,
                    "embedding": embedding  # Vetor de embedding
                }
                for text, metadata, embedding in zip(texts, metadatas, embeddings)
            ]
        }
        
        # Salvar em JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # Use o default=float para converter arrays numpy para listas
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=float)
            
        return json_file_path
    
    def load_embeddings(self, document_id: str) -> Dict[str, Any]:
        """
        Carrega embeddings de um arquivo JSON.
        
        Args:
            document_id: ID do documento
            
        Returns:
            Dicionário com embeddings, textos e metadados
        """
        json_file_path = os.path.join(self.embeddings_dir, f"{document_id}_embeddings.json")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return {
            "embeddings": [item["embedding"] for item in data["items"]],
            "texts": [item["text"] for item in data["items"]],
            "metadatas": [item["metadata"] for item in data["items"]]
        }
    
    def list_processed_documents(self) -> List[str]:
        """
        Lista todos os IDs de documentos já processados (que têm chunks).
        
        Returns:
            Lista de IDs de documentos
        """
        if not os.path.exists(self.chunks_dir):
            return []
            
        return [
            os.path.splitext(file)[0]
            for file in os.listdir(self.chunks_dir)
            if file.endswith(".json")
        ]
    
    def list_embedded_documents(self) -> List[str]:
        """
        Lista todos os IDs de documentos que têm embeddings.
        
        Returns:
            Lista de IDs de documentos
        """
        if not os.path.exists(self.embeddings_dir):
            return []
            
        return [
            os.path.splitext(file)[0].replace("_embeddings", "")
            for file in os.listdir(self.embeddings_dir)
            if file.endswith("_embeddings.json")
        ]