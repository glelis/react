import os
import json
from typing import List, Dict, Any
from datetime import datetime
from langchain.docstore.document import Document
import hashlib
import numpy as np

class JsonSerializer:
    """Class to serialize and store chunks and embeddings in JSON files."""
    
    def __init__(self, chunks_dir="chunks", embeddings_dir="embeddings"):
        """
        Initializes the JSON serializer.
        
        Args:
            chunks_dir: Directory where chunks will be stored
            embeddings_dir: Directory where embeddings will be stored
        """
        self.chunks_dir = chunks_dir
        self.embeddings_dir = embeddings_dir
        
        # Create directories if they do not exist
        os.makedirs(chunks_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
    
    def _generate_document_id(self, file_path: str, content_hash: str = None) -> str:
        """
        Generates a unique ID for a document based on its path and content.
        
        Args:
            file_path: File path
            content_hash: Optional content hash to ensure uniqueness
            
        Returns:
            Unique ID for the document
        """
        base_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if content_hash is None:
            # If no content hash is provided, use only the name and timestamp
            document_id = f"{base_name.split('.')[0]}_{timestamp}"
        else:
            # If content hash is provided, use it to ensure uniqueness
            document_id = f"{base_name.split('.')[0]}_{content_hash[:8]}"
            
        return document_id
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculates the SHA-256 hash of the content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _document_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Converts a Document object to a dictionary."""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    
    def _dict_to_document(self, doc_dict: Dict[str, Any]) -> Document:
        """Converts a dictionary to a Document object."""
        return Document(
            page_content=doc_dict["page_content"],
            metadata=doc_dict["metadata"]
        )
    
    def save_chunks(self, documents: List[Document], file_path: str) -> str:
        """
        Saves the chunks of a document to a JSON file.
        
        Args:
            documents: List of documents (chunks) to be saved
            file_path: Path of the original file
            
        Returns:
            Document ID used to save the chunks
        """
        if not documents:
            return None
            
        # Generate a unique ID for the document based on the content of the first chunk
        content_hash = self._calculate_content_hash(documents[0].page_content)
        document_id = self._generate_document_id(file_path, content_hash)
        
        # Path of the output JSON file
        json_file_path = os.path.join(self.chunks_dir, f"{document_id}.json")
        
        # Convert documents to dictionaries
        doc_dicts = [self._document_to_dict(doc) for doc in documents]
        
        # Add processing metadata
        output_data = {
            "document_id": document_id,
            "original_file": file_path,
            "chunk_count": len(documents),
            "processed_at": datetime.now().isoformat(),
            "chunks": doc_dicts
        }
        
        # Save to JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return document_id
    
    def load_chunks(self, document_id: str) -> List[Document]:
        """
        Loads chunks from a JSON file.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of Document objects
        """
        json_file_path = os.path.join(self.chunks_dir, f"{document_id}.json")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Chunks file not found: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [self._dict_to_document(chunk) for chunk in data["chunks"]]
    
    def save_embeddings(self, document_id: str, embeddings: List[List[float]], texts: List[str], 
                       metadatas: List[Dict[str, Any]]) -> str:
        """
        Saves embeddings to a JSON file.
        
        Args:
            document_id: Document ID
            embeddings: List of embeddings (vectors)
            texts: List of corresponding texts
            metadatas: List of corresponding metadata
            
        Returns:
            Path of the embeddings JSON file
        """
        # Path of the output JSON file
        json_file_path = os.path.join(self.embeddings_dir, f"{document_id}_embeddings.json")
        
        # Prepare data for saving
        output_data = {
            "document_id": document_id,
            "embedding_count": len(embeddings),
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "created_at": datetime.now().isoformat(),
            "items": [
                {
                    "text": text[:200] + ("..." if len(text) > 200 else ""),  # Truncated text to save space
                    "metadata": metadata,
                    "embedding": embedding  # Embedding vector
                }
                for text, metadata, embedding in zip(texts, metadatas, embeddings)
            ]
        }
        
        # Save to JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # Use default=float to convert numpy arrays to lists
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=float)
            
        return json_file_path
    
    def load_embeddings(self, document_id: str) -> Dict[str, Any]:
        """
        Loads embeddings from a JSON file.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with embeddings, texts, and metadata
        """
        json_file_path = os.path.join(self.embeddings_dir, f"{document_id}_embeddings.json")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Embeddings file not found: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return {
            "embeddings": [item["embedding"] for item in data["items"]],
            "texts": [item["text"] for item in data["items"]],
            "metadatas": [item["metadata"] for item in data["items"]]
        }
    
    def list_processed_documents(self) -> List[str]:
        """
        Lists all IDs of documents that have been processed (have chunks).
        
        Returns:
            List of document IDs
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
        Lists all IDs of documents that have embeddings.
        
        Returns:
            List of document IDs
        """
        if not os.path.exists(self.embeddings_dir):
            return []
            
        return [
            os.path.splitext(file)[0].replace("_embeddings", "")
            for file in os.listdir(self.embeddings_dir)
            if file.endswith("_embeddings.json")
        ]