from typing import Dict, Any, List
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Processes TXT files."""
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Extract metadata
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'txt',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path)
        }
        
        # Split text into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Update metadata for each chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)
            
        return chunks
    
    def process_htm(self, file_path: str) -> List[Dict[str, Any]]:
        """Processes HTM files."""
        loader = BSHTMLLoader(file_path)
        documents = loader.load()

        # Clean HTML content
        cleaned_documents = []
        for doc in documents:
            soup = BeautifulSoup(doc.page_content, 'html.parser')
            cleaned_text = soup.get_text(separator=" ").strip()  # Extract clean text
            doc.page_content = cleaned_text  # Update document content
            cleaned_documents.append(doc)

        # Extract metadata
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'htm',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path)
        }

        # Split text into chunks
        chunks = self.text_splitter.split_documents(cleaned_documents)

        # Update metadata for each chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)

        return chunks
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Processes PDF files without using OCR."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Extract metadata
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'pdf',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path),
            'page_count': len(documents)
        }
        
        # Add page-specific metadata
        for doc in documents:
            doc.metadata.update({
                'page': doc.metadata.get('page', 0),
                'total_pages': len(documents)
            })
        
        # Split text into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Update metadata for each chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)
            
        return chunks
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Processes a file based on its extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.split('.')[-1].lower()
        
        if extension == 'txt':
            return self.process_txt(file_path)
        elif extension == 'htm':
            return self.process_htm(file_path)
        elif extension == 'pdf':
            return self.process_pdf(file_path)
        else:
            raise ValueError(f"Unsupported format: {extension}")