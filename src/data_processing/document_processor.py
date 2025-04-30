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
        """Processa arquivos TXT."""
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Extrair metadados
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'txt',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path)
        }
        
        # Dividir texto em chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Atualizar metadados para cada chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)
            
        return chunks
    
    def process_htm(self, file_path: str) -> List[Dict[str, Any]]:
        """Processa arquivos HTM."""
        loader = BSHTMLLoader(file_path)
        documents = loader.load()

        # Limpeza do conteúdo HTML
        cleaned_documents = []
        for doc in documents:
            soup = BeautifulSoup(doc.page_content, 'html.parser')
            cleaned_text = soup.get_text(separator=" ").strip()  # Extrai texto limpo
            doc.page_content = cleaned_text  # Atualiza o conteúdo do documento
            cleaned_documents.append(doc)

        # Extrair metadados
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'htm',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path)
        }

        # Dividir texto em chunks
        chunks = self.text_splitter.split_documents(cleaned_documents)

        # Atualizar metadados para cada chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)

        return chunks
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Processa arquivos PDF sem usar OCR."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Extrair metadados
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': 'pdf',
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path),
            'page_count': len(documents)
        }
        
        # Adicionar metadados específicos de página
        for doc in documents:
            doc.metadata.update({
                'page': doc.metadata.get('page', 0),
                'total_pages': len(documents)
            })
        
        # Dividir texto em chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Atualizar metadados para cada chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)
            
        return chunks
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Processa um arquivo com base em sua extensão."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        extension = file_path.split('.')[-1].lower()
        
        if extension == 'txt':
            return self.process_txt(file_path)
        elif extension == 'htm':
            return self.process_htm(file_path)
        elif extension == 'pdf':
            return self.process_pdf(file_path)
        else:
            raise ValueError(f"Formato não suportado: {extension}")