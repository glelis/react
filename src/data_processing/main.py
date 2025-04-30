import os
import argparse
from typing import List
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from json_serializer import JsonSerializer
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("document_pipeline")

def process_directory(directory_path: str, extensions: List[str] = None, 
                     chunks_dir: str = "chunks", embeddings_dir: str = "embeddings", 
                     vector_store_dir: str = "chroma_db", skip_existing: bool = True):
    """
    Processa todos os documentos suportados em um diretório com etapas intermediárias.
    
    Args:
        directory_path: Caminho para o diretório com os documentos
        extensions: Lista de extensões para processar (padrão: txt, html, htm, pdf)
        chunks_dir: Diretório para salvar chunks em JSON
        embeddings_dir: Diretório para salvar embeddings em JSON
        vector_store_dir: Diretório para o banco de dados vetorial
        skip_existing: Pular arquivos já processados
    """
    if not extensions:
        extensions = ['txt', 'html', 'htm', 'pdf']
    
    document_processor = DocumentProcessor()
    json_serializer = JsonSerializer(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    vector_store = VectorStoreManager(persist_directory=vector_store_dir)
    
    processed_files = 0
    processed_document_ids = []
    
    logger.info(f"Iniciando processamento de documentos em: {directory_path}")
    
    # Etapa 1: Processar documentos e salvar chunks em JSON
    for root, _, files in os.walk(directory_path):
        for file in files:
            extension = file.split('.')[-1].lower()
            if extension in extensions:
                file_path = os.path.join(root, file)
                try:
                    logger.info(f"ETAPA 1 - Processando arquivo: {file_path}")
                    
                    # Processar o documento
                    documents = document_processor.process_file(file_path)
                    
                    # Salvar chunks em JSON
                    document_id = json_serializer.save_chunks(documents, file_path)
                    if document_id:
                        logger.info(f"Chunks salvos com ID: {document_id}")
                        processed_document_ids.append(document_id)
                        processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Erro ao processar {file_path}: {str(e)}")
    
    logger.info(f"ETAPA 1 concluída. {processed_files} arquivos foram processados.")
    
    # Etapa 2: Gerar embeddings e salvá-los em JSON
    for document_id in processed_document_ids:
        try:
            logger.info(f"ETAPA 2 - Gerando embeddings para documento: {document_id}")
            
            # Carregar chunks do arquivo JSON
            documents = json_serializer.load_chunks(document_id)
            
            # Preparar dados para embeddings
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Gerar embeddings
            embeddings = vector_store.get_embeddings(texts)
            
            # Salvar embeddings em JSON
            embedding_path = json_serializer.save_embeddings(
                document_id=document_id, 
                embeddings=embeddings, 
                texts=texts, 
                metadatas=metadatas
            )
            
            logger.info(f"Embeddings salvos em: {embedding_path}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings para {document_id}: {str(e)}")
    
    logger.info("ETAPA 2 concluída. Embeddings gerados e salvos.")
    
    # Etapa 3: Adicionar ao banco de dados vetorial
    for document_id in processed_document_ids:
        try:
            logger.info(f"ETAPA 3 - Adicionando ao banco vetorial: {document_id}")
            
            # Carregar embeddings do arquivo JSON
            embedding_data = json_serializer.load_embeddings(document_id)
            
            # Adicionar ao banco de dados vetorial usando embeddings pré-calculados
            vector_store.add_documents_with_embeddings(
                texts=embedding_data["texts"],
                embeddings=embedding_data["embeddings"],
                metadatas=embedding_data["metadatas"]
            )
            
            logger.info(f"Documento {document_id} adicionado ao banco vetorial.")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar {document_id} ao banco vetorial: {str(e)}")
    
    logger.info("ETAPA 3 concluída. Documentos adicionados ao banco vetorial.")
    
    # Mostrar estatísticas do banco de dados
    stats = vector_store.get_collection_stats()
    logger.info(f"Estatísticas do banco de dados: {stats}")
    
    return {
        "processed_files": processed_files,
        "document_ids": processed_document_ids
    }


def main():
    parser = argparse.ArgumentParser(description='Pipeline de processamento de documentos para criar banco de dados vetorial')
    parser.add_argument('--dir', type=str, default='data_raw', help='Diretório contendo os documentos para processar')
    parser.add_argument('--chunks_dir', type=str, default='chunks', help='Diretório para salvar chunks em JSON')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Diretório para salvar embeddings em JSON')
    parser.add_argument('--db_dir', type=str, default='chroma_db', help='Diretório para armazenar o banco de dados vetorial')
    parser.add_argument('--extensions', type=str, default='txt,htm,pdf', help='Extensões de arquivo para processar (separadas por vírgula)')
    parser.add_argument('--skip_existing', action='store_false', help='Pular arquivos já processados')
    parser.add_argument('--query', type=str, help='Consulta para testar o banco de dados (opcional)')
    
    args = parser.parse_args()
    
    # Verificar se o diretório de documentos existe
    if not os.path.exists(args.dir):
        logger.error(f"O diretório {args.dir} não existe")
        return
    
    # Processar os documentos em etapas
    extensions = args.extensions.split(',')
    result = process_directory(
        directory_path=args.dir, 
        extensions=extensions,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        vector_store_dir=args.db_dir,
        skip_existing=args.skip_existing
    )
    
    logger.info(f"Processamento concluído. {len(result['document_ids'])} documentos processados.")
    
    # Testar o banco de dados com uma consulta (se fornecida)
    if args.query:
        vector_store = VectorStoreManager(persist_directory=args.db_dir)
        
        logger.info(f"Executando consulta de teste: '{args.query}'")
        results = vector_store.search(args.query, k=3)
        
        logger.info(f"Resultados da consulta (top 3):")
        for i, doc in enumerate(results):
            logger.info(f"Resultado {i+1}:")
            logger.info(f"Conteúdo: {doc.page_content[:150]}...")
            logger.info(f"Metadados: {doc.metadata}")
            logger.info("---")


if __name__ == "__main__":
    main()