import os
import sys
import argparse
from pathlib import Path
from typing import List
import logging

# Add the root directory to the Python PATH to allow relative imports
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from document_processor import DocumentProcessor
from src.database import VectorStoreManager
from json_serializer import JsonSerializer

# Logging configuration
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
    Processes all supported documents in a directory with intermediate steps.
    
    Args:
        directory_path: Path to the directory containing the documents
        extensions: List of file extensions to process (default: txt, html, htm, pdf)
        chunks_dir: Directory to save chunks in JSON format
        embeddings_dir: Directory to save embeddings in JSON format
        vector_store_dir: Directory for the vector database
        skip_existing: Skip already processed files
    """
    if not extensions:
        extensions = ['txt', 'html', 'htm', 'pdf']
    
    document_processor = DocumentProcessor()
    json_serializer = JsonSerializer(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    vector_store = VectorStoreManager(persist_directory=vector_store_dir)
    
    processed_files = 0
    processed_document_ids = []
    
    logger.info(f"Starting document processing in: {directory_path}")
    
    # Step 1: Process documents and save chunks in JSON
    for root, _, files in os.walk(directory_path):
        for file in files:
            extension = file.split('.')[-1].lower()
            if extension in extensions:
                file_path = os.path.join(root, file)
                try:
                    logger.info(f"STEP 1 - Processing file: {file_path}")
                    
                    # Process the document
                    documents = document_processor.process_file(file_path)
                    
                    # Save chunks in JSON
                    document_id = json_serializer.save_chunks(documents, file_path)
                    if document_id:
                        logger.info(f"Chunks saved with ID: {document_id}")
                        processed_document_ids.append(document_id)
                        processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"STEP 1 completed. {processed_files} files were processed.")
    
    # Step 2: Generate embeddings and save them in JSON
    for document_id in processed_document_ids:
        try:
            logger.info(f"STEP 2 - Generating embeddings for document: {document_id}")
            
            # Load chunks from JSON file
            documents = json_serializer.load_chunks(document_id)
            
            # Prepare data for embeddings
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings = vector_store.get_embeddings(texts)
            
            # Save embeddings in JSON
            embedding_path = json_serializer.save_embeddings(
                document_id=document_id, 
                embeddings=embeddings, 
                texts=texts, 
                metadatas=metadatas
            )
            
            logger.info(f"Embeddings saved in: {embedding_path}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {document_id}: {str(e)}")
    
    logger.info("STEP 2 completed. Embeddings generated and saved.")
    
    # Step 3: Add to the vector database
    for document_id in processed_document_ids:
        try:
            logger.info(f"STEP 3 - Adding to vector database: {document_id}")
            
            # Load embeddings from JSON file
            embedding_data = json_serializer.load_embeddings(document_id)
            
            # Add to the vector database using precomputed embeddings
            vector_store.add_documents_with_embeddings(
                texts=embedding_data["texts"],
                embeddings=embedding_data["embeddings"],
                metadatas=embedding_data["metadatas"]
            )
            
            logger.info(f"Document {document_id} added to vector database.")
            
        except Exception as e:
            logger.error(f"Error adding {document_id} to vector database: {str(e)}")
    
    logger.info("STEP 3 completed. Documents added to vector database.")
    
    # Display database statistics
    stats = vector_store.get_collection_stats()
    logger.info(f"Database statistics: {stats}")
    
    return {
        "processed_files": processed_files,
        "document_ids": processed_document_ids
    }


def main():
    parser = argparse.ArgumentParser(description='Document processing pipeline to create a vector database')
    parser.add_argument('--dir', type=str, default='data_raw', help='Directory containing the documents to process')
    parser.add_argument('--chunks_dir', type=str, default='chunks', help='Directory to save chunks in JSON format')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Directory to save embeddings in JSON format')
    parser.add_argument('--db_dir', type=str, default='chroma_db', help='Directory to store the vector database')
    parser.add_argument('--extensions', type=str, default='txt,htm,pdf', help='File extensions to process (comma-separated)')
    parser.add_argument('--skip_existing', action='store_false', help='Skip already processed files')
    parser.add_argument('--query', type=str, help='Query to test the database (optional)')
    
    args = parser.parse_args()
    
    # Check if the document directory exists
    if not os.path.exists(args.dir):
        logger.error(f"The directory {args.dir} does not exist")
        return
    
    # Process the documents in steps
    extensions = args.extensions.split(',')
    result = process_directory(
        directory_path=args.dir, 
        extensions=extensions,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        vector_store_dir=args.db_dir,
        skip_existing=args.skip_existing
    )
    
    logger.info(f"Processing completed. {len(result['document_ids'])} documents processed.")
    
    # Test the database with a query (if provided)
    if args.query:
        vector_store = VectorStoreManager(persist_directory=args.db_dir)
        
        logger.info(f"Executing test query: '{args.query}'")
        results = vector_store.search(args.query, k=3)
        
        logger.info(f"Query results (top 3):")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}:")
            logger.info(f"Content: {doc.page_content[:150]}...")
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("---")


if __name__ == "__main__":
    main()