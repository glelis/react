import argparse
import json
from vector_store import VectorStoreManager
from json_serializer import JsonSerializer
import os
import logging

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_tool")

def format_document(doc, index):
    """Formats a document for display."""
    return {
        "index": index,
        "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
        "metadata": doc.metadata
    }

def query_from_vector_db(db_dir, query, k, with_score=False):
    """Performs a query directly from the vector database."""
    
    # Check if the database directory exists
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"The database directory '{db_dir}' was not found.")
    
    vector_store = VectorStoreManager(persist_directory=db_dir)
    
    if with_score:
        results = vector_store.search_with_score(query, k=k)
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = format_document(doc, i+1)
            result["score"] = score
            formatted_results.append(result)
    else:
        results = vector_store.search(query, k=k)
        formatted_results = [format_document(doc, i+1) for i, doc in enumerate(results)]
        
    return formatted_results, vector_store.get_collection_stats()

def list_embeddings_files(embeddings_dir):
    """Lists all available embedding files."""
    if not os.path.exists(embeddings_dir):
        return []
    
    return [f for f in os.listdir(embeddings_dir) if f.endswith("_embeddings.json")]

def query_from_json_files(chunks_dir, embeddings_dir, query, k):
    """Performs a query directly from JSON embedding files."""
    # Initialize the JSON serializer
    json_serializer = JsonSerializer(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    
    # Initialize the vector database manager (only to use the embedding function)
    vector_store = VectorStoreManager()
    
    # Get the query embedding
    query_embedding = vector_store.get_embeddings([query])[0]
    
    # List of results with scores
    all_results = []
    
    # Function to calculate cosine similarity
    def cosine_similarity(v1, v2):
        import numpy as np
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # List all JSON embedding files
    embedding_files = list_embeddings_files(embeddings_dir)
    logger.info(f"Found {len(embedding_files)} embedding files for query")
    
    # Process each embedding file
    for embedding_file in embedding_files:
        document_id = embedding_file.replace("_embeddings.json", "")
        try:
            # Load embeddings from the JSON file
            embedding_data = json_serializer.load_embeddings(document_id)
            
            # Calculate similarity for each embedding
            for i, (text, metadata, embedding) in enumerate(zip(
                embedding_data["texts"], 
                embedding_data["metadatas"], 
                embedding_data["embeddings"]
            )):
                # Calculate similarity
                similarity = cosine_similarity(query_embedding, embedding)
                
                # Add to the results list
                all_results.append({
                    "text": text,
                    "metadata": metadata,
                    "similarity": similarity,
                    "document_id": document_id
                })
        except Exception as e:
            logger.error(f"Error processing file {embedding_file}: {str(e)}")
    
    # Sort by similarity (highest to lowest)
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Take only the top k results
    top_results = all_results[:k]
    
    # Format results
    formatted_results = []
    for i, result in enumerate(top_results):
        formatted_result = {
            "index": i+1,
            "content": result["text"],
            "metadata": result["metadata"],
            "score": result["similarity"],
            "document_id": result["document_id"]
        }
        formatted_results.append(formatted_result)
    
    # Statistics
    stats = {
        "total_embeddings": len(all_results),
        "files_processed": len(embedding_files)
    }
    
    return formatted_results, stats

def query_tool_langgraph(db_dir, query, k=5, with_score=False):
    """
    Tool for integration with LangGraph that performs queries on the vector database.

    Args:
        db_dir (str): Directory of the vector database.
        query (str): Query to be performed.
        k (int): Number of results to return.
        with_score (bool): Include similarity score in the results.

    Returns:
        dict: Formatted query results.
    """
    try:
        formatted_results, stats = query_from_vector_db(db_dir=db_dir, query=query, k=k, with_score=with_score)
        return {
            "success": True,
            "results": formatted_results,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error performing query on vector DB: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Tool for querying the vector database or JSON embedding files')
    parser.add_argument('--db_dir', type=str, default='chroma_db', help='Directory of the vector database')
    parser.add_argument('--chunks_dir', type=str, default='chunks', help='Directory of JSON chunks')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Directory of JSON embeddings')
    parser.add_argument('--query', type=str, required=True, help='Query to search')
    parser.add_argument('--k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--with_score', action='store_true', help='Include similarity score in the results')
    parser.add_argument('--output', type=str, choices=['text', 'json'], default='text', help='Output format')
    parser.add_argument('--source', type=str, choices=['db', 'json'], default='db', 
                        help='Query source: vector database (db) or JSON files (json)')

    args = parser.parse_args()
    
    # Perform the query on the appropriate source
    if args.source == 'db':
        formatted_results, stats = query_from_vector_db(
            db_dir=args.db_dir, 
            query=args.query, 
            k=args.k, 
            with_score=args.with_score
        )
        source_name = "vector database"
    else:
        formatted_results, stats = query_from_json_files(
            chunks_dir=args.chunks_dir, 
            embeddings_dir=args.embeddings_dir, 
            query=args.query, 
            k=args.k
        )
        source_name = "JSON embedding files"
    
    # Display results
    if args.output == 'json':
        print(json.dumps(formatted_results, ensure_ascii=False, indent=2, default=str))
    else:
        # Text formatted output
        print(f"\nResults for query: '{args.query}' (via {source_name})\n")
        
        for result in formatted_results:
            print(f"Result {result['index']}:")
            print(f"Content: {result['content']}")
            
            if 'score' in result:
                print(f"Similarity score: {result['score']:.4f}")
            
            if 'document_id' in result and args.source == 'json':
                print(f"Document ID: {result['document_id']}")
                
            print("Metadata:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
            print("-" * 50)
    
    # Display statistics
    if args.output == 'text':
        print(f"\nQuery statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()