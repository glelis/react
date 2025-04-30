import argparse
import json
from vector_store import VectorStoreManager
from json_serializer import JsonSerializer
import os
import logging

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_tool")

def format_document(doc, index):
    """Formata um documento para exibição."""
    return {
        "índice": index,
        "conteúdo": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
        "metadados": doc.metadata
    }

def query_from_vector_db(db_dir, query, k, with_score=False):
    """Realiza consulta diretamente do banco de dados vetorial."""
    
    # Verificar se o diretório do banco de dados existe
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"O diretório do banco de dados '{db_dir}' não foi encontrado.")
    
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
    """Lista todos os arquivos de embeddings disponíveis."""
    if not os.path.exists(embeddings_dir):
        return []
    
    return [f for f in os.listdir(embeddings_dir) if f.endswith("_embeddings.json")]

def query_from_json_files(chunks_dir, embeddings_dir, query, k):
    """Realiza consulta diretamente dos arquivos JSON de embeddings."""
    # Inicializar o serializador JSON
    json_serializer = JsonSerializer(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    
    # Inicializar o gerenciador do banco de dados vetorial (apenas para usar a função de embedding)
    vector_store = VectorStoreManager()
    
    # Obter o embedding da consulta
    query_embedding = vector_store.get_embeddings([query])[0]
    
    # Lista de resultados com score
    all_results = []
    
    # Função para calcular similaridade de cosseno
    def cosine_similarity(v1, v2):
        import numpy as np
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Listar todos os arquivos JSON de embeddings
    embedding_files = list_embeddings_files(embeddings_dir)
    logger.info(f"Encontrados {len(embedding_files)} arquivos de embeddings para consulta")
    
    # Processar cada arquivo de embeddings
    for embedding_file in embedding_files:
        document_id = embedding_file.replace("_embeddings.json", "")
        try:
            # Carregar embeddings do arquivo JSON
            embedding_data = json_serializer.load_embeddings(document_id)
            
            # Calcular similaridade para cada embedding
            for i, (text, metadata, embedding) in enumerate(zip(
                embedding_data["texts"], 
                embedding_data["metadatas"], 
                embedding_data["embeddings"]
            )):
                # Calcular similaridade
                similarity = cosine_similarity(query_embedding, embedding)
                
                # Adicionar à lista de resultados
                all_results.append({
                    "text": text,
                    "metadata": metadata,
                    "similarity": similarity,
                    "document_id": document_id
                })
        except Exception as e:
            logger.error(f"Erro ao processar arquivo {embedding_file}: {str(e)}")
    
    # Ordenar por similaridade (maior para menor)
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Pegar apenas os k melhores resultados
    top_results = all_results[:k]
    
    # Formatar resultados
    formatted_results = []
    for i, result in enumerate(top_results):
        formatted_result = {
            "índice": i+1,
            "conteúdo": result["text"],
            "metadados": result["metadata"],
            "score": result["similarity"],
            "document_id": result["document_id"]
        }
        formatted_results.append(formatted_result)
    
    # Estatísticas
    stats = {
        "total_embeddings": len(all_results),
        "arquivos_processados": len(embedding_files)
    }
    
    return formatted_results, stats

def query_tool_langgraph(db_dir, query, k=5, with_score=False):
    """
    Ferramenta para integração com LangGraph que realiza consultas no banco de dados vetorial.

    Args:
        db_dir (str): Diretório do banco de dados vetorial.
        query (str): Consulta a ser realizada.
        k (int): Número de resultados a retornar.
        with_score (bool): Incluir pontuação de similaridade nos resultados.

    Returns:
        dict: Resultados formatados da consulta.
    """
    try:
        formatted_results, stats = query_from_vector_db(db_dir=db_dir, query=query, k=k, with_score=with_score)
        return {
            "success": True,
            "results": formatted_results,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Erro ao realizar consulta no vector DB: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Ferramenta para consultar o banco de dados vetorial ou arquivos JSON de embeddings')
    parser.add_argument('--db_dir', type=str, default='chroma_db', help='Diretório do banco de dados vetorial')
    parser.add_argument('--chunks_dir', type=str, default='chunks', help='Diretório dos chunks JSON')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Diretório dos embeddings JSON')
    parser.add_argument('--query', type=str, required=True, help='Consulta para pesquisar')
    parser.add_argument('--k', type=int, default=5, help='Número de resultados a retornar')
    parser.add_argument('--with_score', action='store_true', help='Incluir pontuação de similaridade nos resultados')
    parser.add_argument('--output', type=str, choices=['text', 'json'], default='text', help='Formato de saída')
    parser.add_argument('--source', type=str, choices=['db', 'json'], default='db', 
                        help='Fonte da consulta: banco de dados vetorial (db) ou arquivos JSON (json)')

    args = parser.parse_args()
    
    # Realizar a consulta na fonte apropriada
    if args.source == 'db':
        formatted_results, stats = query_from_vector_db(
            db_dir=args.db_dir, 
            query=args.query, 
            k=args.k, 
            with_score=args.with_score
        )
        source_name = "banco de dados vetorial"
    else:
        formatted_results, stats = query_from_json_files(
            chunks_dir=args.chunks_dir, 
            embeddings_dir=args.embeddings_dir, 
            query=args.query, 
            k=args.k
        )
        source_name = "arquivos JSON de embeddings"
    
    # Exibir resultados
    if args.output == 'json':
        print(json.dumps(formatted_results, ensure_ascii=False, indent=2, default=str))
    else:
        # Saída formatada em texto
        print(f"\nResultados para consulta: '{args.query}' (via {source_name})\n")
        
        for result in formatted_results:
            print(f"Resultado {result['índice']}:")
            print(f"Conteúdo: {result['conteúdo']}")
            
            if 'score' in result:
                print(f"Score de similaridade: {result['score']:.4f}")
            
            if 'document_id' in result and args.source == 'json':
                print(f"ID do documento: {result['document_id']}")
                
            print("Metadados:")
            for key, value in result['metadados'].items():
                print(f"  {key}: {value}")
            print("-" * 50)
    
    # Exibir estatísticas
    if args.output == 'text':
        print(f"\nEstatísticas da consulta:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()