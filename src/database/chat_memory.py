import sqlite3
import os
import sys
from pathlib import Path
# Adicionar o diretório raiz ao PATH do Python para permitir importações relativas
# quando o script é executado diretamente
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.config.settings import DB_PATH

def clear_agent_memory(thread_id=None):
    """
    Limpa a memória do agente armazenada no banco de dados SQLite.
    
    Args:
        thread_id (str, optional): ID específico do thread para limpar.
                                   Se None, limpa todos os threads.
    
    Returns:
        int: Número de registros removidos
    """
    try:
        # Usa a mesma conexão que o agente
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        # Verifica se as tabelas existem
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        records_removed = 0
        
        if 'checkpoints' in tables:
            if thread_id:
                # Remove apenas registros do thread específico
                cursor.execute("DELETE FROM checkpoints WHERE config LIKE ?", (f'%"thread_id": "{thread_id}"%',))
            else:
                # Remove todos os registros
                cursor.execute("DELETE FROM checkpoints")
            records_removed += cursor.rowcount
        
        if 'events' in tables:
            if thread_id:
                cursor.execute("DELETE FROM events WHERE config LIKE ?", (f'%"thread_id": "{thread_id}"%',))
            else:
                cursor.execute("DELETE FROM events")
            records_removed += cursor.rowcount
            
        conn.commit()
        conn.close()
        
        return records_removed
    except Exception as e:
        print(f"Erro ao limpar a memória do agente: {e}")
        return 0
    

if __name__ == "__main__":
    n = clear_agent_memory()
    print(f"Número de registros removidos: {n}")