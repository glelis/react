import sqlite3
import os
import sys
from pathlib import Path
# Add the root directory to the Python PATH to allow relative imports when the script is executed directly
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.config.settings import DB_PATH

def clear_agent_memory(thread_id=None):
    """
    Clears the agent's memory stored in the SQLite database.

    Args:
        thread_id (str, optional): Specific thread ID to clear.
                                   If None, clears all threads.

    Returns:
        int: Number of records removed
    """
    try:
        # Uses the same connection as the agent
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        # Check if the tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        records_removed = 0
        
        if 'checkpoints' in tables:
            if thread_id:
                # Remove only records from the specific thread
                cursor.execute("DELETE FROM checkpoints WHERE config LIKE ?", (f'%"thread_id": "{thread_id}"%',))
            else:
                # Remove all records
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
        print(f"Error clearing the agent's memory: {e}")
        return 0
    

if __name__ == "__main__":
    n = clear_agent_memory()
    print(f"Number of records removed: {n}")