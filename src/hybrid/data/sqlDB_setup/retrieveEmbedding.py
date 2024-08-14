import sqlite3
import numpy as np

def get_embedding(key):
    # Connect to SQLite database
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()

    # Retrieve data
    cursor.execute('SELECT embedding FROM embeddings WHERE id = ?', (key,))
    data = cursor.fetchone()

    # Convert bytes back to numpy array
    if data:
        embedding = np.frombuffer(data[0], dtype=np.float64)  # Adjust dtype according to how you store it
        return embedding
    return None

# Example usage
retrieved_embedding = get_embedding('my_unique_key')
print(retrieved_embedding)