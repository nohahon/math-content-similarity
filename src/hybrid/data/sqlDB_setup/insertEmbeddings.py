import sqlite3
import numpy as np

def insert_embedding(key, embedding):
    # Convert numpy array to bytes
    embedding_bytes = sqlite3.Binary(embedding.tobytes())

    # Connect to SQLite database
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()

    # Insert data
    cursor.execute('INSERT OR REPLACE INTO embeddings (id, embedding) VALUES (?, ?)', (key, embedding_bytes))

    # Commit and close
    conn.commit()
    conn.close()

# Example usage
embedding = np.random.rand(512)  # A random embedding for demonstration
insert_embedding('my_unique_key', embedding)