import sqlite3

# Connect to SQLite database (or create if it doesn't exist)
conn = sqlite3.connect('sqlLite_DB/rfrncsEmbeddings.db')

# Create a cursor object using the cursor() method
cursor = conn.cursor()

# Create table as per requirement
sql ='''
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    embedding BLOB
)
'''
cursor.execute(sql)

# Commit your changes in the database
conn.commit()

# Close the connection
conn.close()