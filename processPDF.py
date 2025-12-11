import sys
import sqlite3
from pypdf import PdfReader
import sqlite_vec
from sqlite_vec import serialize_float32
import ollama
import logging
import json 

DB_PATH = "rag.db"
LOG_FILE = "app.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text, size=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def create_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk TEXT,
        embedding BLOB
    );
    """)
    
    try:
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[768]
        );
        """)
    except Exception as e:
        logger.info(f"Virtual table might already exist: {e}")
    
    conn.commit()
    return conn

def embed_with_ollama(text, max_retries=2):
    words = text.split()
    attempts = [
        (words[:150], "150 words"),
        (words[:100], "100 words"),
        (words[:50], "50 words"),
    ]
    for attempt_num, (word_list, _) in enumerate(attempts):
        try:
            truncated_text = " ".join(word_list)
            response = ollama.embed(
                model='nomic-embed-text',
                input=truncated_text
            )
            if 'embeddings' in response and len(response['embeddings']) > 0:
                return response['embeddings'][0]
            elif 'embedding' in response:
                return response['embedding']
        except Exception as e:
            error_msg = str(e).lower()
            if 'batch' in error_msg or 'context' in error_msg or 'too large' in error_msg:
                if attempt_num < len(attempts) - 1:
                    continue
                else:
                    return None
            else:
                return None
    return None

def save_to_db(conn, chunk, vector):
    try:
        if not isinstance(vector, list):
            return False
        if len(vector) == 0:
            return False
        
        serialized = serialize_float32(vector)

        cursor = conn.execute(
            "INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)",
            (chunk, serialized)
        )
        row_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO vec_embeddings (id, embedding) VALUES (?, ?)",
            (row_id, serialized)
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.exception(f"Database error: {e}")
        return False

def search_for_go(query_text, k=3):
    try:
        query_vector = embed_with_ollama(query_text)
        if not query_vector:
            return []
        
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn) 
        conn.enable_load_extension(False)
        
        query_serialized = serialize_float32(query_vector)
        
        results = conn.execute("""
            SELECT 
                e.chunk
            FROM vec_embeddings v
            JOIN embeddings e ON e.id = v.id
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY distance
        """, (query_serialized, k)).fetchall()
        
        conn.close()
        
        return [row[0] for row in results] 
        
    except Exception as e:
        logger.exception(f"Search error in Python utility: {e}") 
        return []

def test_ollama():
    try:
        response = ollama.embed(
            model='nomic-embed-text',
            input='test'
        )
        if 'embeddings' in response or 'embedding' in response:
            return True
        else:
            return False
    except Exception:
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Index PDF: python script.py <pdf_path>")
        print("  Search:    python script.py --search '<query>' [k]")
        print("  Search GO: python script.py --search-go '<query>' [k]") 
        sys.exit(1)

    if sys.argv[1] == "--search-go": 
        if len(sys.argv) < 3:
            print(json.dumps([])) 
            sys.exit(0)
        
        query = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        
        chunks = search_for_go(query, k)
        
        print(json.dumps(chunks))
        sys.exit(0)

    if not test_ollama():
        sys.exit(1)

    pdf_path = sys.argv[1]

    text = extract_text(pdf_path)
    if not text:
        sys.exit(1)

    chunks = chunk_text(text, size=150)

    conn = create_db()

    success_count = 0
    failed_count = 0

    for i, chunk in enumerate(chunks, 1):
        vector = embed_with_ollama(chunk)
        if vector:
            if save_to_db(conn, chunk, vector):
                success_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1

    conn.close()

    if success_count > 0:
        try:
            conn = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            sample = conn.execute("SELECT chunk FROM embeddings LIMIT 1").fetchone()
            conn.close()
        except Exception:
            pass