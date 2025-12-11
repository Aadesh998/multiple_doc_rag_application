import sys
import sqlite3
from pypdf import PdfReader
import sqlite_vec
import ollama
import logging

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
    
    # Create table with embeddings
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk TEXT,
        embedding BLOB
    );
    """)
    
    # Create virtual table for vector search
    # Note: You need to know the embedding dimension (768 for nomic-embed-text)
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
    for attempt_num, (word_list, description) in enumerate(attempts):
        try:
            truncated_text = " ".join(word_list)
            if attempt_num > 0:
                logger.warning(f"Retry with {description}")
            response = ollama.embed(
                model='nomic-embed-text',
                input=truncated_text
            )
            if 'embeddings' in response and len(response['embeddings']) > 0:
                return response['embeddings'][0]
            elif 'embedding' in response:
                return response['embedding']
            else:
                logger.warning("Unexpected response format")
                continue
        except Exception as e:
            error_msg = str(e).lower()
            if 'batch' in error_msg or 'context' in error_msg or 'too large' in error_msg:
                if attempt_num < len(attempts) - 1:
                    continue
                else:
                    logger.error(f"Failed even with smallest chunk: {e}")
                    return None
            else:
                logger.error(f"Embedding error: {e}")
                return None
    return None

def save_to_db(conn, chunk, vector):
    try:
        if not isinstance(vector, list):
            logger.error(f"Invalid vector type: {type(vector)}")
            return False
        if len(vector) == 0:
            logger.error("Empty vector")
            return False
        
        serialized = sqlite_vec.serialize(vector)
        cursor = conn.execute(
            "INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)",
            (chunk, serialized)
        )
        row_id = cursor.lastrowid
        
        # Also insert into virtual table for vector search
        conn.execute(
            "INSERT INTO vec_embeddings (id, embedding) VALUES (?, ?)",
            (row_id, serialized)
        )
        
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        logger.info(f"Saved chunk. Total rows: {count}")
        return True
    except Exception as e:
        logger.exception(f"Database error: {e}")
        return False

def search_similar(query_text, k=5):
    """
    Search for top-k most similar chunks to the query
    
    Args:
        query_text: The search query
        k: Number of top results to return (default: 5)
    
    Returns:
        List of tuples: (chunk_text, distance, id)
    """
    try:
        # Get embedding for the query
        query_vector = embed_with_ollama(query_text)
        if not query_vector:
            logger.error("Failed to embed query")
            return []
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        # Serialize query vector
        query_serialized = sqlite_vec.serialize(query_vector)
        
        # Perform vector search
        # Using L2 distance (you can also use 'cosine' or 'ip' for inner product)
        results = conn.execute("""
            SELECT 
                e.id,
                e.chunk,
                distance
            FROM vec_embeddings v
            JOIN embeddings e ON e.id = v.id
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY distance
        """, (query_serialized, k)).fetchall()
        
        conn.close()
        
        # Format results as (chunk, distance, id)
        formatted_results = [(row[1], row[2], row[0]) for row in results]
        
        logger.info(f"Found {len(formatted_results)} similar chunks")
        return formatted_results
        
    except Exception as e:
        logger.exception(f"Search error: {e}")
        return []

def search_similar_with_threshold(query_text, k=5, max_distance=1.0):
    """
    Search for similar chunks with a distance threshold
    
    Args:
        query_text: The search query
        k: Maximum number of results
        max_distance: Maximum distance threshold (smaller = more similar)
    
    Returns:
        List of tuples: (chunk_text, distance, id)
    """
    results = search_similar(query_text, k)
    # Filter by distance threshold
    filtered_results = [(chunk, dist, id_) for chunk, dist, id_ in results if dist <= max_distance]
    return filtered_results

def test_ollama():
    try:
        response = ollama.embed(
            model='nomic-embed-text',
            input='test'
        )
        if 'embeddings' in response or 'embedding' in response:
            logger.info("Ollama is working correctly")
            return True
        else:
            logger.error("Unexpected response from Ollama")
            return False
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Index PDF: python script.py <pdf_path>")
        print("  Search:    python script.py --search '<query>' [k]")
        sys.exit(1)

    # Search mode
    if sys.argv[1] == "--search":
        if len(sys.argv) < 3:
            print("Please provide a search query")
            sys.exit(1)
        
        query = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        print(f"\nSearching for: '{query}'")
        print(f"Returning top {k} results\n")
        
        results = search_similar(query, k)
        
        if not results:
            print("No results found")
        else:
            for i, (chunk, distance, chunk_id) in enumerate(results, 1):
                print(f"--- Result {i} (ID: {chunk_id}, Distance: {distance:.4f}) ---")
                print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                print()
        
        sys.exit(0)

    # Index mode (original functionality)
    if not test_ollama():
        sys.exit(1)

    pdf_path = sys.argv[1]
    logger.info(f"Processing: {pdf_path}")

    text = extract_text(pdf_path)
    if not text:
        logger.error("Could not extract text from PDF")
        sys.exit(1)
    logger.info(f"Extracted {len(text)} characters")

    chunks = chunk_text(text, size=150)
    logger.info(f"Created {len(chunks)} chunks")

    conn = create_db()
    logger.info(f"Database ready: {DB_PATH}")

    success_count = 0
    failed_count = 0

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] Processing chunk...")
        vector = embed_with_ollama(chunk)
        if vector:
            if save_to_db(conn, chunk, vector):
                success_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1

    conn.close()
    logger.info(f"Processing complete. Success: {success_count}, Failed: {failed_count}")

    if success_count > 0:
        try:
            conn = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            logger.info(f"Total embeddings in database: {count}")
            sample = conn.execute("SELECT chunk FROM embeddings LIMIT 1").fetchone()
            if sample:
                logger.info(f"Sample chunk: {sample[0][:200]}...")
            conn.close()
        except Exception as e:
            logger.warning(f"Could not verify database: {e}")