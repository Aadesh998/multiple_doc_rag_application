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
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk TEXT,
        embedding BLOB
    );
    """)
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
        conn.execute(
            "INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)",
            (chunk, serialized)
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        logger.info(f"Saved chunk. Total rows: {count}")
        return True
    except Exception as e:
        logger.exception(f"Database error: {e}")
        return False

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
        logger.error("Usage: python script.py <pdf_path>")
        sys.exit(1)

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
