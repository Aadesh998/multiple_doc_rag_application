import sys
import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
import ollama
import logging
import json
import re
import os
import PyPDF2
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    '''Extracts text from a PDF file.'''
    print(f"Reading PDF: {pdf_path}")
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        print("PDF reading complete.")
        output_path = os.path.abspath("extracted.txt")
        print(f"Saving raw text to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def preprocess_text(text):
    """Cleans the extracted PDF text."""
    print("Preprocessing text...")
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    print("Text preprocessing complete.")
    output_path = os.path.abspath("preprocess.txt")
    print(f"Saving raw text to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text.strip()

def get_text_chunks(text):
    '''Splits text into manageable chunks.    '''
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def create_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk TEXT, embedding BLOB)")
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[768]
            );
        """)
    except Exception as e:
        logger.info(f"{e}")
    conn.commit()
    return conn

# def embed_with_ollama(text, max_retries=2):
#     words = text.split()
#     attempts = [
#         (words[:150], "150"),
#         (words[:100], "100"),
#         (words[:50], "50"),
#     ]
#     for attempt_num, (word_list, _) in enumerate(attempts):
#         try:
#             truncated = " ".join(word_list)
#             response = ollama.embed(model="nomic-embed-text", input=truncated)
#             if "embeddings" in response:
#                 return response["embeddings"][0]
#             if "embedding" in response:
#                 return response["embedding"]
#         except Exception as e:
#             msg = str(e).lower()
#             if "batch" in msg or "context" in msg or "too large" in msg:
#                 if attempt_num < len(attempts) - 1:
#                     continue
#                 return None
#             return None
#     return None

def get_embeddings(prompt, model="nomic-embed-text"):
    '''Generates embeddings for a given prompt using the specified model.'''
    try:
        response = ollama.embeddings(model=model, prompt=prompt)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def save_to_db(conn, chunk, vector):
    try:
        if not isinstance(vector, list):
            return False
        if len(vector) == 0:
            return False
        serialized = serialize_float32(vector)
        cursor = conn.execute("INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)", (chunk, serialized))
        row_id = cursor.lastrowid
        conn.execute("INSERT INTO vec_embeddings (id, embedding) VALUES (?, ?)", (row_id, serialized))
        conn.commit()
        return True
    except Exception as e:
        logger.exception(f"{e}")
        return False

def search_for_go(query_text, k=3):
    try:
        query_vector = get_embeddings(query_text)
        if not query_vector:
            return []
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        q = serialize_float32(query_vector)
        rows = conn.execute("""
            SELECT e.chunk
            FROM vec_embeddings v
            JOIN embeddings e ON e.id = v.id
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY v.distance
        """, (q, k)).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        logger.exception(f"{e}")
        return []

def test_ollama():
    try:
        r = ollama.embed(model="nomic-embed-text", input="test")
        return "embeddings" in r or "embedding" in r
    except Exception:
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    if sys.argv[1] == "--search-go":
        if len(sys.argv) < 3:
            print(json.dumps([]))
            sys.exit(0)
        q = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        res = search_for_go(q, k)
        print(json.dumps(res))
        sys.exit(0)

    if not test_ollama():
        sys.exit(1)

    pdf_path = sys.argv[1]
    text = extract_text(pdf_path)
    if not text:
        sys.exit(1)

    text = preprocess_text(text)
    chunks = get_text_chunks(text)

    conn = create_db()
    success = 0
    fail = 0

    for chunk in chunks:
        vec = get_embeddings(chunk)
        if vec:
            if save_to_db(conn, chunk, vec):
                success += 1
            else:
                fail += 1
        else:
            fail += 1

    conn.close()
