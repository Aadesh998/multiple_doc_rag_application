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
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)

    text = re.sub(r'\.{3,}', '', text)

    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    def normalize_acronyms(match):
        return match.group(0).upper()
    
    text = re.sub(r'\b[A-Z]{2,}\b', normalize_acronyms, text)

    print("Text preprocessing complete.")
    output_path = os.path.abspath("preprocess.txt")
    print(f"Saving preprocessed text to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def get_text_chunks(text):
    '''Splits text into manageable chunks.    '''
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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
            SELECT e.chunk, v.embedding
            FROM vec_embeddings v
            JOIN embeddings e ON e.id = v.id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """, (q, k)).fetchall()

        conn.close()

        results = []
        for chunk, embedding_blob in rows:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
            results.append({
                "chunk": chunk,
                "embedding": embedding
            })

        return query_vector, results

    except Exception as e:
        logger.exception(e)
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
        query_vec, res = search_for_go(q, k)
        print(json.dumps({
            "query_vector": query_vec,
            "result": res
        }))
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
