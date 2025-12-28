import ollama
import PyPDF2
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
import json
import os
import re

# --- Configuration ---
PDF_PATH = "654144034-BR-2170-1-SHIP-NBCD-MANUAL-Inc-Change-1-2-3.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "mistral:7b-instruct-q4_K_M"
DB_PATH = "vector_store.db"

# --- Global Variables ---
vector_store = []
text_chunks = []

def init_db():
    '''Initializes the SQLite database and creates the table if it doesn't exist.'''
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def is_db_populated():
    '''Checks if the database already contains embeddings.'''
    if not os.path.exists(DB_PATH):
        return False
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def get_pdf_text(pdf_path):
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

def get_text_chunks(text):
    '''Splits text into manageable chunks.    '''
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

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

    return text

def get_embeddings(model, prompt):
    '''Generates embeddings for a given prompt using the specified model.'''
    try:
        response = ollama.embeddings(model=model, prompt=prompt)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def populate_vector_store():
    '''Populates the vector store from the PDF if the DB is empty.'''
    if is_db_populated():
        print("Database is already populated. Skipping population.")
        return

    print("Populating vector store from PDF...")
    pdf_text = get_pdf_text(PDF_PATH)
    if not pdf_text:
        return
    
    cleaned_text = preprocess_text(pdf_text)
    chunks = get_text_chunks(cleaned_text)
    if not chunks:
        print("No text chunks to process.")
        return


    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(EMBEDDING_MODEL, chunk)
        if embedding:
            cursor.execute(
                "INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)",
                (chunk, json.dumps(embedding))
            )
            print(f"Embedded and saved chunk {i + 1}/{len(chunks)}")
    
    conn.commit()
    conn.close()
    print("Vector store populated successfully.")

def load_vector_store_from_db():
    '''Loads the vector store from the SQLite database.'''
    print("Loading vector store from database...")
    global vector_store, text_chunks
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk, embedding FROM embeddings")
    rows = cursor.fetchall()
   
    conn.close()

    text_chunks = [row[0] for row in rows]
    vector_store = [json.loads(row[1]) for row in rows]
    print(f"Loaded {len(vector_store)} embeddings from the database.")

def find_similar_chunks(query_embedding, top_k=3):
    '''Finds the most similar chunks to a query embedding using cosine similarity.'''
    if not vector_store:
        return []

    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        print("Warning: Query embedding has zero magnitude.")
        return []

    similarities = []
    for vec in vector_store:
        vec_np = np.array(vec)
        vec_norm = np.linalg.norm(vec_np)
        if vec_norm == 0:
            similarities.append(0)
        else:
            cosine_sim = np.dot(query_vec, vec_np) / (query_norm * vec_norm)
            similarities.append(cosine_sim)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [text_chunks[i] for i in top_indices]


def rag(query):
    '''Performs the RAG process.'''
    print(f"\nProcessing query: '{query}'")
    
    query_embedding = get_embeddings(EMBEDDING_MODEL, query)
    if not query_embedding:
        return "Could not generate query embedding."

    similar_chunks = find_similar_chunks(query_embedding)
    if not similar_chunks:
        return "No relevant information found in the document."

    context = "\n\n".join(similar_chunks)
    print("--- Retrieved Context ---\n" + context + "\n------------------------")

    prompt = (
        "You are an assistant who answers questions only using the provided context.\n"
        "Do not make up or assume information. If the answer is not in the context, reply with:\n"
        "'The context does not contain information about this.'\n\n"
        f"Context:\n{context}\n\nQuery: {query}"
    )
    
    print("Generating response...")
    try:
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )
        
        response_content = ""
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            response_content += content
        print()
        return response_content

    except Exception as e:
        return f"Error generating response from chat model: {e}"

def main():
    '''Main function to run the RAG system.'''
    init_db()
    populate_vector_store()
    load_vector_store_from_db()

    print("\n--- RAG System Ready ---")
    print("Enter your query below. Type 'exit' to quit.")

    while True:
        query = input("\nQuery: ")
        if query.lower() == 'exit':
            break
        rag(query)

if __name__ == "__main__":
    main()