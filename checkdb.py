import sqlite3
import os
import logging

DB_PATH = "rag.db"
LOG_FILE = "check_database.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def check_database():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database file '{DB_PATH}' does not exist!")
        return

    logger.info(f"Database file exists: {DB_PATH} | Size: {os.path.getsize(DB_PATH)} bytes")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='embeddings'
        """)

        if not cursor.fetchone():
            logger.error("Table 'embeddings' does not exist!")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            logger.info(f"Available tables: {tables}")
            return

        logger.info("Table 'embeddings' exists")

        cursor.execute("PRAGMA table_info(embeddings)")
        columns = cursor.fetchall()
        logger.info("Table Schema:")
        for col in columns:
            logger.info(f"  {col[1]} ({col[2]})")

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        logger.info(f"Total rows: {count}")

        if count > 0:
            cursor.execute("SELECT id, chunk, length(embedding) FROM embeddings LIMIT 1")
            row = cursor.fetchone()
            logger.info("Sample row:")
            logger.info(f"  ID: {row[0]}")
            logger.info(f"  Chunk (first 100 chars): {row[1][:100]}...")
            logger.info(f"  Embedding size: {row[2]} bytes")

            cursor.execute("SELECT id FROM embeddings")
            ids = [r[0] for r in cursor.fetchall()]
            logger.info(f"All row IDs: {ids}")
        else:
            logger.warning("No rows in the table!")
            logger.warning("Possible issues:")
            logger.warning("  1. The script failed to insert data")
            logger.warning("  2. Transactions weren't committed")
            logger.warning("  3. There was an error during insertion")

        conn.close()

    except Exception as e:
        logger.exception(f"Error reading database: {e}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Database Checker Started")
    logger.info("="*60)
    check_database()
    logger.info("="*60)
    logger.info("Database Checker Finished")
    logger.info("="*60)
