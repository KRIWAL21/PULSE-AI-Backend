# store_index.py — Build the Pinecone vector index from medical PDF documents
#
# HOW THIS WORKS:
# 1. Load all PDFs from the /data directory
# 2. Split them into overlapping chunks (1000 chars, 150 overlap)
# 3. Generate vector embeddings using Gemini gemini-embedding-001
# 4. Upsert all vectors to Pinecone in batches
#
# Run this script ONCE to populate Pinecone, then use app.py to serve queries.
# Re-run if you add new PDF documents to /data.
# The script auto-resumes from where it left off if interrupted.
#
# RATE LIMITS (free tier):
# - gemini-embedding-001: 100 requests/minute
# - batch_size=25 + 20s sleep = ~75 req/min (safe margin)

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Force unbuffered output so logs appear immediately
sys.stdout.reconfigure(line_buffering=True)

LOG_FILE = "indexing.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
from pinecone import Pinecone, ServerlessSpec

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.helper import load_pdf, filter_to_minimal_doc, text_split


def make_embeddings(api_key):
    """Create a fresh embeddings client (recreate on network errors)."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def embed_with_retry(texts, api_key, batch_num, max_attempts=8):
    """Embed texts with retry for rate limits and network errors."""
    embeddings = make_embeddings(api_key)
    for attempt in range(max_attempts):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
            is_network = "10053" in err or "ReadError" in err or "ConnectionError" in err or "10054" in err

            if is_rate_limit:
                wait = 70 + (attempt * 30)
                log(f"  [RATE LIMIT] Batch {batch_num}: waiting {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
                # Recreate client after long wait to get fresh connection
                embeddings = make_embeddings(api_key)

            elif is_network:
                wait = 15 * (attempt + 1)
                log(f"  [NETWORK ERROR] Batch {batch_num}: reconnecting in {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
                # Always recreate client on network error
                embeddings = make_embeddings(api_key)

            else:
                raise

    raise RuntimeError(f"Batch {batch_num} failed after {max_attempts} attempts.")


def main():
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Load and process PDFs
    log("=== store_index.py started ===")
    log("Loading PDF documents from data/...")
    raw_docs = load_pdf(data='data/')
    minimal_docs = filter_to_minimal_doc(raw_docs)
    chunks = text_split(minimal_docs)
    log(f"[OK] Processed {len(chunks)} chunks from {len(raw_docs)} pages")

    # Initialize Pinecone
    log("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "medical-chatbot-gemini"

    # Create index if it doesn't exist (3072 dims for gemini-embedding-001)
    if index_name not in pc.list_indexes().names():
        log(f"Creating new Pinecone index '{index_name}' (dimension=3072, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        log("[OK] Index created")
    else:
        log(f"[INFO] Index '{index_name}' already exists.")

    # Auto-detect resume point from existing vector count
    index = pc.Index(index_name)
    batch_size = 25  # Small batches: 25 req/batch * 3 batches/min = 75 req/min (safe)

    existing = index.describe_index_stats().total_vector_count
    resume_from = (existing // batch_size) * batch_size
    if resume_from > 0:
        log(f"[RESUME] Found {existing} existing vectors. Resuming from chunk {resume_from}...")
        chunks = chunks[resume_from:]

    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    if total_chunks == 0:
        log("[DONE] All chunks already indexed!")
        return

    log(f"Uploading {total_chunks} remaining chunks in {total_batches} batches of {batch_size}...")
    log("[INFO] Sleeping 20s between batches to respect 100 req/min free tier limit.")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        batch_num = i // batch_size + 1
        global_chunk_start = resume_from + i

        batch_embeddings = embed_with_retry(texts, GEMINI_API_KEY, batch_num)

        vectors = [
            (
                f"chunk_{global_chunk_start + j}",
                batch_embeddings[j],
                {"text": doc.page_content, "source": doc.metadata.get("source", "")}
            )
            for j, doc in enumerate(batch)
        ]
        index.upsert(vectors=vectors)
        log(f"  [OK] Batch {batch_num}/{total_batches} | chunks {global_chunk_start}-{global_chunk_start + len(batch) - 1} | total indexed: {existing + global_chunk_start - resume_from + len(batch)}")

        # Sleep between batches to stay within rate limit
        if i + batch_size < total_chunks:
            log("  [WAIT] Sleeping 20s...")
            time.sleep(20)

    log("[DONE] All chunks indexed in Pinecone successfully!")
    log("Restart app.py to enable RAG.")


if __name__ == "__main__":
    main()