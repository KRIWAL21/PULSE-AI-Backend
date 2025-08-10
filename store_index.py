# store_index.py (Corrected and Cleaned)

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# --- LangChain and Google Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Your Helper Function Imports ---
# These now correctly match the names in your helper.py file
from src.helper import load_pdf, filter_to_minimal_doc, text_split

# In store_index.py

def main():
    """
    Main function to process documents and store them in Pinecone using Gemini.
    """
    # 1. Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # 2. Load and Process Documents
    print("Loading and processing PDF documents...")
    extracted_data = load_pdf(data='data/')
    filter_data = filter_to_minimal_doc(extracted_data)
    text_chunks = text_split(filter_data)
    print(f"Successfully loaded and split documents into {len(text_chunks)} chunks.")

    # 3. Initialize Gemini Embeddings
    print("Initializing Google Gemini embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # 4. Initialize Pinecone
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "medical-chatbot-gemini"

    # 5. Create Pinecone Index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{index_name}' with dimension 768...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    # --- THIS IS THE CORRECTED SECTION ---
    # 6. Upsert Embeddings to Pinecone in Batches
    print(f"Upserting {len(text_chunks)} chunks to Pinecone index '{index_name}' in batches...")
    
    # Get a handle to the Pinecone index
    index = pc.Index(index_name)
    
    # Define the batch size
    batch_size = 100 
    
    for i in range(0, len(text_chunks), batch_size):
        # Get the current batch of documents
        batch_docs = text_chunks[i:i + batch_size]
        
        # Extract the text content
        batch_texts = [doc.page_content for doc in batch_docs]
        
        # Generate embeddings for the batch
        batch_embeddings = embeddings.embed_documents(batch_texts)
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for j, doc in enumerate(batch_docs):
            vectors_to_upsert.append(
                (
                    f"chunk_{i+j}",  # Unique ID for each chunk
                    batch_embeddings[j],
                    {"text": doc.page_content, "source": doc.metadata.get("source")}
                )
            )
        
        # Upsert the batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"  > Upserted batch {i//batch_size + 1}")

    print("✅ All documents have been successfully processed and stored in Pinecone.")