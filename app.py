import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LangChain and Google/Pinecone Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Initialize Flask App ---
# We no longer need the static_folder argument
app = Flask(__name__)
CORS(app)

# --- Global variable for the RAG chain ---
rag_chain = None

def initialize_rag_chain():
    """
    Initializes all components of the RAG pipeline.
    This function is called once when the server starts.
    """
    global rag_chain

    # --- 1. Load API Keys ---
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # --- 2. Initialize Gemini Embeddings ---
    print("Initializing Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # --- 3. Load Existing Pinecone Index ---
    index_name = "medical-chatbot-gemini"
    print(f"Loading Pinecone index: {index_name}...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # --- 4. Create Retriever ---
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # --- 5. Create Prompt Template ---
    system_prompt = (
        "You are a helpful medical assistant. Use the following retrieved context to "
        "answer the user's question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise and clear."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # --- 6. Initialize Gemini Chat Model ---
    print("Initializing Gemini Chat Model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )

    # --- 7. Create the RAG Chain ---
    print("Creating the RAG chain...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- API Endpoints ---

# We have removed the @app.route('/') endpoint.
# The backend's only job is to handle API requests.

@app.route('/ask', methods=['POST'])
def ask():
    """Handles chat requests from the frontend."""
    if rag_chain is None:
        return jsonify({"error": "Chatbot is not initialized. Please check server logs."}), 500
    
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided in JSON body"}), 400
        
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    try:
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer": response.get("answer", "Sorry, I could not generate a response.")})
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        print("--- Attempting to initialize the chatbot... ---")
        initialize_rag_chain()
        print("✅ PulseAI Chatbot initialization complete.")

        if rag_chain:
            print("--- Starting Flask API server... ---")
            app.run(host="0.0.0.0", port=8080, debug=False)
        else:
            print("❌ CRITICAL ERROR: RAG chain is None. Server will not start.")

    except Exception as e:
        print(f"❌ FAILED TO INITIALIZE AND START SERVER. ERROR: {e}")



        # In app.py

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handles summarization requests from the frontend."""
    data = request.get_json()
    messages = data.get("messages")

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Combine the chat history into a single string
    transcript = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in messages])
    
    # Create a new prompt for summarization
    summary_prompt = f"""
    Based on the following chat conversation, please provide a concise summary. 
    Use bullet points for the key topics discussed.

    Conversation:
    {transcript}

    Summary:
    """

    try:
        # Use the same RAG chain's LLM to generate the summary
        # NOTE: This assumes your `llm` is accessible. If not, you might need to re-initialize it.
        # For simplicity, we re-use the rag_chain's llm component.
        summary_response = rag_chain.question_answer_chain.llm.invoke({"input": summary_prompt})
        
        # Extract the text from the response
        summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

        return jsonify({"summary": summary_text})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "Failed to generate summary."}), 500

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Allow all origins — so your React app can call the backend

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     question = data.get('question', '')

#     # Temporary test response
#     answer = f"You asked: {question}"
#     return jsonify({"answer": answer})

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8080, debug=True)
