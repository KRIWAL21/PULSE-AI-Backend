"""
PulseAI Backend — FastAPI + SQLite + RAG Pipeline
================================================

Architecture:
  1. FastAPI ASGI framework (async-first)
  2. SQLite database with SQLAlchemy ORM
  3. JWT authentication (no Firebase)
  4. History-Aware Retriever → Reformulates query using chat history
  5. MMR Retrieval → Maximal Marginal Relevance for diverse chunks
  6. Gemini 1.5 Flash LLM → Generates grounded medical answers
  7. SSE Streaming → Real-time token streaming via Server-Sent Events (async)
  8. Source Citations → Every answer references source documents
  9. Rate Limiting → Protects against abuse via slowapi
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from src.database import SessionLocal, get_db, init_db
from src.models import User, Conversation, Message
from src.schemas import (
    UserRegister, UserLogin, TokenResponse, UserResponse,
    AskRequest, AskResponse, MessageRequest, MessageResponse,
    ConversationCreate, ConversationUpdate, ConversationResponse,
    SummarizeRequest, SummarizeResponse
)
from src.auth import hash_password, verify_password, create_access_token, decode_access_token

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PulseAI Backend",
    description="AI Medical Assistant API",
    version="2.0.0"
)

# CORS — allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"}
))

@app.get("/health", tags=["System"])
async def health_check():
    """Uptime heartbeat check for zero-downtime cloud monitors."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "service": "PulseAI-Backend"}

# ─────────────────────────────────────────────────────────────────────────
# Global State — initialized once at startup
# ─────────────────────────────────────────────────────────────────────────

llm: Optional[ChatGoogleGenerativeAI] = None
retriever = None          # MMR retriever from Pinecone (may be None if Pinecone unavailable)
pinecone_available = False


# ─────────────────────────────────────────────────────────────────────────
# Dependency: Get Current User from JWT Token
# ─────────────────────────────────────────────────────────────────────────

async def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Extract user from Authorization header (Bearer token)."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]
    payload = decode_access_token(token)

    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload["sub"]
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ─────────────────────────────────────────────────────────────────────────
# RAG Pipeline Initialization
# ─────────────────────────────────────────────────────────────────────────

def initialize_rag_chain():
    """Initialize the RAG pipeline on startup.
    
    Sets up:
    - Gemini embeddings (text-embedding-004)
    - Pinecone vector store with MMR retriever (k=5, fetch_k=20)
    - Gemini 1.5 Flash LLM
    
    If Pinecone is unavailable (no index), the LLM still loads so the
    app can respond without RAG context (graceful degradation).
    """
    global llm, retriever, pinecone_available

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # --- 1. Embeddings ---
    print("Initializing Gemini Embeddings (gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # --- 2. Load Pinecone Vector Store ---
    index_name = "medical-chatbot-gemini"
    print(f"Loading Pinecone index: '{index_name}'...")
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
        )
        pinecone_available = True
        print("[OK] Pinecone retriever initialized (MMR k=5, fetch_k=20)")
    except Exception as e:
        print(f"[WARN] Could not load Pinecone index: {e}")
        print("       Run: python store_index.py (after placing PDFs in /Data)")
        print("       Continuing without RAG — LLM will answer from general knowledge.")
        retriever = None
        pinecone_available = False

    # --- 3. LLM ---
    print("Initializing Gemini Chat Model (gemini-2.5-flash)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.4,
    )
    print("[OK] LLM initialized.")


# ─────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────

def build_chat_history(raw_history: list, limit: int = 10) -> list:
    """Convert raw message dicts to LangChain message objects."""
    history = []
    for msg in raw_history[-limit:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def extract_sources(docs: list) -> list:
    """Extract unique source filenames from retrieved documents."""
    sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src:
            sources.add(os.path.basename(src))
    return sorted(list(sources))


# ─────────────────────────────────────────────────────────────────────────
# Event Handlers
# ─────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG pipeline on startup."""
    print("Starting PulseAI Backend...")
    init_db()
    print("[OK] Database initialized")
    initialize_rag_chain()
    print("[OK] PulseAI Backend ready.")


# ─────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user. Returns JWT token for immediate login."""
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(new_user)
    }


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with email and password. Returns JWT token."""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }


@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse.from_orm(current_user)


# ─────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "embeddings": "gemini-embedding-001",
        "retrieval": "mmr (k=5, fetch_k=20)" if pinecone_available else "unavailable",
        "rag_ready": pinecone_available and llm is not None,
        "llm_ready": llm is not None,
        "database": "sqlite"
    }


# ─────────────────────────────────────────────────────────────────────────
# CHAT ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────

@app.get("/conversations", response_model=list[ConversationResponse], tags=["Chat"])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all conversations for the current user."""
    convs = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return [ConversationResponse.from_orm(c) for c in convs]


@app.post("/conversations", response_model=ConversationResponse, tags=["Chat"])
async def create_conversation(
    conv_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    conv = Conversation(
        user_id=current_user.id,
        title=conv_data.title
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    return ConversationResponse.from_orm(conv)


@app.put("/conversations/{conv_id}", response_model=ConversationResponse, tags=["Chat"])
async def update_conversation(
    conv_id: str,
    conv_data: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update conversation title."""
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.title = conv_data.title
    db.commit()
    db.refresh(conv)

    return ConversationResponse.from_orm(conv)


@app.delete("/conversations/{conv_id}", tags=["Chat"])
async def delete_conversation(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()

    return {"detail": "Conversation deleted"}


@app.get("/conversations/{conv_id}/messages", response_model=list[MessageResponse], tags=["Chat"])
async def get_messages(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages in a conversation."""
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = db.query(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.created_at).all()

    result = []
    for msg in msgs:
        try:
            sources = json.loads(msg.sources) if msg.sources else []
        except Exception:
            sources = []

        result.append(MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            sender=msg.sender,
            text=msg.text,
            sources=sources,
            created_at=msg.created_at
        ))

    return result


# ─────────────────────────────────────────────────────────────────────────
# STREAMING ASK ENDPOINT
# ─────────────────────────────────────────────────────────────────────────

@app.post("/conversations/{conv_id}/ask/stream", tags=["Chat"])
@limiter.limit("20/minute")
async def ask_stream(
    conv_id: str,
    request: Request,
    ask_req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Streaming Q&A endpoint using Server-Sent Events (SSE).

    Flow:
      1. Validate conversation ownership
      2. Retrieve relevant docs via global Pinecone MMR retriever (if available)
      3. Stream tokens via async `llm.astream()` — non-blocking
      4. Persist user message + bot response to SQLite after streaming completes
      5. Emit SSE events: { sources }, { token }, { done } or { error }
    """
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check server logs.")

    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    question = ask_req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question exceeds 1000 character limit.")

    chat_history = build_chat_history(ask_req.chat_history)

    # Capture db session data needed inside generator (avoid session scope issues)
    conv_id_captured = conv_id

    async def generate():
        """Async generator — yields SSE events as tokens arrive."""
        full_answer = ""
        sources = []

        try:
            # ── Step 1: Retrieve context from Pinecone (if available) ──
            context_text = ""
            if retriever is not None:
                try:
                    retrieved_docs = await retriever.ainvoke(question)
                    sources = extract_sources(retrieved_docs)
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                except Exception as retrieval_err:
                    print(f"[WARN] Retrieval error: {retrieval_err}")
                    sources = []
                    context_text = ""

            # Send sources first so the client can display them immediately
            if sources:
                yield f"data: {json.dumps({'sources': sources})}\n\n"


            # ── Step 2: Build prompt messages ──
            if context_text:
                print(f"[RAG] Retrieved {len(retrieved_docs)} chunks for: '{question[:60]}'")
                system_content = (
                    "You are PulseAI, a concise AI medical assistant. "
                    "Answer the user's question using ONLY the retrieved context below. "
                    "Be brief — 2 to 4 short paragraphs maximum. No long bullet lists. "
                    "If the context does not contain the answer, say so in one sentence.\n\n"
                    f"RETRIEVED CONTEXT FROM MEDICAL ENCYCLOPEDIA:\n{context_text}\n\n"
                    "*For educational purposes only. Consult a doctor for medical advice.*"
                )
            else:
                system_content = (
                    "You are PulseAI, a concise AI medical assistant. "
                    "Answer briefly — 2 to 3 short paragraphs maximum. No long bullet lists.\n\n"
                    "*For educational purposes only. Consult a doctor for medical advice.*"
                )

            messages = [SystemMessage(content=system_content)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=question))

            # ── Persist User Message to DB ──
            from src.database import SessionLocal as _SessionLocal
            with _SessionLocal() as fresh_db:
                try:
                    fresh_db.add(Message(
                        conversation_id=conv_id_captured,
                        sender="user",
                        text=question,
                        sources="[]"
                    ))
                    fresh_db.commit()
                except Exception as db_err:
                    print(f"[DEBUG] DB Save User Message Error: {db_err}")

            # ── Step 3: Stream tokens asynchronously ──
            async for chunk in llm.astream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # ── Step 4: Persist messages to DB ──
            # Use a fresh session to avoid event-loop/thread issues with the
            # request-scoped session that was already yielded.
            from src.database import SessionLocal as _SessionLocal
            with _SessionLocal() as fresh_db:
                print("[DEBUG] Attempting to save messages to DB...")
                try:
                    fresh_db.add(Message(
                        conversation_id=conv_id_captured,
                        sender="bot",
                        text=full_answer,
                        sources=json.dumps(sources)
                    ))
                    fresh_db.commit()
                    print("[DEBUG] Messages successfully saved to DB!")
                except Exception as db_err:
                    print(f"[DEBUG] DB Save Error: {db_err}")

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            import traceback
            with open("streaming_error.log", "w") as f:
                f.write(traceback.format_exc())
            print(f"[ERROR] Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        }
    )


# ─────────────────────────────────────────────────────────────────────────
# SUMMARIZE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────

@app.post("/conversations/{conv_id}/summarize", response_model=SummarizeResponse, tags=["Chat"])
@limiter.limit("5/minute")
async def summarize(
    conv_id: str,
    request: Request,
    sum_req: SummarizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Summarize a conversation using the LLM directly."""
    if llm is None:
        raise HTTPException(status_code=500, detail="Language model not initialized.")

    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not sum_req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    transcript = "\n".join([
        f"{'USER' if msg['sender'] == 'user' else 'PULSEAI'}: {msg['text']}"
        for msg in sum_req.messages
    ])

    summary_prompt = (
        "Analyze this medical chat conversation and provide a structured summary.\n\n"
        "Format your response using these exact markdown headers:\n\n"
        "## 🏥 Main Topics Discussed\n"
        "• [list each topic]\n\n"
        "## 💊 Key Medical Information\n"
        "• [list key medical facts, conditions, medications mentioned]\n\n"
        "## ✅ Recommendations Mentioned\n"
        "• [list any health recommendations, or 'None mentioned']\n\n"
        "## ⚠️ Safety Notes\n"
        "• [note any safety information or disclaimers]\n\n"
        f"CONVERSATION TRANSCRIPT:\n---\n{transcript}\n---"
    )

    try:
        response = await llm.ainvoke(summary_prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        return SummarizeResponse(summary=summary_text)
    except Exception as e:
        print(f"[ERROR] /summarize: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary")


# ─────────────────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
