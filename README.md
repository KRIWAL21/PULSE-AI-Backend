<div align="center">

# ⚡ Dr. Pulse AI — Cloud RAG Diagnostic Backend

**Enterprise Retrieval-Augmented Generation & Medical Intelligence Engine**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Gemini](https://img.shields.io/badge/DeepMind-Gemini_2.5-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-000000?logo=pinecone&logoColor=white)](https://pinecone.io)
[![Supabase](https://img.shields.io/badge/PostgreSQL-Supabase-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

[Live API Swagger Docs](https://pulse-ai-backend-drf3.onrender.com/docs) • [Frontend Console Repo](https://github.com/KRIWAL21/PulseAI-FRONTEND) • [Report Bug](https://github.com/KRIWAL21/PULSE-AI-Backend/issues)

</div>

---

## 🧠 Executive Overview

**PULSE-AI-Backend** is a high-performance asynchronous REST and Server-Sent Events (SSE) diagnostic service built with **FastAPI**. It powers the Dr. Pulse clinical terminal by bridging cutting-edge **Google DeepMind Gemini 2.5** language models with a serverless **Pinecone Vector Database** containing indexed medical textbooks, clinical guidelines, and pharmacological literature.

To ensure enterprise data durability, the backend employs **SQLAlchemy ORM** connected to **Supabase Cloud PostgreSQL** for persistent user profile telemetry, authentication, and encrypted consultation histories.

---

## 🏛️ Architectural Pillars

* ⚡ **FastAPI Asynchronous Core:** High-throughput ASGI server supporting non-blocking concurrent requests and real-time SSE token streaming.
* 🌲 **Pinecone Serverless Vector Search:** Semantic similarity search utilizing high-dimensional embeddings (`text-embedding-004` / `embedding-001`) to retrieve exact top-K relevant clinical paragraphs in milliseconds.
* ☁️ **Supabase Cloud PostgreSQL:** Permanent relational persistence replacing volatile local storage. Automatically manages user accounts, chat sessions, and message transcripts.
* 🔒 **Enterprise JWT Security:** OAuth2 bearer token authentication with bcrypt password hashing and secure token validation.
* 💓 **Automated Health Heartbeat:** Built-in `/health` diagnostic endpoint optimized for zero-downtime cloud pingers (UptimeRobot / Cron-job.org).

---

## 📐 RAG Pipeline Workflow

```mermaid
sequenceDiagram
    participant C as 💻 React Frontend console
    participant F as ⚡ FastAPI Cloud Server
    participant P as 🌲 Pinecone Vector DB
    participant G as 🧠 DeepMind Gemini 2.5
    participant S as [(🔒 Supabase Postgres)]

    C->>F: POST /chat (Symptom Inquiry + JWT)
    F->>S: Verify User & Load Conversation Context
    F->>G: Generate Query Embedding
    G-->>F: High-Dimensional Vector
    F->>P: Semantic Search (Top-K Matches)
    P-->>F: Verified Medical Citations & Excerpts
    F->>G: Stream Grounded Clinical Prompt
    G-->>C: Real-Time SSE Token Stream
    F->>S: Persist Consultation Transcript
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description | Auth Required |
| :---: | :--- | :--- | :---: |
| `GET` | `/health` | Cloud pinger heartbeat & system diagnostics | No |
| `POST` | `/auth/signup` | Register new clinician or patient account | No |
| `POST` | `/auth/login` | Authenticate credentials & generate JWT token | No |
| `GET` | `/chat/conversations`| Retrieve persistent chat history for active user | Yes |
| `POST` | `/chat/conversations`| Initialize new consultation session | Yes |
| `POST` | `/chat` | Submit clinical inquiry for RAG inference | Yes |
| `DELETE`| `/chat/conversations/{id}`| Purge specific patient consultation record | Yes |

---

## 🚀 Local Setup & Installation

### Prerequisites
* **Python** `>= 3.10`
* **Pinecone Account** & Index (`medical-chatbot-gemini`)
* **Google Gemini API Key**
* **Supabase PostgreSQL Database URL**

### 1. Clone Repository
```bash
git clone https://github.com/KRIWAL21/PULSE-AI-Backend.git
cd PULSE-AI-Backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root folder:
```env
GEMINI_API_KEY="AIzaSy..."
PINECONE_API_KEY="pcsk_..."
DATABASE_URL="postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres"
JWT_SECRET_KEY="your-super-secret-jwt-key"
```

### 5. Launch FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📚 Document Indexing Script

To ingest new medical textbooks or clinical PDFs into Pinecone:
1. Place PDF files into the `data/` directory.
2. Execute the ingestion pipeline:
```bash
python store_index.py
```

---

<div align="center">
  <p>Engineered with ❤️ by <a href="https://github.com/KRIWAL21">KRIWAL21</a></p>
</div>
