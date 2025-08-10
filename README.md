🩺 PulseAI - Your AI Medical Assistant
PulseAI is an intelligent, document-aware chatbot designed to answer medical questions based on a provided set of PDF documents. It leverages a powerful combination of Large Language Models (LLMs) via Google's Gemini API, vector embeddings, and a high-performance vector database (Pinecone) to deliver accurate, context-aware responses through a clean and intuitive web interface.

(Suggestion: Replace the URL above with a real screenshot of your running application)

✨ Key Features
Intelligent Q&A: Ask complex medical questions and receive detailed answers sourced directly from your documents.

RAG Pipeline: Utilizes a Retrieval-Augmented Generation (RAG) architecture to ensure answers are grounded in the provided context, minimizing hallucinations.

High-Performance Vector Search: Uses Pinecone's serverless vector database for fast and efficient similarity searches.

Google Gemini Integration: Powered by Google's Gemini models for both generating high-quality embeddings and crafting natural language responses.

Modern Web Interface: A clean, responsive, and user-friendly chat interface built with HTML and Tailwind CSS.

Scalable Backend: A lightweight and robust backend powered by Flask.

🛠️ Tech Stack
Backend: Python, Flask

LLM & Embeddings: Google Gemini (gemini-1.5-flash, embedding-001)

Framework: LangChain

Vector Database: Pinecone

Frontend: HTML, Tailwind CSS, JavaScript

Deployment: (Example: Render for backend, Netlify for frontend)

🚀 Getting Started
Follow these instructions to get a local copy up and running.

Prerequisites
Python 3.10 or higher

A Pinecone account and API key

A Google Gemini API key

Installation
Clone the repository:

git clone https://github.com/your-username/PulseAI-Your-Digital-Doctor.git
cd PulseAI-Your-Digital-Doctor

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

Set up your environment variables:
Create a file named .env in the root of your project and add your API keys:

PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

⚙️ How to Run the Project
The project has two main parts: indexing your data and running the web application.

Step 1: Index Your Documents
Before you can ask questions, you need to process your PDF files and store their embeddings in Pinecone.

Place all your medical PDF files into the /data directory.

Run the store_index.py script from the terminal:

python store_index.py

This script will read the PDFs, generate embeddings using Gemini, and upload them to your Pinecone index (medical-chatbot-gemini). This only needs to be done once, or whenever you add new documents.

Step 2: Run the Web Application
Once your data is indexed, you can start the chatbot server.

Run the app.py script from the terminal:

python app.py

Your terminal will show that the server is running, usually on http://127.0.0.1:8080.

Open your web browser and navigate to that address to start chatting with PulseAI!

⚠️ Disclaimer
PulseAI is an informational tool and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.