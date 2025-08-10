# New, corrected imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List
from langchain.schema import Document

# from langchain.embeddings import HuggingFaceEmbeddings





#extract text from pdf
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",  # all pdf files
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_doc(docs: List[Document]) -> List[Document]:
    """
    Filter documents to only include the minimal necessary information.
    """
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk



def download_embeddings():
    """
    Download the embeddings model from HuggingFace.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings