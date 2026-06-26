from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def load_pdf(data: str) -> List[Document]:
    """Load all PDF files from the given directory path."""
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_doc(docs: List[Document]) -> List[Document]:
    """
    Strip all metadata except 'source' to reduce Pinecone payload size.
    Keeping 'source' allows us to surface citations in the chat UI.
    """
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        for doc in docs
    ]


def text_split(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.
    
    chunk_size=1000: Enough context for the LLM to understand a concept
    chunk_overlap=150: Prevents cutting answers in half at chunk boundaries
    
    Why not larger? Pinecone has metadata size limits, and LLMs perform
    better with focused, relevant context rather than huge chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)