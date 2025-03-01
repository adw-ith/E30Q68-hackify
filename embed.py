# pdf_embedding.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from typing import List, Dict, Any
import pickle

def load_and_split_pdf(pdf_path: str) -> List[Dict[Any, Any]]:
    """
    Load a PDF and split it into chunks.
    """
    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split PDF into {len(chunks)} chunks")
    return chunks

def create_embeddings_and_store(chunks: List[Dict[Any, Any]], index_name: str) -> None:
    """
    Create embeddings for chunks and store them in a FAISS index.
    """
    # Initialize the Sentence Transformers embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create a FAISS index
    db = FAISS.from_documents(chunks, embedding_model)
    
    # Save the index to disk
    db.save_local(index_name)
    print(f"Saved FAISS index to {index_name}")

def main():
    # Path to your government policy PDF
    pdf_path = "data/gov.pdf"
    
    # Name for the index
    index_name = "government_policies_index"
    
    # Load and split the PDF
    chunks = load_and_split_pdf(pdf_path)
    
    # Create embeddings and store them
    create_embeddings_and_store(chunks, index_name)
    
    print("Successfully embedded and stored PDF data!")

if __name__ == "__main__":
    main()