import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_and_chunk_pdf(pdf_path: str):
    """
    Loads a PDF and splits it into chunks.
    
    Args:
        pdf_path (str): The relative path to the PDF file.
        
    Returns:
        list: A list of Document objects (chunks).
    """
    
    # 1. Load the raw PDF
    print(f"Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 2. Split the text
    # chunk_size=1000: A balance between context and precision
    # chunk_overlap=200: Ensures we don't cut sentences in half at the edges
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")
    
    return chunks

    


def ingest_to_vector_db(chunks):
    """
    Takes text chunks, creates embeddings, and saves them to ChromaDB.
    """
    # 1. Initialize the Embedding Model (Local / Free)
    print("Loading local embedding model... (This runs on your CPU)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Initialize and Populate the Vector Database
    # persist_directory tells Chroma where to save the files on your disk
    print("Creating embeddings and saving to ChromaDB... (this may take a moment)")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Success! Data saved to ./chroma_db folder.")
    return db    
    
if __name__ == "__main__":
    # 1. Load and Chunk
    chunks = load_and_chunk_pdf("data/sample.pdf")
    
    # 2. Embed and Store
    ingest_to_vector_db(chunks)
    