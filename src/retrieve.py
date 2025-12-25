import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def retrieve_context(query_text):
    """
    Searches ChromaDB for the most relevant chunks.
    """
    # 1. Initialize the same embedding model used for ingestion
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Connect to the existing database
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 3. Perform the search (k=3 means 'get top 3 results')
    print(f"Searching for: '{query_text}'")
    results = db.similarity_search(query_text, k=3)
    
    return results

if __name__ == "__main__":
    # Test Query 
    query = "What are the main topics discussed in the document?" 
    
    retrieved_docs = retrieve_context(query)
    
    print("\n--- RETRIEVED RESULTS ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...") # Print first 200 chars only
    print("-------------------------")