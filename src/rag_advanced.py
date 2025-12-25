import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def get_advanced_chain():
    print("--- INITIALIZING ADVANCED RAG ---")
    
    # 1. SETUP EMBEDDINGS (Cached locally)
    print("1. Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. LOAD VECTOR DB
    # Ensure the directory exists
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError("ChromaDB not found! Please run 'src/ingest.py' first.")
        
    print("2. Connecting to Vector DB...")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. SETUP KEYWORD RETRIEVER (BM25) - WITH SAFETY CHECK
    # We check if the source file exists. If not, we skip Hybrid Search.
    pdf_path = "data/sample.pdf"
    keyword_retriever = None
    
    if os.path.exists(pdf_path):
        print(f"3. Building Keyword Index from {pdf_path}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = 3
            print("   - BM25 Index Built Successfully.")
        except Exception as e:
            print(f"   - WARNING: Failed to build BM25: {e}")
    else:
        print(f"   - WARNING: '{pdf_path}' not found. Skipping Keyword Search.")

    # 4. COMBINE RETRIEVERS
    if keyword_retriever:
        print("4. Enabling Hybrid Search (Vector + Keyword)")
        final_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )
    else:
        print("4. Fallback: Using Vector Search Only")
        final_retriever = vector_retriever

    # 5. SETUP LLM
    print("5. Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # 6. CREATE CHAINS
    print("6. Compiling Logic Chains...")
    
    # History Awareness Prompt
    context_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed "
        "and otherwise return it as is."
    )
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, final_retriever, context_prompt)

    # QA Prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. \n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("--- SYSTEM READY ---")
    return rag_chain
