import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_rag_chain():
    # 1. Embedding Model (Local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector DB Retriever
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 3. Gemini LLM Setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # 4. Prompt Template
    template = """You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

if __name__ == "__main__":
    print("Initializing Gemini RAG Chain...")
    chain = get_rag_chain()
    
    # Update this query!
    user_query = "What does the document say about marketing?"
    print(f"Asking Gemini: {user_query}")
    
    response = chain.invoke(user_query)
    
    print("\n--- GEMINI ANSWER ---")
    print(response)
    print("---------------------")