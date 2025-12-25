# Enterprise RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-orange.svg)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4.svg?logo=google)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Embeddings-FFD21F.svg?logo=huggingface)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-005C4E.svg)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg?logo=docker)

## Description

A production-grade Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents. It features a robust error-handling system for API quotas and uses a local vector database for privacy and speed.

## Tech Stack

*   **Language:** Python 3.11
*   **Framework:** LangChain (Orchestration), Streamlit (Frontend UI)
*   **AI Models:**
    *   Embedding: HuggingFace (`all-MiniLM-L6-v2`) running locally (CPU-optimized).
    *   LLM: Google Gemini 1.5 Flash (via `langchain-google-genai`).
*   **Database:** ChromaDB (Persistent local vector store).
*   **DevOps:** Docker (Multi-stage build), Environment Variable management (`python-dotenv`).

## Key Features

1.  **Hybrid Architecture:** Leverages cost-effective local embedding models for document processing while utilizing powerful cloud-based LLMs (Google Gemini 1.5 Flash) for advanced reasoning and response generation.
2.  **Robust Error Handling:** Engineered to gracefully handle common API issues, specifically `429 Resource Exhausted` errors. Includes fallback strategies for experimental model APIs to ensure continuous operation and a smooth user experience.
3.  **Session Caching:** Implements Streamlit's `@st.cache_resource` decorator to efficiently cache heavy models and resources. This prevents redundant reloading on subsequent user interactions, significantly reducing latency and improving application responsiveness.
4.  **Interactive UI:** Provides a clean and intuitive chat interface built with Streamlit, offering real-time conversation history tracking for an engaging user experience.

## How it Works

The Enterprise RAG Knowledge Assistant operates through a multi-step process to deliver accurate and contextually relevant responses to user queries:

1.  **Document Ingestion:** PDF documents are processed, chunked into smaller, manageable segments, and then converted into numerical representations (embeddings) using a locally run HuggingFace model (`all-MiniLM-L6-v2`).
2.  **Vector Storage:** These embeddings, along with their corresponding text chunks, are stored in a local ChromaDB vector database. This ensures data privacy and enables fast retrieval.
3.  **Query Processing:** When a user submits a query, it is also converted into an embedding.
4.  **Relevant Document Retrieval:** The query embedding is used to search the ChromaDB for the most semantically similar document chunks. This retrieval step identifies the most relevant information from your ingested PDFs.
5.  **Augmented Generation:** The retrieved document chunks are then provided as context to the Google Gemini 1.5 Flash LLM. The LLM uses this context, along with its extensive knowledge, to generate a comprehensive and accurate answer to the user's query.
6.  **Interactive Chat:** The generated response is displayed in the Streamlit UI, maintaining a history of the conversation for an interactive chat experience.

## Installation Instructions

Follow these steps to set up and run the Enterprise RAG Knowledge Assistant locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/RAG-Knowledge-Assistant.git
    cd RAG-Knowledge-Assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
    _Replace `"YOUR_GOOGLE_API_KEY"` with your actual Google Gemini API key._

4.  **Run ingestion script:**
    This script processes your PDF documents and populates the local vector database.
    ```bash
    python src/ingest.py
    ```

5.  **Launch the application:**
    ```bash
    streamlit run src/app.py
    ```
    The application will open in your web browser.

## Troubleshooting

### Common API Quota Errors (429 Resource Exhausted)

This error indicates that you have exceeded the usage limits for the Google Gemini API. Here's how to address it:

*   **Wait and Retry:** API quotas are often reset after a certain period (e.g., per minute, per day). Waiting a few minutes and retrying your request can often resolve the issue.
*   **Check Google Cloud Console:** Visit the [Google Cloud Console](https://console.cloud.google.com/apis/dashboard) to monitor your API usage and quota limits. You may be able to request an increase in your quota if your project requires higher limits.
*   **Optimize API Calls:** If you are making a large number of requests in a short period, consider implementing client-side rate limiting or batching your requests to stay within the allowed limits.
*   **Review `rag.py` for Fallback Logic:** The application is designed with basic error handling. Review the `src/rag.py` file to understand how `429` errors are caught and managed, and consider enhancing this logic for more sophisticated retry mechanisms or alternative model usage if applicable.

---

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details. (Note: A `LICENSE` file should be created in the repository if not already present.)
