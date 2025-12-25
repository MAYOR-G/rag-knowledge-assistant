# Enterprise RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-005C4E?style=for-the-badge)

A production-grade Retrieval-Augmented Generation (RAG) system that evolved from a basic prototype to a resilient, full-stack application. Query your PDF documents using natural language, powered by **Google Gemini** and **LangChain**.

---

## 🚀 The Evolution (Architecture)

This project demonstrates the iterative development of a RAG pipeline, showcasing the journey from a minimal viable prototype to a production-ready system.

### Version 1: The Simple RAG

**File:** [`src/rag.py`](src/rag.py)

| Aspect | Details |
|--------|---------|
| **Goal** | Minimal viable prototype to validate the core concept. |
| **Tech** | Standard Vector Search (ChromaDB) + Google Gemini 1.5 Flash. |
| **How it Works** | User queries are embedded and matched against document embeddings using cosine similarity. The top-k results are passed to the LLM for answer generation. |

**Limitations:**
- ❌ **Keyword Retrieval Failure:** Struggled with specific terms like SKU numbers, acronyms, and exact phrases. Vector similarity is great for concepts, not exact matches.
- ❌ **No Conversation Memory:** Each query was treated in isolation. Users couldn't ask follow-up questions like *"How much does it cost?"* without restating the full context.

---

### Version 2: The Advanced RAG

**File:** [`src/rag_advanced.py`](src/rag_advanced.py)

| Aspect | Details |
|--------|---------|
| **Goal** | Production-grade resilience, accuracy, and user experience. |
| **Tech** | Hybrid Search (BM25 + Vector) + History-Aware Retriever + Gemini 1.5 Flash. |

#### Upgrade 1: Hybrid Search

**Implementation:** `EnsembleRetriever` combining `BM25Retriever` (keyword) with `ChromaDB` (semantic).

```python
final_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)
```

**Why it matters:**
- **BM25 (Keyword Search):** Excels at finding exact matches—perfect for part numbers, acronyms, and technical terms.
- **Vector Search (Semantic):** Understands meaning and context—finds relevant information even when wording differs.
- **Combined:** Best of both worlds. No more missed results.

---

#### Upgrade 2: Conversational Memory

**Implementation:** `HistoryAwareRetriever` with `MessagesPlaceholder` for chat history.

```python
history_aware_retriever = create_history_aware_retriever(
    llm, final_retriever, context_prompt
)
```

**Why it matters:**
- Users can now have natural, multi-turn conversations.
- Follow-up questions like *"What about the warranty?"* or *"How much does it cost?"* work seamlessly without restating context.
- The system reformulates queries using chat history to retrieve the most relevant documents.

---

#### Upgrade 3: Source Citations

**Implementation:** The Streamlit UI displays exact page numbers and source snippets alongside every answer.

**Why it matters:**
- ✅ **Trust:** Users can verify answers against the original source.
- ✅ **Transparency:** Clear provenance for every piece of information.
- ✅ **Compliance:** Essential for regulated industries where source attribution is required.

---

## 🛠️ Tech Stack & Features

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive chat UI with `@st.cache_resource` for model caching and performance optimization. |
| **Orchestration** | LangChain | Chains, Retrievers, Prompts, and Document Loaders for RAG pipeline construction. |
| **Database** | ChromaDB | Local persistent vector storage for document embeddings. |
| **Keyword Search** | `rank_bm25` | BM25 algorithm for exact keyword matching. |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Lightweight, CPU-optimized sentence embeddings running locally. |
| **LLM** | Google Gemini 1.5 Flash | Fast, cost-effective language model for response generation. |
| **Environment** | `python-dotenv` | Secure API key management via `.env` files. |
| **Containerization** | Docker | Multi-stage build for production deployment. |

---

## 📁 Project Structure

```
rag-knowledge-assistant/
├── src/
│   ├── app.py              # Streamlit UI (Basic RAG)
│   ├── app_advanced.py     # Streamlit UI (Advanced RAG)
│   ├── rag.py              # Simple RAG chain
│   ├── rag_advanced.py     # Advanced RAG with Hybrid Search + Memory
│   └── ingest.py           # Document ingestion pipeline
├── data/                   # PDF documents for ingestion
├── chroma_db/              # Persistent vector database
├── notebooks/              # Jupyter notebooks for experimentation
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── .env                    # API keys (not committed)
```

---

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud API Key with Gemini access

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-knowledge-assistant.git
   cd rag-knowledge-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Ingest your documents:**
   
   Place your PDF files in the `data/` directory, then run:
   ```bash
   python src/ingest.py
   ```

5. **Launch the application:**

   **Basic RAG (Simple Vector Search):**
   ```bash
   streamlit run src/app.py
   ```

   **Advanced RAG (Hybrid Search + Memory):**
   ```bash
   streamlit run src/app_advanced.py
   ```

---

## 🔗 Live Demo

| Version | Link |
|---------|------|
| **Simple RAG** | [LINK]([https://rag-knowledge-assistant-ahwodv2q6do5iuux8zktwb.streamlit.app/] |
| **Advanced RAG** | [Coming Soon]() |

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using LangChain and Google Gemini
</p>
