# 📚 PDF Q&A Assistant

A conversational question answering system that lets you upload PDF documents and ask questions about them using a state of the art RAG pipeline built from scratch.

## Features
- 📄 Multiple PDF support with per-session filtering
- 🔍 Hybrid search — vector similarity (BAAI/bge-m3) + BM25 keyword matching
- 🎯 Cross encoder reranking for better context quality
- 💬 Conversation memory — follow up questions reference previous answers
- 📌 Source citations — every answer shows PDF name and page number
- ⚡ Streaming responses via Groq API
- 🛡️ Error handling for corrupted PDFs and API failures

## How it Works
```
PDF → Extract text → Chunk → Embed with bge-m3 → Store in ChromaDB
                                                         ↓
User question → Vector search + BM25 → Combine → Rerank → Top 3 chunks
                                                         ↓
                                    Groq LLM (llama-3.3-70b) → Streaming answer
```

## Tech Stack
- **Embeddings:** SentenceTransformers (BAAI/bge-m3)
- **Vector Database:** ChromaDB
- **Keyword Search:** BM25 (rank-bm25)
- **Reranking:** CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **UI:** Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ankitbista483/pdf-qa-rag.git
cd pdf-qa-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```
GROQ_API_KEY=your_groq_api_key
```

4. Run the app:
```bash
streamlit run streamlit_app.py
```

## Performance
- Embeds 3,330+ chunks from 400+ page documents
- 3.6 second end-to-end response time with full hybrid search and reranking