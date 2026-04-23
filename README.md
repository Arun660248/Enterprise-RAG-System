# Enterprise Document RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built to query and extract insights from complex documents using Google's Gemini models and LangChain.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** LangChain (LCEL)
* **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`)
* **Vector Database:** FAISS (Local)

## 🚀 Features
* **Document Ingestion:** Automated chunking and embedding of PDF documents.
* **Semantic Search:** Fast and accurate context retrieval using FAISS.
* **Enterprise Citation:** Automatically cites the source document and page number to prevent AI hallucinations.
