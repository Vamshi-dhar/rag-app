# NCERT Physics Class 11 RAG Q&A System

A Retrieval Augmented Generation (RAG) application that answers questions from the NCERT Class 11 Physics textbook using Qdrant vector database.

## Setup

1. **Create virtual environment** (Python 3.12):
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
uv pip install qdrant-client pymupdf sentence-transformers openai python-dotenv
```

## Usage

### Step 1: Ingest PDF into Qdrant

First, process and store the PDF content in Qdrant:

```bash
python ingest_pdf.py
```

This will:
- Extract text from the PDF
- Split it into manageable chunks
- Generate embeddings using sentence-transformers
- Store everything in your Qdrant vector database

### Step 2: Run Q&A Session

Start the interactive Q&A system:

```bash
python qa_app.py
```

Then ask questions like:
- "What is Newton's first law?"
- "Explain the concept of momentum"
- "What is the difference between speed and velocity?"
- "Define kinetic energy"

## Features

- **Vector Search**: Uses semantic search to find relevant content
- **Context-Aware**: Retrieves multiple relevant passages for comprehensive answers
- **Page References**: Shows which pages the information comes from
- **Interactive**: Command-line interface for continuous Q&A
- **Optional LLM**: Can use OpenAI API for generated answers (add OPENAI_API_KEY to .env)

## Project Structure

- `ingest_pdf.py` - PDF processing and Qdrant ingestion
- `qa_app.py` - Interactive Q&A application
- `connet_qdrant.py` - Qdrant connection test
- `.env` - Environment variables (API keys)
- `requirements.txt` - Python dependencies

## Notes

- The system works with or without OpenAI API key
- Without OpenAI: Returns relevant text passages from the book
- With OpenAI: Generates natural language answers based on the context
