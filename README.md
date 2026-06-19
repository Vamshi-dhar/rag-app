# NCERT Physics Class 11 RAG Q&A System

A Retrieval Augmented Generation (RAG) application that answers questions from the NCERT Class 11 Physics textbook using Qdrant vector database and OpenAI.

## Features

- **OpenAI Embeddings**: Uses `text-embedding-3-small` for semantic search
- **GPT-4 Answers**: Generates clear, accurate answers using `gpt-4o-mini`
- **Qdrant Vector DB**: Fast semantic search across the entire textbook
- **Secure**: All API keys loaded from environment variables
- **Interactive**: Command-line Q&A session

## Setup

### 1. Create Virtual Environment (Python 3.12)

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
uv pip install qdrant-client pymupdf openai python-dotenv
```

### 3. Configure Environment Variables

Your `.env` file should contain:

```env
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
OPENAI_API_KEY=your_openai_api_key
```

### 4. Test Connections (Optional)

```bash
python test_connection.py
```

## Usage

### Step 1: Ingest PDF into Qdrant

Process and store the PDF content with OpenAI embeddings:

```bash
python ingest_pdf.py
```

This will:
- Extract text from the PDF (page by page)
- Split into manageable chunks
- Generate embeddings using OpenAI
- Store everything in Qdrant

### Step 2: Start Q&A Session

Launch the interactive Q&A system:

```bash
python qa_app.py
```

### Example Questions

- "What is Newton's first law?"
- "Explain the concept of momentum"
- "What is the difference between speed and velocity?"
- "Define kinetic energy"
- "What is the law of conservation of energy?"

## How It Works

1. **Question Input**: You ask a physics question
2. **Embedding**: Your question is converted to a vector using OpenAI
3. **Retrieval**: Top 3 most relevant text chunks are retrieved from Qdrant
4. **Context Building**: Retrieved chunks are formatted with page references
5. **LLM Generation**: GPT-4 generates a clear answer based on the context
6. **Answer Display**: You receive an accurate, textbook-based answer

## Project Structure

```
.
├── ingest_pdf.py           # PDF ingestion script
├── qa_app.py               # Interactive Q&A application
├── test_connection.py      # Connection test script
├── connet_qdrant.py        # Original Qdrant test
├── .env                    # Environment variables (not in git)
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── NCERT-Class-11-Physics-Part-1.pdf
```

## Security Notes

- Never commit `.env` file to version control
- API keys are loaded from environment variables only
- No hardcoded credentials in code

## Technologies

- **Qdrant**: Vector database for semantic search
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
- **PyMuPDF**: PDF text extraction
- **Python 3.12**: Modern Python features
