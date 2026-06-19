"""
PDF Ingestion Script - Extracts text from PDF and stores in Qdrant
"""
import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Qdrant using environment variables
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Collection name
COLLECTION_NAME = "ncert_physics_class11"

def get_embedding(text):
    """Generate embedding using OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF page by page"""
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    pages_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  # Only add non-empty pages
            pages_content.append({
                'page_number': page_num + 1,
                'content': text.strip()
            })
    
    print(f"Extracted {len(pages_content)} pages")
    return pages_content

def create_chunks(pages_content, chunk_size=800, overlap=200):
    """Split pages into smaller chunks with better context preservation"""
    chunks = []
    
    for page in pages_content:
        content = page['content']
        page_num = page['page_number']
        
        # Split by sentences for better semantic chunking
        sentences = content.replace('\n', ' ').split('. ')
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'text': chunk_text,
                    'page_number': page_num,
                    'chunk_id': len(chunks)
                })
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                'text': chunk_text,
                'page_number': page_num,
                'chunk_id': len(chunks)
            })
    
    print(f"Created {len(chunks)} chunks with improved semantic boundaries")
    return chunks

def ingest_to_qdrant(chunks):
    """Store chunks in Qdrant with embeddings"""
    
    # Create collection if it doesn't exist
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting...")
        qdrant_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    print(f"Creating collection '{COLLECTION_NAME}'...")
    # OpenAI text-embedding-3-small has 1536 dimensions
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    
    # Prepare points for batch upload
    points = []
    print("Generating embeddings using OpenAI...")
    
    for idx, chunk in enumerate(chunks):
        # Generate embedding using OpenAI
        embedding = get_embedding(chunk['text'])
        
        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                'text': chunk['text'],
                'page_number': chunk['page_number'],
                'chunk_id': chunk['chunk_id']
            }
        )
        points.append(point)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(chunks)} chunks")
    
    # Upload to Qdrant
    print("Uploading to Qdrant...")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"Successfully ingested {len(points)} chunks to Qdrant!")

if __name__ == "__main__":
    # Extract text from PDF
    pages = extract_text_from_pdf("NCERT-Class-11-Physics-Part-1.pdf")
    
    # Create chunks
    chunks = create_chunks(pages)
    
    # Ingest to Qdrant
    ingest_to_qdrant(chunks)
    
    print("\n✓ PDF ingestion complete!")
