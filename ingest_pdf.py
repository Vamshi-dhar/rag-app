"""
PDF Ingestion Script - Extracts text from PDF and stores in Qdrant
"""
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://6e8b2b56-7e71-4cb3-8a1d-fd6149e731c3.eu-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzhlNmE0MDAtZTkxOC00MTFhLTlhMjQtMWNmOWYxODVhYTkzIn0.hFbR62CvPdrm79UKvfgZCB-QnLpBgMYs7SH9i8xCccE",
)

# Collection name
COLLECTION_NAME = "ncert_physics_class11"

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

def create_chunks(pages_content, chunk_size=500, overlap=50):
    """Split pages into smaller chunks for better retrieval"""
    chunks = []
    
    for page in pages_content:
        content = page['content']
        page_num = page['page_number']
        
        # Split content into chunks
        words = content.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'page_number': page_num,
                    'chunk_id': len(chunks)
                })
    
    print(f"Created {len(chunks)} chunks")
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
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    # Prepare points for batch upload
    points = []
    print("Generating embeddings...")
    
    for idx, chunk in enumerate(chunks):
        # Generate embedding
        embedding = model.encode(chunk['text']).tolist()
        
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
