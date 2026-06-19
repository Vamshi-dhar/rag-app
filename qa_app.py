"""
RAG Q&A Application - Interactive question answering from NCERT Physics PDF
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize models
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://6e8b2b56-7e71-4cb3-8a1d-fd6149e731c3.eu-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzhlNmE0MDAtZTkxOC00MTFhLTlhMjQtMWNmOWYxODVhYTkzIn0.hFbR62CvPdrm79UKvfgZCB-QnLpBgMYs7SH9i8xCccE",
)

COLLECTION_NAME = "ncert_physics_class11"

def retrieve_relevant_chunks(question, top_k=3):
    """Retrieve most relevant chunks from Qdrant"""
    # Generate embedding for the question
    question_embedding = embedding_model.encode(question).tolist()
    
    # Search in Qdrant - try different methods based on client version
    try:
        # Try newer API first
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_embedding,
            limit=top_k
        )
        return search_results.points if hasattr(search_results, 'points') else search_results
    except (AttributeError, TypeError):
        try:
            # Fallback to older search API
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_embedding,
                limit=top_k
            )
            return search_results
        except AttributeError:
            # Last resort - use scroll
            results, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=top_k
            )
            return results

def generate_answer_local(question, context_chunks):
    """Generate answer using local context (without LLM)"""
    if not context_chunks:
        return "No relevant information found."
    
    # Simple extraction-based answer
    context_parts = []
    for chunk in context_chunks:
        # Handle different response structures
        if hasattr(chunk, 'payload'):
            page = chunk.payload.get('page_number', 'Unknown')
            text = chunk.payload.get('text', '')
        else:
            page = chunk.get('page_number', 'Unknown')
            text = chunk.get('text', '')
        
        context_parts.append(f"[Page {page}]: {text}")
    
    return "\n\n".join(context_parts)

def generate_answer_with_llm(question, context_chunks):
    """Generate answer using OpenAI (optional - requires API key)"""
    try:
        # Check if OpenAI API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return None
        
        openai.api_key = openai_api_key
        
        # Prepare context - handle different structures
        context_parts = []
        for chunk in context_chunks:
            if hasattr(chunk, 'payload'):
                page = chunk.payload.get('page_number', 'Unknown')
                text = chunk.payload.get('text', '')
            else:
                page = chunk.get('page_number', 'Unknown')
                text = chunk.get('text', '')
            context_parts.append(f"[Page {page}]: {text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context from NCERT Class 11 Physics textbook, answer the question.

Context:
{context_text}

Question: {question}

Answer: Provide a clear and concise answer based only on the context provided. If the context doesn't contain enough information, say so."""
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful physics tutor answering questions based on NCERT Class 11 Physics textbook."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Note: OpenAI error - {e}")
        return None

def answer_question(question):
    """Main function to answer a question"""
    print(f"\n🔍 Searching for: {question}")
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, top_k=3)
    
    if not relevant_chunks:
        print("Sorry, I couldn't find any relevant information in the textbook.")
        return
    
    print(f"✓ Found {len(relevant_chunks)} relevant passages\n")
    
    # Try to generate answer with LLM first
    llm_answer = generate_answer_with_llm(question, relevant_chunks)
    
    if llm_answer:
        print("📝 Answer (Generated):")
        print(llm_answer)
    else:
        print("📝 Relevant Context from Textbook:")
        local_answer = generate_answer_local(question, relevant_chunks)
        print(local_answer)
    
    print("\n" + "="*80)

def interactive_qa():
    """Run interactive Q&A session"""
    print("\n" + "="*80)
    print("🎓 NCERT Class 11 Physics Q&A System")
    print("="*80)
    print("\nAsk questions about physics topics from your textbook.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            question = input("❓ Your Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for using the Q&A system!")
                break
            
            if not question:
                continue
            
            answer_question(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session ended.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    # Check if collection exists
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        interactive_qa()
    except Exception as e:
        print(f"\n❌ Collection '{COLLECTION_NAME}' not found!")
        print(f"Error: {e}")
        print("Please run 'python ingest_pdf.py' first to ingest the PDF.\n")
