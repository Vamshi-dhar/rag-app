"""
RAG Q&A Application - Interactive question answering from NCERT Physics PDF
"""
from openai import OpenAI
from qdrant_client import QdrantClient
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

COLLECTION_NAME = "ncert_physics_class11"

print("✓ Loaded configuration from environment variables")

def get_embedding(text):
    """Generate embedding using OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def expand_query(question):
    """Expand the query for better retrieval using LLM"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a search query expander. Generate additional keywords and phrases that would help find relevant physics content."},
                {"role": "user", "content": f"Original query: {question}\n\nGenerate 3-5 related keywords or short phrases that would help retrieve relevant content about this topic from a physics textbook. Return only the keywords separated by commas."}
            ],
            temperature=0.3,
            max_tokens=50
        )
        keywords = response.choices[0].message.content.strip()
        expanded = f"{question} {keywords}"
        return expanded
    except:
        return question

def retrieve_relevant_chunks(question, top_k=5):
    """Retrieve most relevant chunks from Qdrant with query expansion"""
    # Expand the query for better retrieval
    expanded_query = expand_query(question)
    print(f"   Expanded query: {expanded_query[:100]}...")
    
    # Generate embedding for the expanded question using OpenAI
    question_embedding = get_embedding(expanded_query)
    
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

def generate_answer_with_llm(question, context_chunks):
    """Generate answer using OpenAI with RAG context"""
    # Prepare context from retrieved chunks with scores
    context_parts = []
    for idx, chunk in enumerate(context_chunks, 1):
        if hasattr(chunk, 'payload'):
            page = chunk.payload.get('page_number', 'Unknown')
            text = chunk.payload.get('text', '')
            score = getattr(chunk, 'score', 0)
        else:
            page = chunk.get('page_number', 'Unknown')
            text = chunk.get('text', '')
            score = chunk.get('score', 0)
        
        context_parts.append(f"[Passage {idx} - Page {page}, Relevance: {score:.2f}]:\n{text}")
    
    context_text = "\n\n".join(context_parts)
    
    # Create prompt for GPT
    prompt = f"""You are a helpful physics tutor. Based on the following excerpts from NCERT Class 11 Physics textbook, answer the student's question clearly and comprehensively.

Context from Textbook (multiple passages):
{context_text}

Student's Question: {question}

Instructions:
- Provide a complete, accurate answer based on the context
- If the context contains definitions, laws, or formulas, include them clearly
- Explain concepts in simple, educational terms
- Use bullet points or numbered lists for clarity when appropriate
- If multiple passages discuss the topic, synthesize the information
- If the context doesn't fully answer the question, mention what information is available and what might be missing
- Always cite page numbers when referencing specific information

Answer:"""
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a knowledgeable and helpful physics tutor specializing in NCERT Class 11 Physics. You provide clear, comprehensive answers based on textbook content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )
    
    return response.choices[0].message.content

def answer_question(question):
    """Main function to answer a question using RAG"""
    print(f"\n🔍 Processing question...")
    
    # Retrieve relevant chunks from vector database (increased to 5 for better coverage)
    relevant_chunks = retrieve_relevant_chunks(question, top_k=5)
    
    if not relevant_chunks:
        print("❌ Sorry, I couldn't find any relevant information in the textbook.")
        return
    
    print(f"✓ Retrieved {len(relevant_chunks)} relevant passages")
    print(f"✓ Retrieved chunks {relevant_chunks}")
    
    # Show which pages were referenced with relevance scores
    pages_scores = []
    for chunk in relevant_chunks:
        if hasattr(chunk, 'payload'):
            page = chunk.payload.get('page_number')
            score = getattr(chunk, 'score', 0)
        else:
            page = chunk.get('page_number')
            score = chunk.get('score', 0)
        pages_scores.append((page, score))
    
    print(f"📖 Top References: ", end="")
    for page, score in pages_scores[:3]:  # Show top 3
        print(f"Page {page} ({score:.2f}), ", end="")
    print("\n")
    
    print("🤔 Generating comprehensive answer...\n")
    
    # Generate answer using LLM with retrieved context
    answer = generate_answer_with_llm(question, relevant_chunks)
    
    print("📝 Answer:")
    print("="*80)
    print(answer)
    print("="*80)

def interactive_qa():
    """Run interactive Q&A session"""
    print("\n" + "="*80)
    print("🎓 NCERT Class 11 Physics Q&A System (Powered by RAG + OpenAI)")
    print("="*80)
    print("\nAsk questions about physics topics from your textbook.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            question = input("\n❓ Your Question: ").strip()
            
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
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✓ Connected to collection '{COLLECTION_NAME}'")
        print(f"✓ Total vectors: {collection_info.points_count}\n")
        interactive_qa()
    except Exception as e:
        print(f"\n❌ Collection '{COLLECTION_NAME}' not found!")
        print(f"Error: {e}")
        print("\nPlease run 'python ingest_pdf.py' first to ingest the PDF.\n")
