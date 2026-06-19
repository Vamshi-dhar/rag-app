"""
Test Qdrant and OpenAI connections
"""
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing connections...\n")

# Test Qdrant connection
print("1. Testing Qdrant connection...")
try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collections = qdrant_client.get_collections()
    print(f"   ✓ Qdrant connected successfully!")
    print(f"   Current collections: {collections}")
except Exception as e:
    print(f"   ❌ Qdrant connection failed: {e}")

# Test OpenAI connection
print("\n2. Testing OpenAI connection...")
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test"
    )
    print(f"   ✓ OpenAI connected successfully!")
    print(f"   Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"   ❌ OpenAI connection failed: {e}")

print("\n✓ All tests complete!")
