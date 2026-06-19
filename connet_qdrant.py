from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://6e8b2b56-7e71-4cb3-8a1d-fd6149e731c3.eu-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzhlNmE0MDAtZTkxOC00MTFhLTlhMjQtMWNmOWYxODVhYTkzIn0.hFbR62CvPdrm79UKvfgZCB-QnLpBgMYs7SH9i8xCccE",
)

print(qdrant_client.get_collections())