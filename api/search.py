
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)

documents = ["your preloaded documents here"]  # Replace with your documents list

def handler(request):
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        query_embedding = embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)  # Normalize query embedding

        # Search FAISS index
        k = 3
        distances, indices = index.search(query_embedding, k)

        # Retrieve most relevant documents
        results = [documents[i] for i in indices[0]]
        return {
            "statusCode": 200,
            "body": json.dumps({"results": results})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
