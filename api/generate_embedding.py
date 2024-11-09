import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)

def handler(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        # Generate embeddings
        embedding = embedding_model.encode([text]).astype('float32')
        faiss.normalize_L2(embedding)  # Normalize query embedding
        
        return {
            "statusCode": 200,
            "body": json.dumps({"embeddings": embedding.tolist()})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
