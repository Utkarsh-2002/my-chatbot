import json
from sentence_transformers import SentenceTransformer
import faiss

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)

documents = ["your preloaded documents here"]  # Replace with your documents list

def handler(request):
    try:
        data = json.loads(request.body)
        new_document = data.get('document', '')

        # Add the new document to the FAISS index
        documents.append(new_document)
        new_embedding = embedding_model.encode([new_document]).astype('float32')
        faiss.normalize_L2(new_embedding)  # Normalize new document embedding
        index.add(new_embedding)

        return {
            "statusCode": 200,
            "body": json.dumps({"status": "Document added to index"})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
