import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import mysql.connector  # MySQL connection
from bs4 import BeautifulSoup  # For cleaning HTML content

app = Flask(__name__)
CORS(app)

# Load the sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)

# Initialize the question-answering pipeline with a dedicated model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# List to store documents retrieved from the database
documents = []

# Function to clean the HTML content
def clean_html_content(html_content):
    """Clean HTML content and return only the readable text."""
    soup = BeautifulSoup(html_content, "html.parser")
    clean_content = soup.get_text()  # Extracts text without HTML tags and comments
    return clean_content.strip()  # Strip any leading/trailing whitespace

# Function to fetch posts (content) from WordPress database (wp_posts table)
def fetch_documents_from_db():
    connection = mysql.connector.connect(
        host="localhost",   
        user="root",        
        password="",        
        database="wordpress_db"  # The WordPress database
    )
    
    cursor = connection.cursor()
    
    # Query to fetch post content from wp_posts table
    cursor.execute("SELECT post_content FROM wp_posts WHERE post_status = 'publish'")  # Only published posts
    rows = cursor.fetchall()
    
    # Clear existing documents and populate with new ones from DB
    documents.clear()
    for row in rows:
        post_content = row[0]
        clean_content = clean_html_content(post_content)  # Clean the HTML content
        documents.append(clean_content)  # Append the cleaned content
    
    cursor.close()
    connection.close()

    # Add embeddings for the posts to FAISS index
    embeddings = embedding_model.encode(documents).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize embeddings
    index.add(embeddings)  # Add the embeddings to the FAISS index

    print(f"Added {len(documents)} documents to FAISS index.")

# Call fetch_documents_from_db to initialize FAISS index at the start
fetch_documents_from_db()

# Route to generate embeddings from the input text
@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    data = request.json
    text = data.get('text', '')
    embedding = embedding_model.encode([text]).astype('float32')
    faiss.normalize_L2(embedding)  # Normalize query embedding
    return jsonify(embeddings=embedding.tolist())

# Route for searching relevant documents from the FAISS index
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)  # Normalize query embedding

    print(f"Query: {query}")
    print(f"Query embedding: {query_embedding}")

    # Search FAISS index for the top 3 most similar documents
    k = 3
    distances, indices = index.search(query_embedding, k)

    # Debug: Print the search results
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Get the most relevant documents (post_content)
    results = [documents[i] for i in indices[0]]
    
    # Post-process results to return more meaningful excerpts
    excerpts = []
    for result in results:
        if result:
            # Clean the result again to avoid any leftover HTML tags
            cleaned_result = clean_html_content(result)

            # Ensure the query is matched at any point, not just the beginning
            start_idx = cleaned_result.lower().find(query.lower())
            if start_idx != -1:
                # Extract a snippet around the query match
                snippet_start = max(0, start_idx - 100)
                snippet_end = min(len(cleaned_result), start_idx + len(query) + 150)  # Add some more context
                snippet = cleaned_result[snippet_start:snippet_end]
                cleaned_snippet = " ".join(snippet.split())  # Clean up unnecessary whitespaces
                excerpts.append(cleaned_snippet)
            else:
                # If query not found, return the first 300 characters
                cleaned_snippet = " ".join(cleaned_result[:300].split())
                excerpts.append(cleaned_snippet)
        else:
            excerpts.append("No relevant content found. Please try with a different query.")
    
    return jsonify(results=excerpts)

# Route to generate an answer using the context from relevant documents
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    data = request.json
    context = data.get('context', '')
    question = data.get('question', '')

    # Use the question-answering pipeline to generate an answer
    result = qa_pipeline(question=question, context=context)
    return jsonify(answer=result['answer'])

# Route to update the FAISS index with new document embeddings
@app.route('/update_index', methods=['POST'])
def update_index():
    data = request.json
    new_document = data.get('document', '')
    
    # Add the new document to the documents list
    documents.append(new_document)

    # Generate the embedding for the new document
    new_embedding = embedding_model.encode([new_document]).astype('float32')
    faiss.normalize_L2(new_embedding)  # Normalize new document embedding

    # Add the new document's embedding to the FAISS index
    index.add(new_embedding)
    
    return jsonify(status="Document added to index")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
