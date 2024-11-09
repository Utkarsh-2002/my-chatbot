import json
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def handler(request):
    try:
        data = json.loads(request.body)
        context = data.get('context', '')
        question = data.get('question', '')

        # Generate answer using the QA pipeline
        result = qa_pipeline(question=question, context=context)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": result['answer']})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
