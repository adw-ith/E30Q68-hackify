import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, Dict

app = Flask(__name__)

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-pro-001"
    
    def __init__(self, google_api_key):
        super().__init__()
        self.google_api_key = google_api_key
        genai.configure(api_key=self.google_api_key)
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        gemini_model = genai.GenerativeModel(self.model_name)
        response = gemini_model.generate_content(prompt)
        return response.text
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

def load_existing_faiss_index(faiss_index_path, embedding_model):
    """Load existing FAISS index"""
    return FAISS.load_local(faiss_index_path, embedding_model)

def query_system(question, faiss_index_path, api_key):
    """Query the system with a legal question"""
    # System prompt for legal context
    system_prompt = """
    You are a legal assistant specializing in Indian law, particularly the Indian Penal Code.
    Use the provided context to answer legal questions.
    Always specify:
    1. Relevant IPC sections
    2. Clear explanations of legal terminology
    3. When appropriate, mention that this is information only and not legal advice

    Base your answers only on the context provided. If the question cannot be answered based on the context, say so clearly.
    """
    
    # Initialize embedding model - must match what you used for creating the index
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load existing vector database
    vectordb = load_existing_faiss_index(faiss_index_path, embedding_model)
    
    # Initialize LLM
    llm = GeminiLLM(api_key)
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # Set up QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Build full prompt
    full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
    result = qa_chain({"query": full_prompt})
    
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }

@app.route('/query', methods=['POST'])
def query_endpoint():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400

        question = data['question']
        # Get configuration from environment variables or use defaults
        faiss_index_path = os.environ.get("FAISS_INDEX_PATH", "rag/rag/indian_penal_codes.index")
        google_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyBoGabhtqv5-DAPGV37jw6RAsvwadu6AkA")

        # Add debug log
        print(f"Using FAISS index path: {faiss_index_path}")
        print(f"API Key configured: {'Yes' if google_api_key != 'AIzaSyBoGabhtqv5-DAPGV37jw6RAsvwadu6AkA' else 'No'}")

        response = query_system(question, faiss_index_path, google_api_key)
        answer = response["answer"]

        # Format sources
        sources = []
        for doc in response["sources"]:
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "Unknown")
            })
        
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
