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
    # Define class variables for Pydantic
    google_api_key: str
    model_name: str = "gemini-1.5-pro-001"
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        # Configure API on each call to ensure it's set
        genai.configure(api_key=self.google_api_key)
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
    """Load existing FAISS index with dangerous deserialization enabled."""
    return FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

def query_system(question, faiss_index_path, google_api_key):
    """Query the system with a legal question"""
    # System prompt for legal context
    system_prompt = """
You are an experienced legal assistant specializing in Indian law, particularly the Indian Penal Code.

When answering questions:
1. First prioritize information from the provided context, citing specific IPC sections and legal provisions
2. If the context contains sufficient information, provide a detailed analysis including:
   - Relevant IPC sections and their interpretations
   - Legal terminology explained in plain language
   - Potential legal implications and considerations
   - Possible courses of action based on established legal principles

If the context doesn't fully address the question:
1. Clearly state what part cannot be answered from the available context
2. Provide general legal guidance based on common principles of Indian law
3. Suggest potential avenues to explore or types of legal resources that might be helpful
4. Outline general procedural steps that might apply in similar situations

Always include language indicating that while your responses are informed by legal knowledge, they:
1. Should not replace consultation with a qualified legal professional
2. May not account for recent legal developments or jurisdiction-specific variations
3. Are provided for informational purposes to help understand legal concepts and options

Frame your advice in practical, action-oriented terms while acknowledging legal complexities.
"""
    
    # Initialize embedding model - must match what you used for creating the index
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load existing vector database
    vectordb = load_existing_faiss_index(faiss_index_path, embedding_model)
    
    # Initialize LLM
    llm = GeminiLLM(google_api_key=google_api_key)
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    
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
        "sources": [
            {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "page": doc.metadata.get('page', 'Unknown'),
                "source": doc.metadata.get('source', 'Unknown')
            } for doc in result["source_documents"]
        ]
    }

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get configuration from environment or request
        faiss_index_path = data.get('faiss_index_path') or os.environ.get("FAISS_INDEX_PATH", "indian_penal_codes.index")
        google_api_key = data.get('api_key') or os.environ.get("GOOGLE_API_KEY", "AIzaSyBoGabhtqv5-DAPGV37jw6RAsvwadu6AkA")
        
        # Execute query
        response = query_system(question, faiss_index_path, google_api_key)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Legal Assistant API is running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=5000, debug=os.environ.get("FLASK_DEBUG", "False") == "True")