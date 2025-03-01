import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, Dict

from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional
import google.generativeai as genai

class GeminiLLM(LLM):
    # Define class variables for Pydantic
    google_api_key: str
    model_name: str = "gemini-1.5-pro-001"
    
    # No custom _init_ method - let Pydantic handle initialization
    
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
    llm = GeminiLLM(google_api_key=google_api_key)
    
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

if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    faiss_index_path = os.environ.get("FAISS_INDEX_PATH", "indian_penal_codes.index")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyBoGabhtqv5-DAPGV37jw6RAsvwadu6AkA")
    
    # Example question
    question = "I recently started a tech startup and received a call from someone claiming to be from the 'National Startup Registration Office.' They said that, under a new 'Startup Act,' I must pay a ₹25,000 registration fee within 7 days to avoid penalties and possible business suspension. Is there any legal requirement for such a fee, or is this a scam? What should I do in this situation?"
    
    # Debug logs
    print(f"Using FAISS index path: {faiss_index_path}")
    print(f"API Key configured: {'Yes' if google_api_key != 'your_api_key_here' else 'No'}")
    
    try:
        response = query_system(question, faiss_index_path, google_api_key)
        answer = response["answer"]
        print("Answer:")
        print(answer)
        print("\nSources:")
        for idx, doc in enumerate(response["sources"], start=1):
            print(f"Source {idx}:")
            # Print the first 300 characters of the source content for brevity
            print(doc.page_content[:300] + "...")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print("---")
    except Exception as e:
        print("Error processing query:")
        print(e)
