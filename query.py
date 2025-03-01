# policy_query.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PolicyQueryEngine:
    def __init__(self, index_name="government_policies_index"):
        """
        Initialize the query engine with a FAISS index.
        """
        self.index_name = index_name
        self.retriever = self._load_retriever()
        self.rag_chain = self._create_rag_chain()
    
    def _load_retriever(self):
        """
        Load the FAISS index and create a retriever.
        """
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        try:
            db = FAISS.load_local(self.index_name, embedding_model, allow_dangerous_deserialization=True)
            
            # Create a retriever
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most similar chunks
            )
            
            return retriever
        except Exception as e:
            raise Exception(f"Failed to load FAISS index: {str(e)}")
    
    def _create_rag_chain(self):
        """
        Create a RAG chain with the retriever and Gemini model.
        """
        # Check if Google API key is set
        if "GOOGLE_API_KEY" not in os.environ:
            raise Exception("GOOGLE_API_KEY environment variable is not set")
        
        # Initialize the Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.2,
            top_p=0.95,
        )
        
        # Create a template for contextual question answering
        template = """
You are a government policy expert assistant who provides actionable advice and in-depth analysis on government policies and solutions for everyday problems related to them. Your responses must always be based on the provided context, and if the context is insufficient, clearly indicate which parts cannot be answered while offering general guidance based on common government policy principles. Do not invent or assume details that are not present in the context.

When answering questions:
1. Prioritize and directly reference the provided context, citing specific policy provisions, regulatory frameworks, and official guidelines where applicable.
2. If the context contains sufficient information, provide a detailed, point-wise analysis that includes:
   - A brief summary of the key policy provisions or principles addressed in the context.
   - An explanation of any relevant government frameworks, including clear, plain language descriptions of policy or regulatory terminology.
   - An overview of the potential implications and considerations for stakeholders.
   - Practical, actionable recommendations or courses of action based on established government policy principles and frameworks.
   - A summary list of any relevant policies, regulations, or official guidelines referenced.

If the context does not fully address the question:
1. Clearly state which parts of the question cannot be answered due to insufficient context.
2. Offer general guidance and advice based on commonly accepted government policy principles.
3. Suggest further avenues for research or identify types of government resources and official channels that might provide additional insights.
4. Outline general procedural steps that may apply in similar situations.

Always include a disclaimer that while your responses are informed by expert analysis of government policies:
1. They should not replace consultation with a qualified professional or an official government representative.
2. They may not account for the most recent policy changes or jurisdiction-specific variations.
3. They are provided for informational purposes only to help understand policy implications and options.

Context:
{context}

Question:
{question}

Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question):
        """
        Process a question and return the answer.
        """
        try:
            response = self.rag_chain.invoke(question)
            return {"answer": response, "status": "success"}
        except Exception as e:
            return {"answer": f"Error processing your question: {str(e)}", "status": "error"}

# For command line testing
if __name__ == "__main__":
    engine = PolicyQueryEngine()
    
    print("Government Policy Query Engine is ready! Type 'exit' to quit.")
    
    while True:
        query = input("\nAsk a question about government policies: ")
        
        if query.lower() == "exit":
            print("Thank you for using the Government Policy Query Engine!")
            break
        
        result = engine.query(query)
        print("\nResponse:", result["answer"])