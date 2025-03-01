import os
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_txt(file_path):
    """Extract text from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def tokenize_text(text):
    """Tokenize text into sentences."""
    sentences = sent_tokenize(text)
    return sentences

def chunk_text(sentences, chunk_size=5, overlap=1):
    """
    Group sentences into overlapping chunks.
    
    Args:
        sentences (list): List of sentences.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of sentences to overlap between consecutive chunks.
        
    Returns:
        list: List of text chunks.
    """
    chunks = []
    step = chunk_size - overlap  # determine the sliding step
    for i in range(0, len(sentences), step):
        # Create a chunk by joining the next chunk_size sentences
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def main():
    # Replace with your text file path
    file_path = "./IndianPenalCodeBook.txt"  

    # Step 1: Extract text from the text file
    print("Extracting text from file...")
    text = extract_text_from_txt(file_path)
    print("Extraction complete.")

    # Step 2: Tokenize the extracted text into sentences
    print("Tokenizing text into sentences...")
    sentences = tokenize_text(text)
    print("Total sentences extracted:", len(sentences))

    # Step 3: Group sentences into overlapping chunks
    print("Creating text chunks...")
    chunks = chunk_text(sentences, chunk_size=5, overlap=1)
    print("Total chunks created:", len(chunks))

    # Step 4: Create a vectorstore using LangChain's FAISS wrapper
    print("Generating vectorstore from text chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Optionally, add metadata for each chunk (here, we store the chunk index)
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    vectorstore = FAISS.from_texts(chunks, embedding_model, metadatas=metadatas)
    print("Vectorstore created with {} vectors.".format(vectorstore.index.ntotal))

    # Step 5: Save the vectorstore in the expected folder structure
    save_path = "rag/indian_penal_codes.index"
    vectorstore.save_local(save_path)
    print("Vectorstore saved to '{}'.".format(save_path))

    # Step 6: (Optional) Query the vectorstore to test retrieval
    query = input("Enter your query: ")
    results = vectorstore.similarity_search(query, k=5)
    print("\nTop results for the query:")
    for result in results:
        print("Text:", result.page_content)
        print("Metadata:", result.metadata)
        print("------")

if __name__ == '__main__':
    main()
