import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Download NLTK's Punkt tokenizer (if not already available)
nltk.download('punkt')

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
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for each text chunk using a Sentence Transformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

def build_faiss_index(embeddings):
    """Build a FAISS index from the embeddings."""
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance
    index.add(embeddings_np)
    return index

def query_index(query, model, index, chunks, k=5):
    """Encode a query and search the FAISS index for the top k similar chunks."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    print("\nTop {} results for the query: '{}'\n".format(k, query))
    for i, idx in enumerate(indices[0]):
        print("Result {}:".format(i + 1))
        print("Chunk: ", chunks[idx])
        print("Distance: ", distances[0][i])
        print("------")

def main():
    # Replace with your text file path
    file_path = "IndianPenalCodeBook.txt"  

    # Step 1: Extract text from the text file
    print("Extracting text from file...")
    text = extract_text_from_txt(file_path)
    print("Extraction complete.")

    # Step 2: Tokenize the extracted text into sentences
    print("Tokenizing text into sentences...")
    sentences = tokenize_text(text)
    print("Total sentences extracted: ", len(sentences))

    # Step 3: Group sentences into overlapping chunks
    print("Creating text chunks...")
    # Adjust chunk_size and overlap as needed (here, each chunk contains 5 sentences with 1 sentence overlap)
    chunks = chunk_text(sentences, chunk_size=5, overlap=1)
    print("Total chunks created: ", len(chunks))

    # Step 4: Generate embeddings for each chunk
    print("Generating embeddings for text chunks...")
    embeddings, model = create_embeddings(chunks)
    print("Embeddings generated.")

    # Step 5: Build a FAISS index from the embeddings
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("FAISS index built with {} vectors.".format(index.ntotal))

    # Optional: Save the FAISS index to disk for later use
    index_filename = "indian_penal_codes.index"
    faiss.write_index(index, index_filename)
    print("FAISS index saved to disk as '{}'.".format(index_filename))

    # Step 6: Query the index (example)
    query = input("Enter your query: ")
    query_index(query, model, index, chunks)

if __name__ == '__main__':
    main()
