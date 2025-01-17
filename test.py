import tempfile
import os
from pathlib import Path
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
import gradio as gr

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf(file_path):
    """Read PDF and extract text"""
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    """Clean the extracted text"""
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_text(text, chunk_size=1000):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space

        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_and_embed_document(file_path, chunk_size=1000):
    """Process document from PDF to embeddings"""
    # Read and process text
    raw_text = read_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    text_chunks = split_text(cleaned_text, chunk_size)

    # Generate embeddings
    embeddings = []
    for chunk in tqdm(text_chunks):
        embedding = embedding_model.encode(chunk)
        embeddings.append(embedding)

    return text_chunks, embeddings

def create_collection(text_chunks, embeddings, collection_name="document_collection", path="./chroma"):
    """Create and populate Chroma collection"""
    chroma_client = chromadb.PersistentClient(path=path)

    # Remove existing collection if it exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except ValueError:
        pass

    # Create new collection
    collection = chroma_client.create_collection(name=collection_name)

    # Add documents and embeddings
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(text_chunks))]
    )

    return collection

def retrieve_documents(query, collection, top_k=5):
    """Retrieve relevant documents using query"""
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query)

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

def generate_rag_response(query, collection, model="gemma2-9b-it", top_k=3):
    """
    Generate a response using RAG (Retrieval-Augmented Generation)
    Returns both the response and the context used
    """
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, collection, top_k)
    relevant_contexts = [doc for doc in retrieved_docs["documents"][0]]

    # Combine contexts for the prompt
    context_text = "\n".join(relevant_contexts)
    prompt = f"Answer the query based on the following context:\n\n{context_text}\n\nQuery: {query}\nResponse:"

    # Simulate RAG response (replace this with actual API call if available)
    response = f"Simulated response to query: '{query}' using context."

    return response, relevant_contexts

def handle_file_upload(file, query):
    """Handle file upload and query processing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.read())
        temp_file_path = tmp_file.name

    try:
        # Process document
        text_chunks, embeddings = process_and_embed_document(temp_file_path)

        # Create collection
        collection = create_collection(
            text_chunks, 
            embeddings,
            collection_name="temp_collection",
            path=str(Path.home() / ".chroma_temp")
        )

        # Generate RAG response
        response, contexts = generate_rag_response(query, collection, top_k=3)

        os.unlink(temp_file_path)  # Clean up temporary file

        return response, "\n\n".join(contexts)

    except Exception as e:
        os.unlink(temp_file_path)  # Clean up temporary file
        return f"Error: {str(e)}", ""

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# ChatME: PDF Question Answering")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF")
        query_input = gr.Textbox(label="Enter your question")

    response_output = gr.Textbox(label="Answer", lines=5)
    context_output = gr.Textbox(label="Relevant Context", lines=10)

    submit_button = gr.Button("Get Answer")

    submit_button.click(
        handle_file_upload, 
        inputs=[pdf_file, query_input], 
        outputs=[response_output, context_output]
    )

if __name__ == "__main__":
    demo.launch()
