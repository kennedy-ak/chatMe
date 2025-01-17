# import streamlit as st
# import tempfile
# import os
# from pathlib import Path
# import time
# import PyPDF2
# import re
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import chromadb



# # Import all our previous functions here
# # Assuming we have all the previous functions from earlier code
# from sentence_transformers import SentenceTransformer
# import chromadb
# from groq import Groq
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# def read_pdf(file_path):
#     """Read PDF and extract text"""
#     text = ""
#     with open(file_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# def clean_text(text):
#     """Clean the extracted text"""
#     text = re.sub(r"\n", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# def split_text(text, chunk_size=1000):
#     """Split text into chunks"""
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_size = 0
    
#     for word in words:
#         current_chunk.append(word)
#         current_size += len(word) + 1  # +1 for space
        
#         if current_size >= chunk_size:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#             current_size = 0
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# def process_and_embed_document(file_path, chunk_size=1000):
#     """Process document from PDF to embeddings"""
#     # Read and process text
#     raw_text = read_pdf(file_path)
#     cleaned_text = clean_text(raw_text)
#     text_chunks = split_text(cleaned_text, chunk_size)
    
#     # Generate embeddings
#     print("Generating embeddings...")
#     embeddings = []
#     for chunk in tqdm(text_chunks):
#         embedding = embedding_model.encode(chunk)
#         embeddings.append(embedding)
    
#     return text_chunks, embeddings

# def create_collection(text_chunks, embeddings, collection_name="document_collection", path="./chroma"):
#     """Create and populate Chroma collection"""
#     chroma_client = chromadb.PersistentClient(path=path)
    
#     # Remove existing collection if it exists
#     try:
#         chroma_client.delete_collection(name=collection_name)
#     except ValueError:
#         pass
    
#     # Create new collection
#     collection = chroma_client.create_collection(name=collection_name)
    
#     # Add documents and embeddings
#     collection.add(
#         documents=text_chunks,
#         embeddings=embeddings,
#         ids=[f"doc_{i}" for i in range(len(text_chunks))]
#     )
    
#     return collection

# def retrieve_documents(query, collection, top_k=5):
#     """Retrieve relevant documents using query"""
#     # Generate embedding for the query
#     query_embedding = embedding_model.encode(query)
    
#     # Query the collection
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )
    
#     return results

# # Step 4: Generate Response with RAG
# def generate_rag_response(query, collection, model="gemma2-9b-it", top_k=3):
#     """
#     Generate a response using RAG (Retrieval-Augmented Generation)
#     Returns both the response and the context used
#     """
#     # Retrieve relevant documents
#     retrieved_docs = retrieve_documents(query, collection, top_k)
#     relevant_contexts = [doc for doc in retrieved_docs["documents"][0]]

#     # Combine contexts for the prompt
#     context_text = "\n".join(relevant_contexts)
#     prompt = f"Answer the query based on the following context:\n\n{context_text}\n\nQuery: {query}\nResponse:"

#     # Create Groq client and generate response
#     groq_client = Groq(
#         api_key="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"
#     )

#     chat_completion = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant that answers questions based on the provided context."
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         model=model,
#     )

#     # Return both the response and the context
#     return {
#         'response': chat_completion.choices[0].message.content,
#         'context': relevant_contexts
#     }
   
# def main():
#     st.title(" ChatME")
#     st.write("Upload a PDF document and ask questions ")

#     # Initialize session state variables
#     if 'collection' not in st.session_state:
#         st.session_state.collection = None
#     if 'text_chunks' not in st.session_state:
#         st.session_state.text_chunks = None
#     if 'embeddings' not in st.session_state:
#         st.session_state.embeddings = None

#     # Sidebar for API key
  
#     api_key = "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"
        
       
#     # File upload
#     uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

#     if uploaded_file:
#         with st.spinner("Processing document..."):
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 temp_file_path = tmp_file.name

#             # Process document if not already processed
#             if st.session_state.collection is None:
#                 try:
#                     # Process document
#                     text_chunks, embeddings = process_and_embed_document(temp_file_path)
                    
#                     # Create collection
#                     collection = create_collection(
#                         text_chunks, 
#                         embeddings,
#                         collection_name="temp_collection",
#                         path=str(Path.home() / ".chroma_temp")
#                     )
                    
#                     # Store in session state
#                     st.session_state.collection = collection
#                     st.session_state.text_chunks = text_chunks
                    
#                     st.success("Document processed successfully!")
                    
#                 except Exception as e:
#                     st.error(f"Error processing document: {str(e)}")
#                 finally:
#                     # Clean up temporary file
#                     os.unlink(temp_file_path)

#         # Query interface
#         st.markdown("---")
#         st.header("Ask Questions")
        
#         query = st.text_input("Enter your question about the document:")
#         top_k = 3
        
#         if query and st.session_state.collection and api_key:
#             if st.button("Get Answer"):
#                 with st.spinner("Generating response..."):
#                     try:
#                         # Generate RAG response
#                         result = generate_rag_response(
#                             query=query,
#                             collection=st.session_state.collection,
#                             top_k=top_k,
                            
#                         )
                        
#                         # Display response
#                         st.markdown("### Answer:")
#                         st.write(result['response'])
                        
#                         # Display retrieved contexts
#                         with st.expander("View Related Information"):
#                             for i, doc in enumerate(result['context'], 1):
#                                 st.markdown(f"Relevant Info {i}:")
#                                 st.write(doc)
#                                 st.markdown("---")
                                
#                     except Exception as e:
#                         st.error(f"Error generating response: {str(e)}")
        
#         elif not api_key:
#             st.warning("Please enter your Groq API key in the sidebar.")
            
#     else:
#         st.info("Please upload a PDF document to begin.")

    
    

# if __name__ == "__main__":
#     main()



import streamlit as st
import tempfile
import os
from pathlib import Path
import time
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import faiss
import pickle

# Initialize the embedding model
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

class FAISSDocumentStore:
    def __init__(self):
        self.dimension = 384  # dimension of all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        self.documents.extend(documents)
        self.index.add(embeddings)
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5):
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the corresponding documents
        retrieved_docs = [self.documents[i] for i in indices[0]]
        
        return {
            "documents": [retrieved_docs],
            "distances": distances
        }

def process_and_embed_document(file_path, chunk_size=1000):
    """Process document from PDF to embeddings"""
    # Read and process text
    raw_text = read_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    text_chunks = split_text(cleaned_text, chunk_size)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for chunk in tqdm(text_chunks):
        embedding = embedding_model.encode(chunk)
        embeddings.append(embedding)
    
    return text_chunks, np.array(embeddings)

def create_document_store(text_chunks, embeddings):
    """Create and populate FAISS document store"""
    doc_store = FAISSDocumentStore()
    doc_store.add_documents(text_chunks, embeddings)
    return doc_store

def retrieve_documents(query, doc_store, top_k=5):
    """Retrieve relevant documents using query"""
    query_embedding = embedding_model.encode(query)
    results = doc_store.query(query_embedding, top_k)
    return results

def generate_rag_response(query, doc_store, model="gemma2-9b-it", top_k=3):
    from groq import Groq
    """Generate a response using RAG"""
    retrieved_docs = retrieve_documents(query, doc_store, top_k)
    relevant_contexts = [doc for doc in retrieved_docs["documents"][0]]
    
    context_text = "\n".join(relevant_contexts)
    prompt = f"Answer the query based on the following context:\n\n{context_text}\n\nQuery: {query}\nResponse:"

    groq_client = Groq(
        api_key="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"
    )

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    return {
        'response': chat_completion.choices[0].message.content,
        'context': relevant_contexts
    }

def main():
    st.title("ChatME")
    st.write("Upload a PDF document and ask questions")

    # Initialize session state variables
    if 'doc_store' not in st.session_state:
        st.session_state.doc_store = None

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            # Process document if not already processed
            if st.session_state.doc_store is None:
                try:
                    # Process document
                    text_chunks, embeddings = process_and_embed_document(temp_file_path)
                    
                    # Create document store
                    doc_store = create_document_store(text_chunks, embeddings)
                    
                    # Store in session state
                    st.session_state.doc_store = doc_store
                    
                    st.success("Document processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)

        # Query interface
        st.markdown("---")
        st.header("Ask Questions")
        
        query = st.text_input("Enter your question about the document:")
        top_k = 3
        
        if query and st.session_state.doc_store:
            if st.button("Get Answer"):
                with st.spinner("Generating response..."):
                    try:
                        # Generate RAG response
                        result = generate_rag_response(
                            query=query,
                            doc_store=st.session_state.doc_store,
                            top_k=top_k
                        )
                        
                        # Display response
                        st.markdown("### Answer:")
                        st.write(result['response'])
                        
                        # Display retrieved contexts
                        with st.expander("View Related Information"):
                            for i, doc in enumerate(result['context'], 1):
                                st.markdown(f"Relevant Info {i}:")
                                st.write(doc)
                                st.markdown("---")
                                
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    else:
        st.info("Please upload a PDF document to begin.")

if __name__ == "__main__":
    main()