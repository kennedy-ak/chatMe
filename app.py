


import streamlit as st
from pathlib import Path
import tempfile
import os
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from typing import List
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class FAISSDocumentStore:
    def __init__(self):
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        self.documents.extend(documents)
        self.index.add(embeddings)
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return {"documents": [retrieved_docs], "distances": distances}

def process_document(file_path, chunk_size=1000):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    embeddings = []
    for chunk in tqdm(chunks):
        embedding = embedding_model.encode(chunk)
        embeddings.append(embedding)
    
    return chunks, np.array(embeddings)

def get_response(query, doc_store, chat_history=[]):
    query_embedding = embedding_model.encode(query)
    results = doc_store.query(query_embedding, top_k=3)
    context = "\n".join(results["documents"][0])
    
    history_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" 
                             for m in chat_history[-3:]])
    
    prompt = f"""Previous conversation:
{history_text}

Context: {context}

User: {query}
Assistant:"""

    client = Groq(api_key=api_key )
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions about documents."},
            {"role": "user", "content": prompt}
        ],
        model="gemma2-9b-it",
        max_tokens=350
    )
    
    return completion.choices[0].message.content, results["documents"][0]

def main():
    st.title("Chat Me")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = None

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        if st.session_state.doc_store is None:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    chunks, embeddings = process_document(tmp_file.name)
                    os.unlink(tmp_file.name)
                
                doc_store = FAISSDocumentStore()
                doc_store.add_documents(chunks, embeddings)
                st.session_state.doc_store = doc_store
                st.success("Ready to chat!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask about your document"):
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        if st.session_state.doc_store:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, context = get_response(
                        query, 
                        st.session_state.doc_store,
                        [{"user": m["content"], "assistant": st.session_state.messages[i+1]["content"]}
                         for i, m in enumerate(st.session_state.messages[:-1:2])]
                    )
                    st.write(response)
                    with st.expander("View sources"):
                        for i, ctx in enumerate(context, 1):
                            st.markdown(f"**Source {i}:**\n{ctx}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
