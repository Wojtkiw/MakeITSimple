import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_embedding, k=3):
        D, I = self.index.search(query_embedding, k)
        return [self.metadata[idx] for idx in I[0] if idx < len(self.metadata)]

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_index(text):
    chunks = chunk_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    metadata = [{"text": chunk} for chunk in chunks]
    embeddings_matrix = np.array([embeddings.embed_query(c) for c in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)
    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    return faiss_index.similarity_search(query_embedding, k)
