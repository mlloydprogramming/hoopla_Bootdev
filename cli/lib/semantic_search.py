from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Input text cannot be empty.")
        
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list[dict]):
        if not documents:
            raise ValueError("Document list cannot be empty.")
        
        self.documents = documents
        movie_strings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            string_rep = f"{doc['title']}: {doc['description']}"
            movie_strings.append(string_rep)

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        with open("cache/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        if os.path.exists("cache/embeddings.npy"):
            with open("cache/embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)



def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text: str) -> list[float]:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    search_instance = SemanticSearch()
    query_embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embedding[:5]}")
    print(f"Shape: {query_embedding.shape}")