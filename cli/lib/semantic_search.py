from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re

from torch import cosine_similarity
from lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT


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
        
    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    words = text.split()
    chunks = []
    n_words = len(words)
    i = 0

    while i < n_words - overlap:
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    n_sentences = len(sentences)
    i = 0

    while i < n_sentences - overlap:
        chunk_sentences = sentences[i:i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")