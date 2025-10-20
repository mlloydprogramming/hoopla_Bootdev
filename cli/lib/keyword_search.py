import os
import pickle
import string
import math
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        count = len(tokens)
        self.doc_lengths[doc_id] = count
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_bm25_idf(self, term: str) -> float:
        tokenize_term = tokenize_text(term)
        if len(tokenize_term) != 1:
            raise ValueError("term must be a single token")
        total_docs = len(self.docmap)
        docs_with_term = len(self.get_documents(tokenize_term[0]))
        bm25_idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
        return bm25_idf
    
    def bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf
    
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        if len(self.doc_lengths) == 0:
            return 0.0
        avg_length = total_length / len(self.doc_lengths)
        return avg_length
    
    def bm25(self, doc_id: int, term: str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.bm25_tf(doc_id, term)
        return idf * tf
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]: #for each document in index, calculate the BM25 scores
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap.keys():
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            if score > 0:
                scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id].copy()
            doc['score'] = score
            results.append(doc)
        return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_text(term)
    if len(token) != 1:
        raise ValueError("term must be a single token")
    term = token[0]
    doc_count = len(idx.docmap)
    term_doc_count = len(idx.get_documents(term))
    idf_value = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf_value

def tf_idf_command(doc_id: int, term: str) -> float:
    tf = tf_command(doc_id, term)
    idf = idf_command(term)
    return tf * idf

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25_idf = idx.get_bm25_idf(term)
    return bm25_idf

def bm25_tf_command(doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.bm25_tf(doc_id, term, k1, b)
    return bm25_tf

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = idx.bm25_search(query, limit)
    return results
