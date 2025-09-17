import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        if _HAS_ST:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        else:
            self.model = None
        self._tfidf = None

    def encode(self, texts):
        if self.model is not None:
            emb = self.model.encode(texts, normalize_embeddings=True)
            return emb
        # fallback: TF-IDF dense vectors (not normalized)
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=512)
            self._tfidf.fit(texts)
        vecs = self._tfidf.transform(texts).toarray()
        # normalize
        norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
        norms[norms==0] = 1.0
        return vecs / norms
