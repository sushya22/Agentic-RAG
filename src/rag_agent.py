import os, glob
import numpy as np
from embeddings import Embedder
from sklearn.metrics.pairwise import cosine_similarity
from tools import Tools
from typing import List, Tuple

class DocumentStore:
    def __init__(self, docs, metadatas=None, embeddings=None):
        self.docs = docs
        self.metadatas = metadatas or [{} for _ in docs]
        self._embeddings = embeddings

    @classmethod
    def from_folder(cls, folder_path):
        files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
        docs = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fh:
                docs.append(fh.read())
        embedder = Embedder()
        embs = embedder.encode(docs)
        return cls(docs, embeddings=embs)

    def retrieve(self, query, k=3):
        if self._embeddings is None or len(self._embeddings)==0:
            return []
        embedder = Embedder()
        q_emb = embedder.encode([query])[0:1]
        sims = cosine_similarity(q_emb, self._embeddings)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(self.docs[i], float(sims[i]), self.metadatas[i]) for i in idx]

class RAGAgent:
    def __init__(self, store: DocumentStore):
        self.store = store
        self.tools = Tools()

    def plan(self, query: str) -> dict:
        # Very simple planner:
        # if query contains math trigger -> call calculator tool
        triggers = []
        if any(tok in query.lower() for tok in ['calculate', 'sum', 'add', '+', '-', '*', '/']):
            triggers.append('calculator')
        if 'file' in query.lower() or 'document' in query.lower():
            triggers.append('file_search')
        return {'query': query, 'triggers': triggers}

    def generate_answer(self, query: str, contexts: List[Tuple[str, float]]):
        # Simple generator: stitch top-k contexts and return
        header = "Answer (RAG assisted):\n"
        if contexts:
            header += "\nRelevant documents:\n"
            for i, (doc, score, meta) in enumerate(contexts):
                header += f"--- Doc {i+1} (score={score:.3f}) ---\n"
                header += doc.strip()[:800] + "\n\n"
        header += "Final response:\n"
        header += f"I processed your query: '{query}'. Using retrieved contexts above, here's a concise answer: "
        # naive summarization: include first sentence of top doc if exists
        if contexts:
            top_doc = contexts[0][0]
            first_sentence = top_doc.split('.')[:1][0]
            header += first_sentence + '.'
        else:
            header += "I couldn't find supporting documents; try rephrasing."
        return header

    def answer(self, query: str, max_steps=3):
        plan = self.plan(query)
        responses = []
        # If any tool triggers, call tools first
        for trig in plan['triggers']:
            if trig == 'calculator':
                calc_res = self.tools.calculator(query)
                responses.append(f"[tool:calculator] {calc_res}")
            if trig == 'file_search':
                fs = self.tools.file_search(self.store.docs, query)
                responses.append(f"[tool:file_search] found {len(fs)} documents matching\n" + '\n'.join(fs[:3]))
        # Retrieval step
        contexts = self.store.retrieve(query, k=3)
        gen = self.generate_answer(query, contexts)
        return '\n'.join(responses + [gen])
