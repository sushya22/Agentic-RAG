from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import RAGAgent, DocumentStore

app = FastAPI(title="Agentic RAG Demo")

class Query(BaseModel):
    query: str

# Initialize store and agent (loads sample docs)
store = DocumentStore.from_folder("../data/sample_docs")
agent = RAGAgent(store)

@app.post("/ask")
async def ask(q: Query):
    response = agent.answer(q.query, max_steps=3)
    return {"query": q.query, "response": response}
