from src.rag_agent import DocumentStore, RAGAgent
import os

def test_retrieval_and_answer():
    here = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_docs')
    store = DocumentStore.from_folder(os.path.abspath(here))
    agent = RAGAgent(store)
    q = 'What is RAG and how does it use documents?'
    resp = agent.answer(q)
    assert 'Retrieval Augmented Generation' in resp or 'RAG' in resp

def test_calculator_tool():
    from src.tools import Tools
    t = Tools()
    r = t.calculator('Calculate 2+3*4')
    assert 'Result' in r and '14' in r
