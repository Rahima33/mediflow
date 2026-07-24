import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "rag/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded once, at import time, so repeated calls don't reload the model
# or reopen the store from disk each time.
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(
        f"Knowledge base not found at {PERSIST_DIR}/. "
        f"Run `uv run python -m rag.build_knowledge_base` first."
    )

_vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=_embeddings,
)


def retrieve_guidelines(query, k=4):
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        query: a natural-language query, e.g. built from the case's
               prediction ("pediatric pneumonia diagnosis chest x-ray
               findings" or "normal chest x-ray pediatric assessment")
        k: how many chunks to retrieve

    Returns:
        list of dicts, each with:
            "text": the chunk's text content
            "source": originating PDF filename
            "page": page number within that PDF
    """
    results = _vectorstore.similarity_search(query, k=k)

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
        }
        for doc in results
    ]


if __name__ == "__main__":
    # Quick manual test -- try a couple of queries and eyeball the results
    test_queries = [
        "pediatric pneumonia diagnosis chest x-ray findings",
        "community-acquired pneumonia management children",
        "normal chest x-ray assessment",
    ]

    for q in test_queries:
        print(f"\n{'='*70}\nQuery: {q}\n{'='*70}")
        results = retrieve_guidelines(q, k=3)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (source: {r['source']}, page {r['page']}) ---")
            print(r["text"][:400], "...")