"""
build_knowledge_base.py

One-time (re-run when documents change) script that:
    1. Loads every PDF in knowledge_base/
    2. Splits each into overlapping text chunks
    3. Embeds the chunks with a local embedding model (no API key needed)
    4. Persists everything to a Chroma vector store on disk

Run this once, whenever the source PDFs change:
    uv run python -m rag.build_knowledge_base

After this runs, rag/retriever.py loads the persisted store for actual
retrieval -- it does NOT re-process the PDFs on every query, since
embedding is comparatively slow and the source documents don't change
between runs.
"""

import os
import shutil

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

KNOWLEDGE_BASE_DIR = "knowledge_base"
PERSIST_DIR = "rag/chroma_db"

# A small, fast, well-regarded general-purpose embedding model. Runs
# locally on CPU -- no API key, no per-call cost. "all-MiniLM-L6-v2" is
# a common, well-tested default for this kind of retrieval task.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size is a tradeoff: too small and chunks lose context (a
# recommendation split away from its rationale); too large and
# retrieval becomes less precise (irrelevant surrounding text gets
# pulled in along with the relevant sentence). 800 characters with 150
# overlap is a reasonable starting point for guideline-style prose;
# worth revisiting once you see real retrieval results.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def build():
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing knowledge base at {PERSIST_DIR}/ for a clean rebuild...")
        shutil.rmtree(PERSIST_DIR)

    pdf_files = [
        f for f in os.listdir(KNOWLEDGE_BASE_DIR)
        if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF(s) in {KNOWLEDGE_BASE_DIR}/: {pdf_files}")

    all_documents = []
    for fname in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_DIR, fname)
        loader = PyPDFLoader(path)
        # PyPDFLoader returns one Document per PDF page, each already
        # tagged with metadata (source filename, page number) -- this
        # metadata survives chunking and is very useful later for
        # citing exactly where a retrieved chunk came from.
        pages = loader.load()
        print(f"  {fname}: {len(pages)} pages loaded")
        all_documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph/sentence boundaries before falling
        # back to arbitrary character cuts, so chunks stay readable.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks total")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run downloads it, ~90MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(chunks)} chunks and persisting to {PERSIST_DIR}/ ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()

    print(f"Done. Knowledge base ready at {PERSIST_DIR}/")


if __name__ == "__main__":
    build()