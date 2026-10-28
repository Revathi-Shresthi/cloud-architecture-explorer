"""
Load cloud documents (PDF, Markdown, text) into a local Chroma vector database.
Stop the app (uvicorn) before you run this script: it replaces the collection.

Usage:
    python ingest.py data/
    python ingest.py data/ --max-pages 200   # quicker first test (per PDF)
"""
import argparse
from pathlib import Path

import chromadb
from pypdf import PdfReader

from components import tag_components
from embeddings import COLLECTION, EMBED_PROVIDER, embed

DB_PATH = "db"
CHUNK_WORDS = 220
OVERLAP_WORDS = 40
BATCH_SIZE = 32


def chunk_words(text: str):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 40:  # skip headers, page numbers, tiny leftovers
            continue
        yield " ".join(piece)


def read_pages(path: Path, max_pages):
    """Yield (page_number, text) for a PDF, or (1, text) for a .md / .txt file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        pages = list(reader.pages)
        if max_pages:
            pages = pages[:max_pages]
        for n, page in enumerate(pages, start=1):
            yield n, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with .pdf, .md or .txt files")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in {".pdf", ".md", ".txt"})
    if not files:
        print(f"No .pdf, .md or .txt files found in {args.folder}")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)  # start fresh, no duplicates
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    print(f"Embeddings: {EMBED_PROVIDER}, collection {COLLECTION}")
    ids, docs, metas = [], [], []
    total = 0

    def flush():
        nonlocal ids, docs, metas, total
        if not docs:
            return
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs, "document").tolist())
        total += len(docs)
        print(f"  stored {total} chunks")
        ids, docs, metas = [], [], []

    for f_idx, path in enumerate(files):
        print(f"Reading {path.name}")
        for page_num, text in read_pages(path, args.max_pages):
            for i, chunk in enumerate(chunk_words(text)):
                ids.append(f"f{f_idx}-p{page_num}-c{i}")
                docs.append(chunk)
                metas.append({"source": path.name, "page": page_num, **tag_components(chunk)})
                if len(docs) >= BATCH_SIZE:
                    flush()
    flush()
    print(f"Done. {total} chunks from {len(files)} files saved to ./{DB_PATH}")


if __name__ == "__main__":
    main()
