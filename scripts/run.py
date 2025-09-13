#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.retriever import SimpleRetriever
from rag_agent.llm import EchoLLM
from rag_agent.pipeline import RAGPipeline


def load_corpus(data_path: Path) -> list[str]:
    """Load corpus from text file, splitting by double newlines."""
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return ["Default doc: The sky is blue.", "Default doc: Grass is green."]
    
    with open(data_path, "r") as f:
        text = f.read()
    
    # Split by double newlines to get documents
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    return docs if docs else ["Empty corpus."]


def main():
    # Load corpus
    data_file = Path(__file__).parent.parent / "data" / "sample.txt"
    corpus = load_corpus(data_file)
    
    # Initialize components
    retriever = SimpleRetriever(corpus=corpus)
    llm = EchoLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)
    
    print("RAG Agent CLI (type 'quit' to exit)")
    print(f"Loaded {len(corpus)} documents from corpus.\n")
    
    while True:
        query = input("Query: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        if not query:
            continue
        
        answer, results = pipeline.query_with_scores(query, k=2)
        
        print("\nRetrieved documents:")
        for i, (doc, score) in enumerate(results, 1):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"  {i}. (score: {score:.3f}) {preview}")
        
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()