#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.retriever import SimpleRetriever


def test_retriever_basic():
    """Test that retriever returns relevant documents."""
    corpus = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Machine learning uses data to make predictions.",
    ]
    
    retriever = SimpleRetriever(corpus=corpus)
    
    # Query about France
    results = retriever.query("What is the capital of France?", k=1)
    assert len(results) == 1
    assert "Paris" in results[0][0]
    assert results[0][1] > 0  # Score should be positive
    
    # Query about Python
    results = retriever.query("Tell me about Python", k=2)
    assert len(results) == 2
    # First result should mention Python
    assert "Python" in results[0][0] or "programming" in results[0][0]
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_retriever_basic()