# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from .retriever import SimpleRetriever
from .llm import EchoLLM


@dataclass
class RAGPipeline:
    retriever: SimpleRetriever
    llm: EchoLLM

    def query(self, q: str, k: int = 3) -> str:
        # Retrieve top-k relevant documents
        results = self.retriever.query(q, k=k)
        if not results:
            return self.llm.complete(q)
        
        context = [doc for doc, _ in results]
        return self.llm.complete_with_context(q, context)
    
    def query_with_scores(self, q: str, k: int = 3) -> Tuple[str, List[Tuple[str, float]]]:
        # Returns both the answer and the retrieved documents with scores
        results = self.retriever.query(q, k=k)
        answer = self.query(q, k=k)
        return answer, results