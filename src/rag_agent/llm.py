# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class EchoLLM:
    """Placeholder LLM implementation that echoes input without needing API keys."""

    def complete(self, prompt: str) -> str:
        # In production, you'd call OpenAI or similar here
        return f"[ECHO] {prompt}"

    def complete_with_context(self, query: str, context: List[str]) -> str:
        full_prompt = f"Context: {' | '.join(context)}\n\nQuery: {query}\n\nAnswer:"
        return self.complete(full_prompt)