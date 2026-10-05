import os
import time
from typing import Any, Dict, List

from google import genai
from google.genai import errors, types

from .config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's documents. "
    "Answer ONLY using the provided context. If the context does not contain the answer, "
    "say you could not find it in the documents. Be concise and do not invent facts."
)
NO_CONTEXT_MESSAGE = "I couldn't find anything relevant in the documents."


def _source_label(metadata: dict) -> str:
    name = os.path.basename(str(metadata.get("source", "unknown")))
    page = metadata.get("page")
    return f"{name} (page {int(page) + 1})" if isinstance(page, (int, float)) else name


class GeminiGenerator:
    """Turns retrieved chunks + a question into a grounded answer."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _call(self, prompt: str, max_retries: int = 4):
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
                )
            except errors.ServerError as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt * 2
                print(f"Gemini busy ({e.code}), retrying in {wait}s...")
                time.sleep(wait)

    def answer(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not docs:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": []}

        blocks = [f"[Source {i}: {_source_label(d['metadata'])}]\n{d['content']}" for i, d in enumerate(docs, 1)]
        prompt = f"Context:\n{chr(10).join(blocks)}\n\nQuestion: {question}\n\nAnswer:"
        response = self._call(prompt)

        seen, sources = set(), []
        for d in docs:
            label = _source_label(d["metadata"])
            if label not in seen:
                seen.add(label)
                sources.append({"label": label, "score": round(d["similarity_score"], 3), "snippet": d["content"][:300]})

        return {"answer": response.text or NO_CONTEXT_MESSAGE, "sources": sources}