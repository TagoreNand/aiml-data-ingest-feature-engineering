"""src/serving/rag.py — Retrieval-Augmented Generation pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import RAGRequest, RAGResponse


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Embed the query with a sentence-transformer.
      2. Retrieve top-k chunks from a FAISS index.
      3. Build a prompt and call an LLM for the final answer.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.vdb_cfg = self.cfg.get("vector_db", {})
        self.rag_cfg = self.cfg.get("serving", {}).get("rag", {})
        self._embedder = None
        self._index = None
        self._chunks: list[str] = []
        self._meta: list[dict] = []

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.vdb_cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return self._embedder

    def _get_index(self):
        if self._index is None:
            import faiss
            index_path = Path(self.vdb_cfg.get("index_path", "models/faiss_index"))
            if index_path.exists():
                self._index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index from {index_path} ({self._index.ntotal} vectors).")
            else:
                dim = self.vdb_cfg.get("embedding_dim", 384)
                self._index = faiss.IndexFlatL2(dim)
                logger.warning(f"No FAISS index at {index_path}. Created empty index.")
        return self._index

    # ── Index management ──────────────────────────────────────────────────────

    def add_documents(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        """Embed and add documents to the FAISS index."""
        import faiss
        embedder = self._get_embedder()
        index = self._get_index()
        vectors = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        index.add(np.array(vectors).astype("float32"))
        self._chunks.extend(texts)
        self._meta.extend(metadata or [{} for _ in texts])
        index_path = Path(self.vdb_cfg.get("index_path", "models/faiss_index"))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        logger.info(f"Added {len(texts)} documents. Index now has {index.ntotal} vectors.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        embedder = self._get_embedder()
        index = self._get_index()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        distances, indices = index.search(np.array(q_vec).astype("float32"), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._chunks):
                results.append({
                    "text": self._chunks[idx],
                    "score": float(1 / (1 + dist)),
                    "metadata": self._meta[idx] if idx < len(self._meta) else {},
                })
        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.rag_cfg.get("llm_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": self.rag_cfg.get("system_prompt", "Answer based on the context.")},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.rag_cfg.get("max_tokens", 512),
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}. Returning context summary.")
            return "[LLM unavailable. Retrieved context returned above.]"

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def query(self, request: RAGRequest) -> RAGResponse:
        t0 = time.perf_counter()
        sources = self.retrieve(request.query, top_k=request.top_k)
        context = "\n\n".join(f"[{i+1}] {s['text']}" for i, s in enumerate(sources))
        prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
        answer = self._call_llm(prompt, request.temperature)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResponse(answer=answer, sources=sources, latency_ms=round(latency_ms, 2))
