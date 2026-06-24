"""src/serving/rag_v2.py — Production RAG with chunking, FAISS, and cross-encoder reranking."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from src.utils.logger import logger


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    metadata: dict
    embedding: np.ndarray | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rerank_score: float | None = None


class DocumentChunker:
    """
    Splits documents into overlapping chunks for better retrieval.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        import re
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current, current_len = [], 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                chunks.append(Chunk(
                    text=chunk_text, doc_id=doc_id, chunk_id=chunk_id,
                    metadata=metadata or {},
                ))
                # Keep overlap sentences
                overlap_text = " ".join(current)
                overlap_words = overlap_text.split()[-self.overlap:]
                current = [" ".join(overlap_words)]
                current_len = len(current[0])
            current.append(sent)
            current_len += sent_len

        if current:
            chunk_text = " ".join(current)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunks.append(Chunk(text=chunk_text, doc_id=doc_id, chunk_id=chunk_id,
                                metadata=metadata or {}))
        return chunks


class EmbeddingEncoder:
    """Encodes text using sentence-transformers with batching and caching."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        )


class FAISSIndex:
    """FAISS vector store with persistence."""

    def __init__(self, dim: int = 384, index_path: str = "models/faiss_v2"):
        self.dim        = dim
        self.index_path = Path(index_path)
        self._index     = None
        self._chunks: list[Chunk] = []

    def _build_index(self) -> Any:
        import faiss
        index = faiss.IndexFlatIP(self.dim)   # Inner product (cosine for normalised vecs)
        return faiss.IndexIDMap(index)

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np
        if self._index is None:
            self._index = self._build_index()
        embeddings = np.array([c.embedding for c in chunks if c.embedding is not None],
                               dtype=np.float32)
        ids = np.arange(len(self._chunks), len(self._chunks) + len(chunks), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)
        self._chunks.extend(chunks)
        logger.info(f"FAISS index: {self._index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievedChunk]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, ids = self._index.search(q, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0 and idx < len(self._chunks):
                results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    def save(self) -> None:
        import faiss
        import pickle
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> bool:
        import faiss
        import pickle
        idx_file = self.index_path / "index.faiss"
        if not idx_file.exists():
            return False
        self._index = faiss.read_index(str(idx_file))
        with open(self.index_path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
        return True


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder for better precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_k: int = 5) -> list[RetrievedChunk]:
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda x: x.rerank_score or 0, reverse=True)[:top_k]


class ProductionRAGPipeline:
    """
    End-to-end RAG:
      index() — chunk + embed + store documents
      query() — retrieve + rerank + generate answer
    """

    def __init__(self, config: dict | None = None):
        from src.utils.config import load_config
        self.cfg      = config or load_config()
        vec_cfg       = self.cfg.get("vector_db", {})
        emb_model     = vec_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunker  = DocumentChunker(chunk_size=512, overlap=64)
        self.encoder  = EmbeddingEncoder(model_name=emb_model)
        self.index    = FAISSIndex(dim=vec_cfg.get("embedding_dim", 384),
                                   index_path=vec_cfg.get("index_path", "models/faiss_v2"))
        self.reranker = CrossEncoderReranker()
        self.top_k    = vec_cfg.get("top_k", 10)

    def index_documents(self, documents: list[dict]) -> int:
        """documents: list of {'text': ..., 'id': ..., 'metadata': {...}}"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc["text"], doc["id"], doc.get("metadata", {}))
            all_chunks.extend(chunks)

        texts = [c.text for c in all_chunks]
        embeddings = self.encoder.encode(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self.index.add(all_chunks)
        self.index.save()
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
        return len(all_chunks)

    def query(self, question: str, top_k: int | None = None,
              llm_fn: callable | None = None) -> dict:
        t0 = time.perf_counter()
        k = top_k or self.top_k

        # Retrieve
        q_emb = self.encoder.encode([question])[0]
        candidates = self.index.search(q_emb, top_k=k * 2)

        # Rerank
        if candidates:
            reranked = self.reranker.rerank(question, candidates, top_k=k)
        else:
            reranked = []

        # Build context
        context = "\n\n".join([f"[{i+1}] {r.chunk.text}"
                                 for i, r in enumerate(reranked[:5])])

        # Generate answer
        if llm_fn and context:
            prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don\'t know."

Context:
{context}

Question: {question}
Answer:"""
            answer = llm_fn(prompt)
        else:
            answer = context[:500] + "..." if context else "No relevant documents found."

        return {
            "answer": answer,
            "sources": [{"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                          "score": round(r.rerank_score or r.score, 4),
                          "text_preview": r.chunk.text[:200]}
                        for r in reranked],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
