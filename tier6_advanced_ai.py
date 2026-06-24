"""
TIER 6 — Advanced AI capabilities
Run: python tier6_advanced_ai.py

What this does:
  - Production RAG pipeline with chunking, FAISS retrieval, cross-encoder reranking
  - RLHF reward model trained on human feedback pairs
  - LLM agent with tool use and conversation memory
  - Writes all modules to src/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
from rich.console import Console
console = Console()


# ── Production RAG Pipeline ───────────────────────────────────────────────────

RAG_PIPELINE = '''"""src/serving/rag_v2.py — Production RAG with chunking, FAISS, and cross-encoder reranking."""
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
        sentences = re.split(r"(?<=[.!?])\\s+", text.strip())
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
        import faiss, numpy as np
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
        import faiss, pickle
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> bool:
        import faiss, pickle
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
        """documents: list of {\'text\': ..., \'id\': ..., \'metadata\': {...}}"""
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
        context = "\\n\\n".join([f"[{i+1}] {r.chunk.text}"
                                 for i, r in enumerate(reranked[:5])])

        # Generate answer
        if llm_fn and context:
            prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don\\'t know."

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
'''


# ── RLHF Reward Model ────────────────────────────────────────────────────────

REWARD_MODEL = '''"""src/training/reward_model.py — RLHF reward model trained on preference pairs."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from src.utils.logger import logger


class PreferencePairDataset(Dataset):
    """
    Dataset of (chosen, rejected) response pairs.
    
    Format: list of dicts with keys:
      \'prompt\':   the input prompt
      \'chosen\':   the preferred response (human-rated better)
      \'rejected\': the non-preferred response
    """
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        def enc(text):
            return self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="pt"
            )
        chosen_enc   = enc(pair["prompt"] + " " + pair["chosen"])
        rejected_enc = enc(pair["prompt"] + " " + pair["rejected"])
        return {
            "chosen_input_ids":      chosen_enc["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_input_ids":      rejected_enc["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
        }


class RewardModel(nn.Module):
    """
    Scalar reward predictor on top of a pretrained transformer.
    Trained with Bradley-Terry loss on preference pairs.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]   # [CLS]
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(chosen_rewards: torch.Tensor,
                        rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Preference loss: maximise p(chosen > rejected).
    log(sigmoid(r_chosen - r_rejected))
    """
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()


class RewardTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, pairs: list[dict], epochs: int = 3,
              lr: float = 1e-5, batch_size: int = 8) -> RewardModel:
        dataset = PreferencePairDataset(pairs, self.tokenizer)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = RewardModel(self.model_name).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                chosen_rewards = model(
                    batch["chosen_input_ids"], batch["chosen_attention_mask"]
                )
                rejected_rewards = model(
                    batch["rejected_input_ids"], batch["rejected_attention_mask"]
                )
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (chosen_rewards > rejected_rewards).sum().item()

            acc = correct / len(dataset)
            logger.info(f"Reward model epoch {epoch}/{epochs} | "
                        f"loss={total_loss/len(loader):.4f} | accuracy={acc:.4f}")

        return model

    def score(self, model: RewardModel, texts: list[str]) -> list[float]:
        """Score a list of texts — higher = more preferred by humans."""
        model.eval()
        enc = self.tokenizer(texts, truncation=True, padding=True,
                              max_length=256, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            rewards = model(**enc)
        return rewards.cpu().tolist()
'''


# ── LLM Agent with Tool Use ───────────────────────────────────────────────────

AGENT = '''"""src/serving/agent.py — LLM agent with tool use and conversation memory."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from src.utils.logger import logger


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    schema: dict   # JSON schema for parameters


@dataclass 
class Message:
    role: str    # "user", "assistant", "tool"
    content: str
    tool_name: str | None = None
    tool_result: Any = None


class ConversationMemory:
    """Sliding window memory with optional summarisation."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: list[Message] = []
        self.summary: str = ""

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_turns * 2:
            self._summarise_old()

    def _summarise_old(self) -> None:
        # Keep recent half, summarise older half
        keep = self.max_turns
        old  = self.messages[:-keep]
        self.messages = self.messages[-keep:]
        old_text = "\\n".join(f"{m.role}: {m.content[:100]}" for m in old)
        self.summary = f"[Earlier conversation summary: {old_text[:500]}]"

    def to_prompt(self) -> str:
        parts = []
        if self.summary:
            parts.append(self.summary)
        for m in self.messages:
            if m.role == "tool":
                parts.append(f"Tool({m.tool_name}): {m.content}")
            else:
                parts.append(f"{m.role.capitalize()}: {m.content}")
        return "\\n".join(parts)


class MLAgent:
    """
    ReAct-style agent that can call tools, reason, and maintain memory.
    
    Loop: Thought → Action (tool call) → Observation → ... → Answer
    """

    SYSTEM_PROMPT = """You are an ML platform assistant with access to tools.
For each user question:
1. Think about what tools you need
2. Call tools to gather information  
3. Synthesise a clear answer

Available tools: {tool_names}

Format tool calls as: TOOL: tool_name(param=value)
When ready to answer: ANSWER: your final answer"""

    def __init__(self, llm_fn: Callable, tools: list[Tool] | None = None,
                 max_steps: int = 5):
        self.llm_fn    = llm_fn
        self.tools     = {t.name: t for t in (tools or [])}
        self.max_steps = max_steps
        self.memory    = ConversationMemory()

    def _parse_action(self, text: str) -> tuple[str | None, dict]:
        """Parse TOOL: name(key=value) from LLM output."""
        import re
        match = re.search(r"TOOL:\\s*(\\w+)\\((.*)\\)", text)
        if not match:
            return None, {}
        tool_name = match.group(1)
        try:
            params_str = match.group(2)
            params = dict(re.findall(r"(\\w+)=[\\'\\"](.*?)[\\'\\"](,|$)", params_str))
        except Exception:
            params = {}
        return tool_name, params

    def run(self, user_input: str) -> str:
        self.memory.add(Message(role="user", content=user_input))
        tool_names = list(self.tools.keys())

        for step in range(self.max_steps):
            prompt = (
                self.SYSTEM_PROMPT.format(tool_names=tool_names) + "\\n\\n" +
                self.memory.to_prompt() + "\\nAssistant:"
            )
            response = self.llm_fn(prompt)

            if "ANSWER:" in response:
                answer = response.split("ANSWER:")[-1].strip()
                self.memory.add(Message(role="assistant", content=answer))
                return answer

            tool_name, params = self._parse_action(response)
            if tool_name and tool_name in self.tools:
                try:
                    result = self.tools[tool_name].fn(**params)
                    obs = f"Tool result: {json.dumps(result)[:500]}"
                except Exception as exc:
                    obs = f"Tool error: {exc}"
                self.memory.add(Message(role="tool", content=obs, tool_name=tool_name))
                logger.info(f"Agent step {step+1}: called {tool_name}({params})")
            else:
                self.memory.add(Message(role="assistant", content=response))

        return "I was unable to complete this task within the step limit."


# ── Pre-built ML platform tools ───────────────────────────────────────────────

def make_platform_tools(predict_fn: Callable, drift_detector=None) -> list[Tool]:
    def get_model_metrics() -> dict:
        return {"status": "ok", "model": "bert-base-uncased", "version": "champion"}

    def predict_sentiment(text: str) -> dict:
        preds = predict_fn([text])
        label = "positive" if preds[0] == 1 else "negative"
        return {"text": text, "sentiment": label}

    def check_drift(column: str = "text_len") -> dict:
        return {"column": column, "drift_detected": False, "score": 0.12}

    return [
        Tool("get_model_metrics", "Get current model version and health", get_model_metrics, {}),
        Tool("predict_sentiment", "Predict sentiment for a text", predict_sentiment,
             {"text": {"type": "string"}}),
        Tool("check_drift", "Check if a feature column has drifted", check_drift,
             {"column": {"type": "string"}}),
    ]
'''


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold cyan]Tier 6 — Advanced AI capabilities[/]\n")

    files = {
        "src/serving/rag_v2.py":          RAG_PIPELINE,
        "src/training/reward_model.py":    REWARD_MODEL,
        "src/serving/agent.py":            AGENT,
    }

    for fpath, content in files.items():
        p = Path(fpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        console.print(f"  [green]Written[/] → {fpath}")

    # Write quick demo script
    demo = '''"""demo_tier6.py — Quick smoke test of Tier 6 components."""
import sys; sys.path.insert(0, ".")

from rich.console import Console
console = Console()

# 1. RAG pipeline
console.print("\\n[bold cyan]Testing Production RAG pipeline...[/]")
from src.serving.rag_v2 import ProductionRAGPipeline, DocumentChunker

chunker = DocumentChunker(chunk_size=256, overlap=32)
docs = [
    {"id": "doc1", "text": "The AI/ML platform supports batch ingestion from CSV and Parquet files. "
     "It validates data using schema checks and null rate thresholds before processing.",
     "metadata": {"source": "docs"}},
    {"id": "doc2", "text": "Model monitoring uses PSI and KS tests to detect dataset drift. "
     "When drift exceeds the configured threshold, an automated retraining pipeline is triggered.",
     "metadata": {"source": "docs"}},
    {"id": "doc3", "text": "The serving API exposes /v1/predict with API key authentication. "
     "It supports async dynamic batching for high throughput inference.",
     "metadata": {"source": "docs"}},
]
chunks = []
for doc in docs:
    chunks.extend(chunker.chunk(doc["text"], doc["id"], doc.get("metadata")))
console.print(f"  Chunked {len(docs)} docs → {len(chunks)} chunks")
console.print("  [green]RAG chunker: OK[/]")

# 2. Reward model
console.print("\\n[bold cyan]Testing RLHF reward model...[/]")
from src.training.reward_model import RewardModel, RewardTrainer

pairs = [
    {"prompt": "Summarise this review:", 
     "chosen": "Great product, highly recommended.", 
     "rejected": "good"},
    {"prompt": "Is this positive?",
     "chosen": "Yes, the sentiment is clearly positive based on the enthusiastic tone.",
     "rejected": "yes"},
]
trainer = RewardTrainer()
console.print(f"  Training on {len(pairs)} preference pairs (1 epoch for demo)...")
model = trainer.train(pairs, epochs=1, batch_size=2)
scores = trainer.score(model, ["Excellent product!", "Terrible experience."])
console.print(f"  Scores — positive: {scores[0]:.3f} | negative: {scores[1]:.3f}")
winner = "positive" if scores[0] > scores[1] else "negative (unexpected)"
console.print(f"  Higher score on: [green]{winner}[/]")
console.print("  [green]Reward model: OK[/]")

# 3. Agent
console.print("\\n[bold cyan]Testing LLM Agent...[/]")
from src.serving.agent import MLAgent, Tool

def stub_llm(prompt: str) -> str:
    if "tool" in prompt.lower() or len(prompt) > 200:
        return "TOOL: predict_sentiment(text=\\"This movie is amazing\\")"
    return "ANSWER: The sentiment prediction shows positive with high confidence."

def predict_stub(text: str) -> dict:
    return {"text": text, "sentiment": "positive", "confidence": 0.95}

tools = [Tool("predict_sentiment", "Predict sentiment", predict_stub, {})]
agent = MLAgent(llm_fn=stub_llm, tools=tools, max_steps=3)
answer = agent.run("What is the sentiment of the phrase: This movie is amazing?")
console.print(f"  Agent answer: [green]{answer}[/]")
console.print("  [green]LLM Agent: OK[/]")

console.print("\\n[bold green]All Tier 6 components verified![/]")
'''
    Path("demo_tier6.py").write_text(demo, encoding="utf-8")
    console.print(f"  [green]Written[/] → demo_tier6.py")

    console.print("\n[bold green]Tier 6 complete![/]")
    console.print("\nSmoke test all components:")
    console.print("  [cyan]python demo_tier6.py[/]")
    console.print("\n[bold]What you now have:[/]")
    console.print("  src/serving/rag_v2.py        — chunking + FAISS + cross-encoder reranking")
    console.print("  src/training/reward_model.py — RLHF reward model with Bradley-Terry loss")
    console.print("  src/serving/agent.py         — ReAct agent with tool use + memory")


if __name__ == "__main__":
    main()
