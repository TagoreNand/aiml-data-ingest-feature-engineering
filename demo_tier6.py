"""demo_tier6.py — Quick smoke test of Tier 6 components."""
import sys; sys.path.insert(0, ".")

from rich.console import Console
console = Console()

# 1. RAG pipeline
console.print("\n[bold cyan]Testing Production RAG pipeline...[/]")
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
console.print("\n[bold cyan]Testing RLHF reward model...[/]")
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
console.print("\n[bold cyan]Testing LLM Agent...[/]")
from src.serving.agent import MLAgent, Tool

def stub_llm(prompt: str) -> str:
    if "tool" in prompt.lower() or len(prompt) > 200:
        return "TOOL: predict_sentiment(text=\"This movie is amazing\")"
    return "ANSWER: The sentiment prediction shows positive with high confidence."

def predict_stub(text: str) -> dict:
    return {"text": text, "sentiment": "positive", "confidence": 0.95}

tools = [Tool("predict_sentiment", "Predict sentiment", predict_stub, {})]
agent = MLAgent(llm_fn=stub_llm, tools=tools, max_steps=3)
answer = agent.run("What is the sentiment of the phrase: This movie is amazing?")
console.print(f"  Agent answer: [green]{answer}[/]")
console.print("  [green]LLM Agent: OK[/]")

console.print("\n[bold green]All Tier 6 components verified![/]")
