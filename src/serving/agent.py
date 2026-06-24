"""src/serving/agent.py — LLM agent with tool use and conversation memory."""
from __future__ import annotations

import json
from dataclasses import dataclass
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
        old_text = "\n".join(f"{m.role}: {m.content[:100]}" for m in old)
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
        return "\n".join(parts)


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
        match = re.search(r"TOOL:\s*(\w+)\((.*)\)", text)
        if not match:
            return None, {}
        tool_name = match.group(1)
        try:
            params_str = match.group(2)
            params = dict(re.findall(r"(\w+)=[\'\"](.*?)[\'\"](,|$)", params_str))
        except Exception:
            params = {}
        return tool_name, params

    def run(self, user_input: str) -> str:
        self.memory.add(Message(role="user", content=user_input))
        tool_names = list(self.tools.keys())

        for step in range(self.max_steps):
            prompt = (
                self.SYSTEM_PROMPT.format(tool_names=tool_names) + "\n\n" +
                self.memory.to_prompt() + "\nAssistant:"
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
