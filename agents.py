"""
AI Agents: wrappers around Groq (Qwen 3.6 27B) and Google Gemini.
Both use async interfaces for proper non-blocking operation.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from groq import AsyncGroq
from google import genai
from google.genai import types as genai_types

import config

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class AgentError(RuntimeError):
    """An LLM call failed; the debate must stop instead of treating the error
    text as the agent's argument (which the other agent would then answer)."""


def strip_reasoning(text: str) -> str:
    """Qwen3 reasoning models may prepend a <think>...</think> block."""
    return _THINK_RE.sub("", text or "").strip()


@dataclass
class Message:
    """A single message in the debate conversation."""
    role: str          # "user", "alpha", "beta", "system"
    sender: str        # Display name, e.g. "Agent Alpha"
    content: str


class BaseAgent(ABC):
    """Abstract base class for an AI agent."""

    def __init__(self, name: str, emoji: str, system_prompt: str):
        self.name = name
        self.emoji = emoji
        self.system_prompt = system_prompt

    @property
    def display_name(self) -> str:
        return f"{self.emoji} {self.name}"

    @abstractmethod
    async def respond(self, conversation: list[Message]) -> str:
        """Generate a response given the conversation history."""
        ...

    def _build_messages_for_api(
        self, conversation: list[Message]
    ) -> list[dict[str, str]]:
        """Convert internal conversation to OpenAI-style message list."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in conversation:
            if msg.role == "user":
                role = "user"
            elif msg.role == "system":
                role = "system"
            else:
                # Own messages are "assistant", others are "user"
                if msg.sender == self.display_name:
                    role = "assistant"
                else:
                    role = "user"
            messages.append({
                "role": role,
                "content": f"[{msg.sender}]: {msg.content}" if role == "user" else msg.content,
            })
        return messages


class GroqAgent(BaseAgent):
    """Agent powered by Groq (Qwen 3.6 27B) — fast inference."""

    def __init__(self):
        super().__init__(
            name="Agent Alpha",
            emoji="\U0001f7e3",  # purple circle
            system_prompt=config.AGENT_ALPHA_SYSTEM,
        )
        self.client = AsyncGroq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL

    async def respond(self, conversation: list[Message]) -> str:
        messages = self._build_messages_for_api(conversation)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise AgentError(f"Agent Alpha (Groq) недоступен: {e}") from e
        return strip_reasoning(response.choices[0].message.content) or "(пустой ответ)"


class GeminiAgent(BaseAgent):
    """Agent powered by Google Gemini Flash — analytical style."""

    def __init__(self):
        super().__init__(
            name="Agent Beta",
            emoji="\U0001f535",  # blue circle
            system_prompt=config.AGENT_BETA_SYSTEM,
        )
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL

    async def respond(self, conversation: list[Message]) -> str:
        # Build a single prompt from conversation history
        prompt_parts = [self.system_prompt, "\n\n--- ИСТОРИЯ ОБСУЖДЕНИЯ ---\n"]
        for msg in conversation:
            prompt_parts.append(f"[{msg.sender}]: {msg.content}")
        prompt_parts.append(f"\nТеперь ответь как [{self.display_name}] на русском языке:")

        full_prompt = "\n".join(prompt_parts)

        try:
            # Use native async interface instead of asyncio.to_thread
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                ),
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise AgentError(f"Agent Beta (Gemini) недоступен: {e}") from e
        return (response.text or "").strip() or "(пустой ответ)"
