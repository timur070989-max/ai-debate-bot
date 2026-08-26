"""
Debate Engine — orchestrates multi-round discussion between two agents.
Detects agreement and produces a final summary.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from agents import BaseAgent, Message
import config

logger = logging.getLogger(__name__)


@dataclass
class DebateState:
    """Tracks the state of an ongoing debate."""
    task: str
    messages: list[Message] = field(default_factory=list)
    round_num: int = 0
    max_rounds: int = config.MAX_ROUNDS
    is_active: bool = True
    is_agreed: bool = False
    final_summary: str = ""

    @property
    def is_finished(self) -> bool:
        return not self.is_active or self.is_agreed or self.round_num >= self.max_rounds


class DebateEngine:
    """
    Manages a debate between two AI agents.

    Accepts an external DebateState so the bot and engine share
    the same state object (fixes user message injection).
    """

    def __init__(
        self,
        agent_alpha: BaseAgent,
        agent_beta: BaseAgent,
        max_rounds: int = config.MAX_ROUNDS,
    ):
        self.agent_alpha = agent_alpha
        self.agent_beta = agent_beta
        self.max_rounds = max_rounds

    def _check_agreement(self, text: str) -> bool:
        """Check if a message contains agreement markers."""
        text_upper = text.upper()
        return any(marker.upper() in text_upper for marker in config.AGREEMENT_MARKERS)

    def _build_final_summary(self, state: DebateState) -> str:
        """Build a formatted summary of the debate (HTML-safe)."""
        task_escaped = html.escape(state.task)
        agreed_text = "Da" if state.is_agreed else "Net (limit)"

        lines = [
            "<b>ITOG OBSUZHDENIYA</b>",
            "",
            f"<b>Zadanie:</b> {task_escaped}",
            f"<b>Raundov:</b> {state.round_num}",
            f"<b>Soglasie:</b> {agreed_text}",
            "",
        ]

        # Find the last agreement message(s)
        agreement_msgs = []
        for msg in reversed(state.messages):
            if self._check_agreement(msg.content):
                agreement_msgs.insert(0, msg)
            if len(agreement_msgs) >= 2:
                break

        if agreement_msgs:
            lines.append("<b>Final positions:</b>")
            lines.append("")
            for msg in agreement_msgs:
                lines.append(f"<b>{html.escape(msg.sender)}:</b>")
                lines.append(html.escape(msg.content[:500]))
                lines.append("")
        else:
            # Use last messages from each agent
            last_alpha = None
            last_beta = None
            for msg in reversed(state.messages):
                if "Alpha" in msg.sender and last_alpha is None:
                    last_alpha = msg
                elif "Beta" in msg.sender and last_beta is None:
                    last_beta = msg
                if last_alpha and last_beta:
                    break
            if last_alpha:
                lines.append(f"<b>Last position {html.escape(last_alpha.sender)}:</b>")
                lines.append(html.escape(last_alpha.content[:500]))
                lines.append("")
            if last_beta:
                lines.append(f"<b>Last position {html.escape(last_beta.sender)}:</b>")
                lines.append(html.escape(last_beta.content[:500]))
                lines.append("")

        return "\n".join(lines)

    async def run_debate(
        self,
        state: DebateState,
        on_message: Callable[[Message], Awaitable[None]] | None = None,
    ) -> DebateState:
        """
        Run a full debate using the provided shared state.

        Args:
            state: Shared DebateState (also used by bot for injection).
            on_message: Async callback for real-time updates.

        Returns:
            The same DebateState, updated with results.
        """
        # Initial user message
        user_msg = Message(
            role="user",
            sender="User",
            content=state.task,
        )
        state.messages.append(user_msg)
        if on_message:
            await on_message(user_msg)

        while not state.is_finished:
            state.round_num += 1
            logger.info(f"Debate round {state.round_num}/{state.max_rounds}")

            # ── Agent Alpha responds ──
            alpha_text = await self.agent_alpha.respond(state.messages)
            alpha_msg = Message(
                role="alpha",
                sender=self.agent_alpha.display_name,
                content=alpha_text,
            )
            state.messages.append(alpha_msg)
            if on_message:
                await on_message(alpha_msg)

            # Check if Alpha agreed
            if self._check_agreement(alpha_text):
                state.is_agreed = True
                beta_text = await self.agent_beta.respond(state.messages)
                beta_msg = Message(
                    role="beta",
                    sender=self.agent_beta.display_name,
                    content=beta_text,
                )
                state.messages.append(beta_msg)
                if on_message:
                    await on_message(beta_msg)
                break

            # ── Agent Beta responds ──
            beta_text = await self.agent_beta.respond(state.messages)
            beta_msg = Message(
                role="beta",
                sender=self.agent_beta.display_name,
                content=beta_text,
            )
            state.messages.append(beta_msg)
            if on_message:
                await on_message(beta_msg)

            # Check if Beta agreed
            if self._check_agreement(beta_text):
                state.is_agreed = True
                alpha_text2 = await self.agent_alpha.respond(state.messages)
                alpha_msg2 = Message(
                    role="alpha",
                    sender=self.agent_alpha.display_name,
                    content=alpha_text2,
                )
                state.messages.append(alpha_msg2)
                if on_message:
                    await on_message(alpha_msg2)
                break

        # Build final summary
        state.final_summary = self._build_final_summary(state)
        state.is_active = False

        return state
