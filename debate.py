"""
Debate Engine — orchestrates multi-round discussion between two agents.
Detects agreement and produces a final summary.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from agents import BaseAgent, Message
import config

logger = logging.getLogger(__name__)

# Marker at the start of a line, optionally wrapped in markdown (**ИТОГ:**,
# "### ИТОГ:", "- ИТОГ:"), followed by a colon.
_AGREEMENT_RE = re.compile(
    r"^[\s>#*_\-]*(?:" + "|".join(re.escape(m) for m in config.AGREEMENT_MARKERS) + r")[\s*_]*:",
    re.IGNORECASE | re.MULTILINE,
)


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
        return bool(_AGREEMENT_RE.search(text or ""))

    def _build_final_summary(self, state: DebateState) -> str:
        """Build a formatted summary of the debate (HTML-safe)."""
        task_escaped = html.escape(state.task)
        agreed_text = "\u2705 \u0414\u0430" if state.is_agreed else "\u23f0 \u041d\u0435\u0442 (\u043b\u0438\u043c\u0438\u0442 \u0440\u0430\u0443\u043d\u0434\u043e\u0432)"

        lines = [
            "\u2501" * 14,
            "\U0001f4cb <b>\u0418\u0422\u041e\u0413 \u041e\u0411\u0421\u0423\u0416\u0414\u0415\u041d\u0418\u042f</b>",
            "\u2501" * 14,
            "",
            f"\U0001f4dd <b>\u0417\u0430\u0434\u0430\u043d\u0438\u0435:</b> {task_escaped}",
            f"\U0001f504 <b>\u0420\u0430\u0443\u043d\u0434\u043e\u0432:</b> {state.round_num}",
            f"\U0001f91d <b>\u0421\u043e\u0433\u043b\u0430\u0441\u0438\u0435:</b> {agreed_text}",
            "",
        ]

        # Find the last agreement message(s)
        agreement_msgs = []
        for msg in reversed(state.messages):
            if msg.role != "user" and self._check_agreement(msg.content):
                agreement_msgs.insert(0, msg)
            if len(agreement_msgs) >= 2:
                break

        if agreement_msgs:
            lines.append("<b>\u0424\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u043e\u0437\u0438\u0446\u0438\u0438:</b>")
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
                lines.append(f"<b>\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u043f\u043e\u0437\u0438\u0446\u0438\u044f {html.escape(last_alpha.sender)}:</b>")
                lines.append(html.escape(last_alpha.content[:500]))
                lines.append("")
            if last_beta:
                lines.append(f"<b>\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u043f\u043e\u0437\u0438\u0446\u0438\u044f {html.escape(last_beta.sender)}:</b>")
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
            sender="\U0001f464 \u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c",
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
