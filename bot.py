"""
Multi-Agent Debate Telegram Bot.

Commands:
    /start   - Welcome message and instructions
    /task    - Submit a task for agents to discuss
    /stop    - Stop the current debate
    /history - View debate history
    /rounds  - Set max rounds
    /status  - Check if a debate is running
"""

from __future__ import annotations

import asyncio
import logging
import html

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode

import config
from agents import GroqAgent, GeminiAgent, Message
from debate import DebateEngine, DebateState

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Global State (per-chat debates) ──────────────────────
active_debates: dict[int, DebateState] = {}
debate_locks: dict[int, asyncio.Lock] = {}


def get_lock(chat_id: int) -> asyncio.Lock:
    """Get or create a per-chat lock to prevent race conditions."""
    if chat_id not in debate_locks:
        debate_locks[chat_id] = asyncio.Lock()
    return debate_locks[chat_id]


# ── Helpers ──────────────────────────────────────────────

def safe_truncate(text: str, max_len: int = 3800) -> str:
    """Truncate text safely BEFORE HTML wrapping to avoid breaking tags."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n... (truncated)"


def format_agent_message(msg: Message) -> str:
    """Format a debate message for Telegram (HTML-safe)."""
    # Truncate raw content first, then escape for HTML
    content = html.escape(safe_truncate(msg.content))
    if msg.role == "user":
        return f"\U0001f464 <b>User:</b>\n{content}"
    elif msg.role == "alpha":
        return f"\U0001f7e3 <b>Agent Alpha</b> <i>(Qwen 3.6)</i>:\n\n{content}"
    elif msg.role == "beta":
        return f"\U0001f535 <b>Agent Beta</b> <i>(Gemini 3.6 Flash)</i>:\n\n{content}"
    else:
        return content


async def safe_send(bot, chat_id: int, text: str, parse_mode=ParseMode.HTML) -> None:
    """Send message with fallback to plain text if HTML parsing fails."""
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    except Exception as e:
        logger.warning(f"HTML send failed ({e}), falling back to plain text")
        try:
            # Strip HTML tags for plain text fallback
            import re
            plain = re.sub(r"<[^>]+>", "", text)
            await bot.send_message(chat_id=chat_id, text=plain)
        except Exception as e2:
            logger.error(f"Failed to send message at all: {e2}")


# ── Command Handlers ─────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "\U0001f916 <b>Multi-Agent AI Debate Bot</b>\n"
        "\n"
        "Two AI agents will discuss your task and find the best solution together!\n\n"
        "\U0001f7e3 <b>Agent Alpha</b> \u2014 practitioner (Groq / Qwen 3.6)\n"
        "\U0001f535 <b>Agent Beta</b> \u2014 analyst (Google Gemini 3.6 Flash)\n\n"
        "<b>Commands:</b>\n"
        "\u2022 /task <i>text</i> \u2014 submit a task\n"
        "\u2022 /stop \u2014 stop the discussion\n"
        "\u2022 /history \u2014 debate history\n"
        "\u2022 /rounds <i>N</i> \u2014 max rounds (default: 8)\n"
        "\u2022 /status \u2014 debate status\n\n"
        "\U0001f4a1 During the debate you can send messages \u2014 agents will consider them!\n"
    )
    await safe_send(context.bot, update.effective_chat.id, text)


async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /task command - start a new debate."""
    chat_id = update.effective_chat.id

    # Use lock to prevent race conditions on double-tap
    async with get_lock(chat_id):
        # Check if debate is already running
        if chat_id in active_debates and active_debates[chat_id].is_active:
            await safe_send(
                context.bot, chat_id,
                "\u26a0\ufe0f Discussion already running! Use /stop to cancel it.",
            )
            return

        # Get the task text
        task_text = " ".join(context.args) if context.args else ""
        if not task_text:
            await safe_send(
                context.bot, chat_id,
                "\u274c Specify a task!\n\nExample: <code>/task Write a sort function in Python</code>",
            )
            return

        # Get max rounds from context or default
        max_rounds = context.chat_data.get("max_rounds", config.MAX_ROUNDS)

        # Create SHARED debate state (used by both engine and bot)
        state = DebateState(task=task_text, max_rounds=max_rounds)
        active_debates[chat_id] = state

        task_escaped = html.escape(task_text)
        await safe_send(
            context.bot, chat_id,
            f"\U0001f680 <b>Starting discussion!</b>\n\n"
            f"\U0001f4dd Task: <i>{task_escaped}</i>\n"
            f"\U0001f504 Max rounds: {max_rounds}\n\n"
            f"\U0001f7e3 Agent Alpha and \U0001f535 Agent Beta begin debating...",
        )

    # Create agents and engine
    agent_alpha = GroqAgent()
    agent_beta = GeminiAgent()
    engine = DebateEngine(agent_alpha, agent_beta, max_rounds=max_rounds)

    # Real-time message callback
    async def on_message(msg: Message) -> None:
        if msg.role == "user":
            return  # Don't echo back user's own message
        formatted = format_agent_message(msg)
        await safe_send(context.bot, chat_id, formatted)

    # Run debate in background
    async def run_debate_task():
        try:
            # Pass the SHARED state into the engine
            await engine.run_debate(state, on_message=on_message)

            # Send final summary
            if state.final_summary:
                await safe_send(context.bot, chat_id, state.final_summary)
        except asyncio.CancelledError:
            logger.info(f"Debate cancelled for chat {chat_id}")
            state.is_active = False
            await safe_send(context.bot, chat_id, "\U0001f6d1 Discussion stopped.")
        except Exception as e:
            logger.error(f"Debate error: {e}", exc_info=True)
            state.is_active = False
            await safe_send(
                context.bot, chat_id,
                f"\u274c Debate error: {html.escape(str(e))}",
            )

    # Store task reference for cancellation
    task = asyncio.create_task(run_debate_task())
    context.chat_data["debate_task"] = task


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command - cancel ongoing debate."""
    chat_id = update.effective_chat.id

    task = context.chat_data.get("debate_task")
    if task and not task.done():
        task.cancel()
        context.chat_data.pop("debate_task", None)
        if chat_id in active_debates:
            active_debates[chat_id].is_active = False
        await safe_send(context.bot, chat_id, "\U0001f6d1 Discussion stopped.")
    else:
        await safe_send(context.bot, chat_id, "No active discussion.")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command - show debate history."""
    chat_id = update.effective_chat.id
    state = active_debates.get(chat_id)

    if not state or not state.messages:
        await safe_send(context.bot, chat_id, "No debate history.")
        return

    lines = [
        f"\U0001f4dc <b>Debate History</b>\n",
        f"\U0001f4dd Task: <i>{html.escape(state.task)}</i>\n",
        f"\U0001f504 Rounds: {state.round_num}\n",
    ]

    for msg in state.messages:
        preview = msg.content[:150]
        if len(msg.content) > 150:
            preview += "..."
        lines.append(f"<b>{html.escape(msg.sender)}:</b> {html.escape(preview)}\n")

    text = "\n".join(lines)
    await safe_send(context.bot, chat_id, safe_truncate(text))


async def cmd_rounds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /rounds command - set max rounds."""
    chat_id = update.effective_chat.id

    if not context.args:
        current = context.chat_data.get("max_rounds", config.MAX_ROUNDS)
        await safe_send(
            context.bot, chat_id,
            f"\U0001f504 Current round limit: <b>{current}</b>\n\n"
            f"To change: <code>/rounds 5</code>",
        )
        return

    try:
        n = int(context.args[0])
        if n < 1 or n > 20:
            raise ValueError("out of range")
        context.chat_data["max_rounds"] = n
        await safe_send(
            context.bot, chat_id,
            f"\u2705 Max rounds set to: <b>{n}</b>",
        )
    except ValueError:
        await safe_send(context.bot, chat_id, "\u274c Enter a number from 1 to 20.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    chat_id = update.effective_chat.id
    state = active_debates.get(chat_id)

    if not state:
        await safe_send(context.bot, chat_id, "No debates yet.")
        return

    status = "\U0001f7e2 Active" if state.is_active else "\u23f9\ufe0f Finished"
    agreed = "\u2705 Yes" if state.is_agreed else "\u274c No"

    text = (
        f"\U0001f4ca <b>Debate Status</b>\n\n"
        f"\U0001f4dd Task: <i>{html.escape(state.task[:100])}</i>\n"
        f"\U0001f504 Round: {state.round_num}/{state.max_rounds}\n"
        f"Status: {status}\n"
        f"Agreement: {agreed}\n"
        f"Messages: {len(state.messages)}"
    )
    await safe_send(context.bot, chat_id, text)


async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages - inject into active debate."""
    chat_id = update.effective_chat.id
    state = active_debates.get(chat_id)

    if state and state.is_active:
        # Inject user's message into the SHARED debate state
        msg = Message(
            role="user",
            sender="User",
            content=update.message.text,
        )
        state.messages.append(msg)
        await safe_send(
            context.bot, chat_id,
            "\U0001f4ac Your message has been added. Agents will consider it in their next response.",
        )
    else:
        await safe_send(
            context.bot, chat_id,
            "No active discussion.\n\nUse /task to start one!",
        )


# ── Main ─────────────────────────────────────────────────

def main() -> None:
    """Start the bot."""
    # Validate configuration
    missing = config.validate_config()
    if missing:
        print("[ERROR] Missing config keys:")
        for key in missing:
            print(f"   - {key}")
        print("\nCreate .env from .env.example and fill in the keys.")
        return

    print("[BOT] Starting Multi-Agent Debate Bot...")
    print(f"   [ALPHA] Agent Alpha: Groq ({config.GROQ_MODEL})")
    print(f"   [BETA]  Agent Beta: Gemini ({config.GEMINI_MODEL})")
    print(f"   [ROUNDS] Max rounds: {config.MAX_ROUNDS}")
    print()

    # Build application
    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("task", cmd_task))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("rounds", cmd_rounds))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    print("[OK] Bot is running! Send /start in Telegram.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
