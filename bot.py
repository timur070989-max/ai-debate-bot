"""
Multi-Agent Debate Telegram Bot.

Commands:
    /start   - Welcome + keyboard buttons
    /stop    - Stop the current debate
    /history - View debate history
    /rounds  - Set max rounds
    /status  - Check if a debate is running
"""

from __future__ import annotations

import asyncio
import logging
import html
import re

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
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

# ── Constants ────────────────────────────────────────────
BTN_NEW_TASK = "\U0001f4dd \u041d\u043e\u0432\u043e\u0435 \u0437\u0430\u0434\u0430\u043d\u0438\u0435"
BTN_STOP = "\U0001f6d1 \u0421\u0442\u043e\u043f"
BTN_STATUS = "\U0001f4ca \u0421\u0442\u0430\u0442\u0443\u0441"
BTN_HISTORY = "\U0001f4dc \u0418\u0441\u0442\u043e\u0440\u0438\u044f"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton(BTN_NEW_TASK)],
        [KeyboardButton(BTN_STOP), KeyboardButton(BTN_STATUS), KeyboardButton(BTN_HISTORY)],
    ],
    resize_keyboard=True,
)

# ── Global State (per-chat) ──────────────────────────────
active_debates: dict[int, DebateState] = {}
debate_locks: dict[int, asyncio.Lock] = {}
waiting_for_task: set[int] = set()  # chats waiting for task text


def get_lock(chat_id: int) -> asyncio.Lock:
    """Get or create a per-chat lock to prevent race conditions."""
    if chat_id not in debate_locks:
        debate_locks[chat_id] = asyncio.Lock()
    return debate_locks[chat_id]


# ── Helpers ──────────────────────────────────────────────

def safe_truncate(text: str, max_len: int = 3800) -> str:
    """Truncate text safely BEFORE HTML wrapping."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n... (\u043e\u0431\u0440\u0435\u0437\u0430\u043d\u043e)"


def format_agent_message(msg: Message) -> str:
    """Format a debate message for Telegram (HTML-safe)."""
    content = html.escape(safe_truncate(msg.content))
    if msg.role == "user":
        return f"\U0001f464 <b>\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c:</b>\n{content}"
    elif msg.role == "alpha":
        return f"\U0001f7e3 <b>Agent Alpha</b> <i>(Qwen 3.6)</i>:\n\n{content}"
    elif msg.role == "beta":
        return f"\U0001f535 <b>Agent Beta</b> <i>(Gemini 3.6 Flash)</i>:\n\n{content}"
    else:
        return content


async def safe_send(bot, chat_id: int, text: str, parse_mode=ParseMode.HTML, reply_markup=None) -> None:
    """Send message with fallback to plain text if HTML parsing fails."""
    try:
        await bot.send_message(
            chat_id=chat_id, text=text,
            parse_mode=parse_mode, reply_markup=reply_markup,
        )
    except Exception as e:
        logger.warning(f"HTML send failed ({e}), falling back to plain text")
        try:
            plain = re.sub(r"<[^>]+>", "", text)
            await bot.send_message(chat_id=chat_id, text=plain, reply_markup=reply_markup)
        except Exception as e2:
            logger.error(f"Failed to send message: {e2}")


# ── Command Handlers ─────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "\U0001f916 <b>\u041c\u0443\u043b\u044c\u0442\u0438\u0430\u0433\u0435\u043d\u0442\u043d\u044b\u0439 AI \u0414\u0435\u0431\u0430\u0442-\u0411\u043e\u0442</b>\n"
        "\n"
        "\u0414\u0432\u0430 AI-\u0430\u0433\u0435\u043d\u0442\u0430 \u043e\u0431\u0441\u0443\u0434\u044f\u0442 \u0442\u0432\u043e\u0451 \u0437\u0430\u0434\u0430\u043d\u0438\u0435 \u0438 \u0432\u043c\u0435\u0441\u0442\u0435 \u043d\u0430\u0439\u0434\u0443\u0442 \u043b\u0443\u0447\u0448\u0435\u0435 \u0440\u0435\u0448\u0435\u043d\u0438\u0435!\n\n"
        "\U0001f7e3 <b>Agent Alpha</b> \u2014 \u043f\u0440\u0430\u043a\u0442\u0438\u043a (Groq / Qwen 3.6)\n"
        "\U0001f535 <b>Agent Beta</b> \u2014 \u0430\u043d\u0430\u043b\u0438\u0442\u0438\u043a (Google Gemini 3.6 Flash)\n\n"
        "\U0001f447 <b>\u041d\u0430\u0436\u043c\u0438 \u043a\u043d\u043e\u043f\u043a\u0443 \u043d\u0438\u0436\u0435 \u0447\u0442\u043e\u0431\u044b \u043d\u0430\u0447\u0430\u0442\u044c!</b>\n\n"
        "\U0001f4a1 \u0412\u043e \u0432\u0440\u0435\u043c\u044f \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f \u043c\u043e\u0436\u0435\u0448\u044c \u043f\u0438\u0441\u0430\u0442\u044c \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u044f \u2014 \u0430\u0433\u0435\u043d\u0442\u044b \u0443\u0447\u0442\u0443\u0442 \u0438\u0445!\n"
    )
    await safe_send(context.bot, update.effective_chat.id, text, reply_markup=MAIN_KEYBOARD)


async def start_task(chat_id: int, task_text: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new debate with the given task text."""
    async with get_lock(chat_id):
        # Check if debate is already running
        if chat_id in active_debates and active_debates[chat_id].is_active:
            await safe_send(
                context.bot, chat_id,
                "\u26a0\ufe0f \u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 \u0443\u0436\u0435 \u0438\u0434\u0451\u0442! \u041d\u0430\u0436\u043c\u0438 \U0001f6d1 \u0421\u0442\u043e\u043f \u0447\u0442\u043e\u0431\u044b \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c.",
                reply_markup=MAIN_KEYBOARD,
            )
            return

        max_rounds = context.chat_data.get("max_rounds", config.MAX_ROUNDS)

        # Create SHARED debate state
        state = DebateState(task=task_text, max_rounds=max_rounds)
        active_debates[chat_id] = state

        task_escaped = html.escape(task_text)
        await safe_send(
            context.bot, chat_id,
            f"\U0001f680 <b>\u0417\u0430\u043f\u0443\u0441\u043a\u0430\u044e \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435!</b>\n\n"
            f"\U0001f4dd \u0417\u0430\u0434\u0430\u043d\u0438\u0435: <i>{task_escaped}</i>\n"
            f"\U0001f504 \u041c\u0430\u043a\u0441. \u0440\u0430\u0443\u043d\u0434\u043e\u0432: {max_rounds}\n\n"
            f"\U0001f7e3 Agent Alpha \u0438 \U0001f535 Agent Beta \u043d\u0430\u0447\u0438\u043d\u0430\u044e\u0442 \u0434\u0435\u0431\u0430\u0442\u044b...",
            reply_markup=MAIN_KEYBOARD,
        )

    # Create agents and engine
    agent_alpha = GroqAgent()
    agent_beta = GeminiAgent()
    engine = DebateEngine(agent_alpha, agent_beta, max_rounds=max_rounds)

    # Real-time message callback
    async def on_message(msg: Message) -> None:
        if msg.role == "user":
            return
        formatted = format_agent_message(msg)
        await safe_send(context.bot, chat_id, formatted)

    # Run debate in background
    async def run_debate_task():
        try:
            await engine.run_debate(state, on_message=on_message)
            if state.final_summary:
                await safe_send(context.bot, chat_id, state.final_summary, reply_markup=MAIN_KEYBOARD)
        except asyncio.CancelledError:
            logger.info(f"Debate cancelled for chat {chat_id}")
            state.is_active = False
            await safe_send(context.bot, chat_id, "\U0001f6d1 \u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043e.", reply_markup=MAIN_KEYBOARD)
        except Exception as e:
            logger.error(f"Debate error: {e}", exc_info=True)
            state.is_active = False
            await safe_send(
                context.bot, chat_id,
                f"\u274c \u041e\u0448\u0438\u0431\u043a\u0430: {html.escape(str(e))}",
                reply_markup=MAIN_KEYBOARD,
            )

    task = asyncio.create_task(run_debate_task())
    context.chat_data["debate_task"] = task


async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /task command — start a new debate."""
    chat_id = update.effective_chat.id
    task_text = " ".join(context.args) if context.args else ""
    if not task_text:
        await safe_send(
            context.bot, chat_id,
            "\u274c \u0423\u043a\u0430\u0436\u0438 \u0437\u0430\u0434\u0430\u043d\u0438\u0435!\n\n\u041f\u0440\u0438\u043c\u0435\u0440: <code>/task \u041d\u0430\u043f\u0438\u0448\u0438 \u0444\u0443\u043d\u043a\u0446\u0438\u044e \u0441\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u043a\u0438 \u043d\u0430 Python</code>",
            reply_markup=MAIN_KEYBOARD,
        )
        return
    await start_task(chat_id, task_text, context)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command or Stop button."""
    chat_id = update.effective_chat.id
    waiting_for_task.discard(chat_id)

    task = context.chat_data.get("debate_task")
    if task and not task.done():
        task.cancel()
        context.chat_data.pop("debate_task", None)
        if chat_id in active_debates:
            active_debates[chat_id].is_active = False
        await safe_send(context.bot, chat_id, "\U0001f6d1 \u041e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u043e.", reply_markup=MAIN_KEYBOARD)
    else:
        await safe_send(context.bot, chat_id, "\u2139\ufe0f \u041d\u0435\u0442 \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0433\u043e \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f.", reply_markup=MAIN_KEYBOARD)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command or History button."""
    chat_id = update.effective_chat.id
    state = active_debates.get(chat_id)

    if not state or not state.messages:
        await safe_send(context.bot, chat_id, "\u2139\ufe0f \u041d\u0435\u0442 \u0438\u0441\u0442\u043e\u0440\u0438\u0438 \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0439.", reply_markup=MAIN_KEYBOARD)
        return

    lines = [
        f"\U0001f4dc <b>\u0418\u0441\u0442\u043e\u0440\u0438\u044f \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f</b>\n",
        f"\U0001f4dd \u0417\u0430\u0434\u0430\u043d\u0438\u0435: <i>{html.escape(state.task)}</i>\n",
        f"\U0001f504 \u0420\u0430\u0443\u043d\u0434\u043e\u0432: {state.round_num}\n",
    ]

    for msg in state.messages:
        preview = msg.content[:150]
        if len(msg.content) > 150:
            preview += "..."
        lines.append(f"<b>{html.escape(msg.sender)}:</b> {html.escape(preview)}\n")

    text = "\n".join(lines)
    await safe_send(context.bot, chat_id, safe_truncate(text), reply_markup=MAIN_KEYBOARD)


async def cmd_rounds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /rounds command."""
    chat_id = update.effective_chat.id

    if not context.args:
        current = context.chat_data.get("max_rounds", config.MAX_ROUNDS)
        await safe_send(
            context.bot, chat_id,
            f"\U0001f504 \u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u043b\u0438\u043c\u0438\u0442 \u0440\u0430\u0443\u043d\u0434\u043e\u0432: <b>{current}</b>\n\n"
            f"\u0427\u0442\u043e\u0431\u044b \u0438\u0437\u043c\u0435\u043d\u0438\u0442\u044c: <code>/rounds 5</code>",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    try:
        n = int(context.args[0])
        if n < 1 or n > 20:
            raise ValueError("out of range")
        context.chat_data["max_rounds"] = n
        await safe_send(context.bot, chat_id, f"\u2705 \u041c\u0430\u043a\u0441. \u0440\u0430\u0443\u043d\u0434\u043e\u0432: <b>{n}</b>", reply_markup=MAIN_KEYBOARD)
    except ValueError:
        await safe_send(context.bot, chat_id, "\u274c \u0423\u043a\u0430\u0436\u0438 \u0447\u0438\u0441\u043b\u043e \u043e\u0442 1 \u0434\u043e 20.", reply_markup=MAIN_KEYBOARD)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command or Status button."""
    chat_id = update.effective_chat.id
    state = active_debates.get(chat_id)

    if not state:
        await safe_send(context.bot, chat_id, "\u2139\ufe0f \u041d\u0435\u0442 \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0439.", reply_markup=MAIN_KEYBOARD)
        return

    status = "\U0001f7e2 \u0410\u043a\u0442\u0438\u0432\u043d\u043e" if state.is_active else "\u23f9\ufe0f \u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e"
    agreed = "\u2705 \u0414\u0430" if state.is_agreed else "\u274c \u041d\u0435\u0442"

    text = (
        f"\U0001f4ca <b>\u0421\u0442\u0430\u0442\u0443\u0441 \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f</b>\n\n"
        f"\U0001f4dd \u0417\u0430\u0434\u0430\u043d\u0438\u0435: <i>{html.escape(state.task[:100])}</i>\n"
        f"\U0001f504 \u0420\u0430\u0443\u043d\u0434: {state.round_num}/{state.max_rounds}\n"
        f"\u0421\u0442\u0430\u0442\u0443\u0441: {status}\n"
        f"\u0421\u043e\u0433\u043b\u0430\u0441\u0438\u0435: {agreed}\n"
        f"\u0421\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0439: {len(state.messages)}"
    )
    await safe_send(context.bot, chat_id, text, reply_markup=MAIN_KEYBOARD)


async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages — buttons and task input."""
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    # ── Handle button presses ──
    if text == BTN_NEW_TASK:
        waiting_for_task.add(chat_id)
        await safe_send(
            context.bot, chat_id,
            "\U0001f4dd <b>\u0412\u0432\u0435\u0434\u0438 \u0437\u0430\u0434\u0430\u043d\u0438\u0435:</b>\n\n"
            "\u041d\u0430\u043f\u0440\u0438\u043c\u0435\u0440: <i>\u041d\u0430\u043f\u0438\u0448\u0438 \u0444\u0443\u043d\u043a\u0446\u0438\u044e \u0441\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u043a\u0438 \u043d\u0430 Python</i>",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    if text == BTN_STOP:
        await cmd_stop(update, context)
        return

    if text == BTN_STATUS:
        await cmd_status(update, context)
        return

    if text == BTN_HISTORY:
        await cmd_history(update, context)
        return

    # ── Waiting for task text after button press ──
    if chat_id in waiting_for_task:
        waiting_for_task.discard(chat_id)
        await start_task(chat_id, text, context)
        return

    # ── Inject into active debate ──
    state = active_debates.get(chat_id)
    if state and state.is_active:
        msg = Message(
            role="user",
            sender="\U0001f464 \u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c",
            content=text,
        )
        state.messages.append(msg)
        await safe_send(
            context.bot, chat_id,
            "\U0001f4ac \u0422\u0432\u043e\u0451 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043e. \u0410\u0433\u0435\u043d\u0442\u044b \u0443\u0447\u0442\u0443\u0442 \u0435\u0433\u043e \u0432 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0435\u043c \u043e\u0442\u0432\u0435\u0442\u0435.",
            reply_markup=MAIN_KEYBOARD,
        )
    else:
        await safe_send(
            context.bot, chat_id,
            "\u2139\ufe0f \u041d\u0435\u0442 \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0433\u043e \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f.\n\n\u041d\u0430\u0436\u043c\u0438 \U0001f4dd \u041d\u043e\u0432\u043e\u0435 \u0437\u0430\u0434\u0430\u043d\u0438\u0435 \u0447\u0442\u043e\u0431\u044b \u043d\u0430\u0447\u0430\u0442\u044c!",
            reply_markup=MAIN_KEYBOARD,
        )


async def cmd_denied(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to users who are not in ALLOWED_USER_IDS."""
    user = update.effective_user
    if not update.effective_chat or not user:
        return
    logger.info(f"Access denied for user {user.id}")
    await safe_send(
        context.bot, update.effective_chat.id,
        f"\u26d4 \u0414\u043e\u0441\u0442\u0443\u043f \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d. "
        f"\u0412\u0430\u0448 ID: <code>{user.id}</code>",
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error while processing an update", exc_info=context.error)


# ── Main ─────────────────────────────────────────────────

def main() -> None:
    """Start the bot."""
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

    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    if config.ALLOWED_USER_IDS:
        allowed = filters.User(user_id=config.ALLOWED_USER_IDS)
    else:
        allowed = filters.ALL
        print("[WARN] ALLOWED_USER_IDS is empty: anyone on Telegram can start debates and spend your API quotas.")

    app.add_handler(CommandHandler("start", cmd_start, filters=allowed))
    app.add_handler(CommandHandler("task", cmd_task, filters=allowed))
    app.add_handler(CommandHandler("stop", cmd_stop, filters=allowed))
    app.add_handler(CommandHandler("history", cmd_history, filters=allowed))
    app.add_handler(CommandHandler("rounds", cmd_rounds, filters=allowed))
    app.add_handler(CommandHandler("status", cmd_status, filters=allowed))
    app.add_handler(MessageHandler(allowed & filters.TEXT & ~filters.COMMAND, handle_user_message))
    # Anyone else gets their ID so the owner can add them to ALLOWED_USER_IDS.
    app.add_handler(MessageHandler(~allowed, cmd_denied))
    app.add_error_handler(on_error)

    print("[OK] Bot is running! Send /start in Telegram.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
