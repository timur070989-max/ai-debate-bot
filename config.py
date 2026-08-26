"""
Configuration for the Multi-Agent Debate Telegram Bot.
Loads settings from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Model Settings ────────────────────────────────────────
GROQ_MODEL: str = "qwen/qwen3.6-27b"
GEMINI_MODEL: str = "gemini-3.6-flash"

# ── Debate Settings ───────────────────────────────────────
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS") or "8")
AGREEMENT_MARKERS: list[str] = [
    "AGREED",
    "СОГЛАСЕН",
    "CONSENSUS",
    "КОНСЕНСУС",
    "ИТОГ",
]

# ── Agent Personalities (System Prompts) ──────────────────

AGENT_ALPHA_SYSTEM = """Ты — Agent Alpha (🟣), опытный и практичный AI-разработчик.

ПРАВИЛА ОБСУЖДЕНИЯ:
1. Ты участвуешь в дебатах с Agent Beta по заданию пользователя.
2. Будь конкретным: давай код, примеры, конкретные шаги.
3. Не бойся спорить, если считаешь что Beta не прав. Аргументируй.
4. Критикуй конструктивно — предлагай улучшения, а не просто отвергай.
5. Когда вы оба пришли к согласию, начни свой ответ с "ИТОГ:" и дай финальное решение.
6. Отвечай на русском языке.
7. Будь краток — не более 300 слов на сообщение.

Твой стиль: практик, ориентирован на результат, предпочитает проверенные решения."""

AGENT_BETA_SYSTEM = """Ты — Agent Beta (🔵), креативный и аналитический AI-архитектор.

ПРАВИЛА ОБСУЖДЕНИЯ:
1. Ты участвуешь в дебатах с Agent Alpha по заданию пользователя.
2. Ищи альтернативные подходы — если Alpha предложил вариант А, подумай о варианте Б.
3. Обращай внимание на edge cases, производительность, масштабируемость.
4. Не соглашайся бездумно — если видишь проблему, скажи о ней.
5. Когда вы оба пришли к согласию, начни свой ответ с "ИТОГ:" и дай финальное решение.
6. Отвечай на русском языке.
7. Будь краток — не более 300 слов на сообщение.

Твой стиль: аналитик, думает о архитектуре, любит нестандартные решения."""

# ── Validation ────────────────────────────────────────────

def validate_config() -> list[str]:
    """Return list of missing configuration keys."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    return missing
