# bot.py
#
# Telegram AI Bot with Multi-Message Aggregation for Render Deployment.
# Routes user queries to OpenRouter (e.g. DeepSeek-R1, Llama 3.3, etc.).
#
# Features:
# - Dummy HTTP health check server for Render port detection.
# - Multi-message aggregation: stitches chunked or rapid sequential messages
#   before sending the full combined prompt to the AI model.
# - Supports both explicit markers ([PART 1/3]) and silence debounce (1.2s).
# - Dynamic model routing via [MODEL:vendor/model] tags with fallback models.
# - Rate limit (429) backoff timeout & retry logic with candidate model rotation.
# - Clean, user-friendly rate limit error message when quota is exhausted.
# - User authorization via ALLOWED_USER_ID.

import os
import re
import asyncio
import threading
from typing import Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

try:
    from openai import AsyncOpenAI, RateLimitError
except ImportError:
    from openai import AsyncOpenAI
    RateLimitError = Exception

# --- 1. Dummy HTTP Server for Render Port Check ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

    def log_message(self, format, *args):
        return

def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# --- 2. Environment Variables & Security ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Set your Telegram numeric ID here or via Render Environment Variables
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180.0,
    max_retries=3
)

def is_authorized(user_id: int) -> bool:
    """If ALLOWED_USER_ID is 0, protection is disabled."""
    if ALLOWED_USER_ID == 0:
        return True
    return user_id == ALLOWED_USER_ID


# --- 3. Multi-Message Aggregator ---

class UserBuffer:
    """Maintains state for assembling multiple incoming message parts from a single user."""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.raw_chunks: list[str] = []
        self.parts: Dict[int, str] = {}
        self.total_parts: Optional[int] = None
        self.last_update: Optional[Update] = None
        self.status_msg = None
        self.timer_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


class MessageAggregator:
    """
    Buffers and stitches incoming message chunks before querying the AI model.
    Handles:
    1. Explicit chunk markers: e.g. [PART 1/3] ... [PART 2/3] ... [PART 3/3]
    2. Silence debounce: combines messages received within a 1.2-second window.
    """
    def __init__(self, debounce_delay: float = 1.2, part_timeout: float = 10.0):
        self.debounce_delay = debounce_delay
        self.part_timeout = part_timeout
        self.buffers: Dict[int, UserBuffer] = {}
        self.manager_lock = asyncio.Lock()

    async def get_buffer(self, user_id: int) -> UserBuffer:
        async with self.manager_lock:
            if user_id not in self.buffers:
                self.buffers[user_id] = UserBuffer(user_id)
            return self.buffers[user_id]

    async def handle_incoming(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_authorized(user_id):
            await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
            return

        text = update.message.text or ""
        buf = await self.get_buffer(user_id)

        async with buf.lock:
            buf.last_update = update

            # Check for explicit [PART X/Y] marker
            part_match = re.match(r"^\[PART\s+(\d+)/(\d+)\]\s*(.*)", text, re.DOTALL)
            if part_match:
                part_num = int(part_match.group(1))
                total = int(part_match.group(2))
                content = part_match.group(3)

                buf.total_parts = total
                buf.parts[part_num] = content

                # Status message feedback
                status_text = f"⏳ Receiving multi-part prompt ({len(buf.parts)}/{total})..."
                if not buf.status_msg:
                    try:
                        buf.status_msg = await update.message.reply_text(status_text)
                    except Exception:
                        pass
                else:
                    try:
                        await buf.status_msg.edit_text(status_text)
                    except Exception:
                        pass

                # If all parts arrived, dispatch immediately
                if len(buf.parts) == total:
                    if buf.timer_task and not buf.timer_task.done():
                        buf.timer_task.cancel()
                    asyncio.create_task(self.dispatch(buf))
                else:
                    # Reset safety fallback timer (in case a chunk is dropped)
                    if buf.timer_task and not buf.timer_task.done():
                        buf.timer_task.cancel()
                    buf.timer_task = asyncio.create_task(self._wait_and_dispatch(buf, self.part_timeout))

            else:
                # Debounce window mode for rapid sequential messages
                buf.raw_chunks.append(text)

                if len(buf.raw_chunks) == 1:
                    try:
                        buf.status_msg = await update.message.reply_text("Thinking...")
                    except Exception:
                        pass
                else:
                    try:
                        if buf.status_msg:
                            await buf.status_msg.edit_text(f"⏳ Buffering input ({len(buf.raw_chunks)} parts)...")
                    except Exception:
                        pass

                # Reset debounce timer: wait debounce_delay seconds of silence before dispatching
                if buf.timer_task and not buf.timer_task.done():
                    buf.timer_task.cancel()
                buf.timer_task = asyncio.create_task(self._wait_and_dispatch(buf, self.debounce_delay))

    async def _wait_and_dispatch(self, buf: UserBuffer, delay: float):
        try:
            await asyncio.sleep(delay)
            await self.dispatch(buf)
        except asyncio.CancelledError:
            pass

    async def dispatch(self, buf: UserBuffer):
        async with buf.lock:
            # Combine all collected parts
            if buf.parts:
                combined_text = "".join(buf.parts[k] for k in sorted(buf.parts.keys()))
                buf.parts.clear()
                buf.total_parts = None
            elif buf.raw_chunks:
                combined_text = "\n\n".join(buf.raw_chunks)
                buf.raw_chunks.clear()
            else:
                return

            status_msg = buf.status_msg
            last_update = buf.last_update
            buf.status_msg = None

        if not combined_text.strip():
            return

        # Query model with unified prompt
        await process_ai_request(combined_text, status_msg, last_update)


# Global Aggregator Singleton
aggregator = MessageAggregator(debounce_delay=1.2, part_timeout=10.0)


# --- 4. OpenRouter Query & Response Delivery ---

async def process_ai_request(user_text: str, status_msg, update: Update):
    """
    Send the unified, assembled prompt to OpenRouter and deliver the response.
    Features:
    - Automatic rate-limit (429) detection.
    - Timeout & exponential backoff retries with live status feedback.
    - Model rotation among free-tier models on quota exhaustion.
    - Returns a clean, helpful rate limit error if all retries fail.
    """
    if not status_msg:
        try:
            status_msg = await update.message.reply_text("Thinking...")
        except Exception:
            pass
    else:
        try:
            await status_msg.edit_text("Thinking...")
        except Exception:
            pass

    # 1. Parse optional model override
    selected_model = os.getenv("AI_MODEL", "deepseek/deepseek-r1:free")
    if user_text.startswith("[MODEL:") and "]" in user_text:
        tag, clean_text = user_text.split("]", 1)
        parsed_model = tag.replace("[MODEL:", "").strip()
        if parsed_model:
            selected_model = parsed_model
        user_text = clean_text.lstrip()

    # Candidate free models pool to rotate on rate limits or failures
    models_pool = [
        selected_model,
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-small-3:free",
        "qwen/qwen-2.5-coder-32b-instruct:free"
    ]
    candidate_models = []
    for m in models_pool:
        if m not in candidate_models:
            candidate_models.append(m)

    max_retries = 3
    base_delay = 4.0
    reply = None
    last_error = None
    is_rate_limited = False

    for attempt in range(max_retries + 1):
        current_model = candidate_models[min(attempt, len(candidate_models) - 1)]
        fallbacks = [m for m in candidate_models if m != current_model][:2]

        try:
            response = await ai_client.chat.completions.create(
                model=current_model,
                extra_body={"models": fallbacks} if fallbacks else {},
                messages=[{"role": "user", "content": user_text}],
                stream=False
            )
            msg = response.choices[0].message
            reply = msg.content

            # 1. Handle native tool_calls from OpenRouter models
            if not reply and getattr(msg, "tool_calls", None):
                calls = []
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    args_raw = tc.function.arguments
                    calls.append(f"{fn_name}({args_raw})")
                reply = f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"

            # 2. Handle reasoning models (DeepSeek-R1) returning thought stream
            if not reply:
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                if not reasoning and hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
                    reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
                if reasoning:
                    reply = reasoning

            if not reply:
                reply = "No response generated."

            # Success! Break out of retry loop
            is_rate_limited = False
            last_error = None
            break

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_429 = (
                isinstance(e, RateLimitError)
                or getattr(e, "status_code", None) == 429
                or "429" in err_str
                or "rate limit" in err_str
            )

            if is_429:
                is_rate_limited = True
                if attempt < max_retries:
                    wait_sec = int(base_delay * (attempt + 1))
                    next_model = candidate_models[min(attempt + 1, len(candidate_models) - 1)]
                    status_text = (
                        f"⏳ Rate limit hit on `{current_model}` (429).\n"
                        f"Waiting {wait_sec}s before retry ({attempt + 1}/{max_retries}) using `{next_model}`..."
                    )
                    if status_msg:
                        try:
                            await status_msg.edit_text(status_text, parse_mode="Markdown")
                        except Exception:
                            try:
                                await status_msg.edit_text(status_text)
                            except Exception:
                                pass
                    await asyncio.sleep(wait_sec)
                    continue
                else:
                    break
            else:
                # Other non-retryable error
                break

    # If all retries failed or an error occurred:
    if reply is None:
        if is_rate_limited:
            err_text = (
                "⚠️ **Rate Limit Error (429)**\n\n"
                "The OpenRouter free tier limit has been exceeded.\n"
                f"Retried {max_retries} times with backoff timeout, but all attempts were rate-limited.\n\n"
                "💡 **How to resolve:**\n"
                "- Wait a few minutes for the quota window to reset.\n"
                "- In Continue, select `@GPT4Telegrambot` to use an alternative Telegram bot.\n"
                "- Or add credits to OpenRouter to raise your free model limit."
            )
        else:
            err_text = f"❌ **Error processing request:**\n`{last_error}`"

        if status_msg:
            try:
                await status_msg.edit_text(err_text, parse_mode="Markdown")
            except Exception:
                try:
                    await status_msg.edit_text(err_text)
                except Exception:
                    await update.message.reply_text(err_text)
        else:
            await update.message.reply_text(err_text)
        return

    # Normal delivery of reply
    if len(reply) <= 4000:
        if status_msg:
            await status_msg.edit_text(reply)
        else:
            await update.message.reply_text(reply)
    else:
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i+4000])


# --- 5. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
        return

    await update.message.reply_text(
        "👋 **Bot connected!**\n\n"
        "Send me any prompt to begin. Multi-message prompts are automatically buffered and assembled.",
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Feed incoming message to the aggregator."""
    await aggregator.handle_incoming(update, context)


if __name__ == "__main__":
    # Start Render keep-alive health check server
    threading.Thread(target=run_dummy_server, daemon=True).start()

    # Start Telegram bot
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
