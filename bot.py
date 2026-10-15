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
# - Returns raw error messages directly so client-side (api_server / ask_ai)
#   can log, inspect, and handle retries locally.
# - User authorization via ALLOWED_USER_ID.

import os
import re
import asyncio
import threading
from typing import Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

# --- Bot Version ---
BOT_VERSION = os.getenv("BOT_VERSION", "v1.2.3")

# --- 1. Dummy HTTP Server for Render Port Check ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(f"Bot is alive! [Bot {BOT_VERSION}]".encode("utf-8"))

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
       Waits until ALL parts arrive. Does NOT timeout automatically.
    2. Silence debounce: combines rapid un-chunked messages received within a 1.2s window.
    3. Manual reset: /restart or 'restart' clears any stuck/buffered state immediately.
    """
    def __init__(self, debounce_delay: float = 1.2):
        self.debounce_delay = debounce_delay
        self.buffers: Dict[int, UserBuffer] = {}
        self.manager_lock = asyncio.Lock()

    async def get_buffer(self, user_id: int) -> UserBuffer:
        async with self.manager_lock:
            if user_id not in self.buffers:
                self.buffers[user_id] = UserBuffer(user_id)
            return self.buffers[user_id]

    async def reset_user(self, user_id: int):
        """Clear all buffered message parts and cancel timers for the user."""
        buf = await self.get_buffer(user_id)
        async with buf.lock:
            if buf.timer_task and not buf.timer_task.done():
                buf.timer_task.cancel()
            buf.parts.clear()
            buf.raw_chunks.clear()
            buf.total_parts = None
            buf.status_msg = None

    async def handle_incoming(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_authorized(user_id):
            await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
            return

        text = update.message.text or ""

        # Instant reset command trigger (plain text or slash command)
        if text.strip().lower() in ("/restart", "restart", "/reset", "reset"):
            await self.reset_user(user_id)
            await update.message.reply_text(
                f"🔄 **Buffer reset!** All pending parts cleared.\n"
                f"Ready for new messages. [Bot {BOT_VERSION}]",
                parse_mode="Markdown"
            )
            return

        buf = await self.get_buffer(user_id)

        async with buf.lock:
            buf.last_update = update

            # Check for explicit [PART X/Y] marker
            part_match = re.match(r"^\[PART\s+(\d+)/(\d+)\]\s*(.*)", text, re.DOTALL)
            if part_match:
                part_num = int(part_match.group(1))
                total = int(part_match.group(2))
                content = part_match.group(3)

                # If starting fresh with Part 1, clear any leftover incomplete parts
                if part_num == 1 and buf.parts and len(buf.parts) < (buf.total_parts or 0):
                    buf.parts.clear()

                buf.total_parts = total
                buf.parts[part_num] = content

                # Dispatch ONLY when ALL parts have arrived! (No timeout, waits patiently)
                if len(buf.parts) == total:
                    if buf.timer_task and not buf.timer_task.done():
                        buf.timer_task.cancel()
                    asyncio.create_task(self.dispatch(buf))
                else:
                    # Still waiting for remaining parts. No timer, no premature dispatch.
                    pass

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
aggregator = MessageAggregator(debounce_delay=1.2)


# --- 4. OpenRouter Query & Response Delivery ---

async def process_ai_request(user_text: str, status_msg, update: Update):
    """
    Send the unified, assembled prompt to OpenRouter and deliver the response.
    Returns every error message as it is so the client (api_server / ask_ai)
    can log, inspect, and handle retries locally.
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

    try:
        # Check if a runtime model was specified via [MODEL:vendor/model] tag
        selected_model = os.getenv("AI_MODEL", "openrouter/free")
        if user_text.startswith("[MODEL:") and "]" in user_text:
            tag, clean_text = user_text.split("]", 1)
            parsed_model = tag.replace("[MODEL:", "").strip()
            if parsed_model:
                selected_model = parsed_model
            user_text = clean_text.lstrip()

        # If deprecated deepseek/deepseek-r1:free is requested, seamlessly route to openrouter/free
        if selected_model == "deepseek/deepseek-r1:free":
            selected_model = "openrouter/free"

        # Fallback candidates (OpenRouter enforces max 3 models total)
        fallback_pool = [
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free"
        ]
        fallbacks = [m for m in fallback_pool if m != selected_model][:2]

        reply = None
        last_err = None
        max_attempts = 3
        current_model = selected_model

        for attempt in range(1, max_attempts + 1):
            try:
                response = await ai_client.chat.completions.create(
                    model=current_model,
                    extra_body={"models": fallbacks} if fallbacks else {},
                    messages=[{"role": "user", "content": user_text}],
                    stream=False
                )
                msg = response.choices[0].message
                candidate = msg.content

                # 1. Handle native tool_calls from OpenRouter models
                if not candidate and getattr(msg, "tool_calls", None):
                    calls = []
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        args_raw = tc.function.arguments
                        calls.append(f"{fn_name}({args_raw})")
                    candidate = f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"

                # 2. Handle reasoning models (DeepSeek-R1) returning thought stream
                if not candidate:
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                    if not reasoning and hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
                        reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
                    if reasoning:
                        candidate = reasoning

                # 3. Strip safety evaluation boilerplate if returned by upstream model
                if candidate:
                    candidate = re.sub(
                        r"^(?:User Safety:\s*\w+\s*\n*|Response Safety:\s*\w+\s*\n*)+",
                        "",
                        candidate.strip(),
                        flags=re.IGNORECASE
                    ).strip()

                # 4. Check if valid response (not empty, not safety stub)
                is_stub = bool(re.match(r"^(?:User Safety:\s*\w+|Response Safety:\s*\w+)+$", candidate.strip(), re.IGNORECASE)) if candidate else False
                if candidate and not is_stub:
                    reply = candidate
                    break

            except Exception as e:
                last_err = e

            # If empty response, stub, or exception: automatically retry sending query again!
            if attempt < max_attempts:
                if status_msg:
                    try:
                        await status_msg.edit_text(f"Thinking (retrying {attempt + 1}/{max_attempts})...")
                    except Exception:
                        pass
                await asyncio.sleep(1.2)
                # On final retry attempt, rotate to a reliable fallback candidate
                if attempt == 2 and fallbacks:
                    current_model = fallbacks[0]

        if not reply:
            reply = "No response generated."

        # Append version tag so active deployment version is visible in every response
        reply = f"{reply}\n\n[Bot {BOT_VERSION}]"

        # Send response back to user
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

    except Exception as e:
        # Return every error message as it is so local client (api_server / ask_ai) can log and inspect it
        err_text = f"Error processing request [Bot {BOT_VERSION}]: {e}"
        if status_msg:
            try:
                await status_msg.edit_text(err_text)
            except Exception:
                await update.message.reply_text(err_text)
        else:
            await update.message.reply_text(err_text)


# --- 5. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
        return

    await update.message.reply_text(
        f"👋 **Bot connected!** [Bot `{BOT_VERSION}`]\n\n"
        "Send me any prompt to begin. Multi-part prompts are buffered until all parts arrive.\n\n"
        "💡 *If ever stuck on an incomplete multi-part prompt, send `/restart` to reset the buffer.*",
        parse_mode="Markdown"
    )

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /restart or /reset to clear buffered parts."""
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
        return
    await aggregator.reset_user(user_id)
    await update.message.reply_text(
        f"🔄 **Buffer reset!** All pending parts cleared.\n"
        f"Ready for new messages. [Bot {BOT_VERSION}]",
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
    app.add_handler(CommandHandler("restart", restart))
    app.add_handler(CommandHandler("reset", restart))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
