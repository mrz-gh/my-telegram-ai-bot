import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

# 1. Dummy HTTP server to satisfy Render's port check
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

    def log_message(self, format, *args):
        return  # Suppress request logging to keep logs clean

def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# 2. Telegram & OpenRouter Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180.0,
    max_retries=3
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot connected! Send me any prompt to begin.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    status_msg = await update.message.reply_text("Thinking...")

    try:
        response = await ai_client.chat.completions.create(
            model="stealth/ox-alpha",
            extra_body={
                "models": [
                    "deepseek/deepseek-r1:free",
                    "meta-llama/llama-3.3-70b-instruct:free"
                ]
            },
            messages=[{"role": "user", "content": user_text}],
            stream=False
        )
        reply = response.choices[0].message.content or "No response generated."

        if len(reply) <= 4000:
            await status_msg.edit_text(reply)
        else:
            await status_msg.delete()
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i:i+4000])

    except Exception as e:
        await status_msg.edit_text(f"Error processing request: {e}")

if __name__ == "__main__":
    # Start the dummy web server in a background thread
    threading.Thread(target=run_dummy_server, daemon=True).start()

    # Start the Telegram Bot
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
