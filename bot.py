import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

# --- 1. Dummy HTTP Server for Render Port Check ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
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
# Example: "123456789"
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180.0,
    max_retries=3
)

# Helper function to check authorization
def is_authorized(user_id: int) -> bool:
    # If ALLOWED_USER_ID is 0, protection is disabled
    if ALLOWED_USER_ID == 0:
        return True
    return user_id == ALLOWED_USER_ID

# --- 3. Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
        return

    await update.message.reply_text("Bot connected! Send me any prompt to begin.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text(f"⛔ Unauthorized. Your User ID is: `{user_id}`", parse_mode="Markdown")
        return

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
    threading.Thread(target=run_dummy_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
