import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Connect to OpenRouter
ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180.0,   # Generous timeout for long thinking models
    max_retries=3
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot connected! Send me any prompt to begin.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    status_msg = await update.message.reply_text("Thinking...")

    try:
        response = await ai_client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            extra_body={
                # Auto-fallback if the free endpoint is busy
                "models": [
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "qwen/qwen-2.5-72b-instruct:free"
                ]
            },
            messages=[{"role": "user", "content": user_text}],
            stream=False
        )
        reply = response.choices[0].message.content or "No response generated."

        # Telegram max message length is 4096 characters
        if len(reply) <= 4000:
            await status_msg.edit_text(reply)
        else:
            await status_msg.delete()
            for i in range(0, len(reply), 4000):
                await update.message.reply_text(reply[i:i+4000])

    except Exception as e:
        await status_msg.edit_text(f"Error processing request: {e}")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot started...")
    app.run_polling()