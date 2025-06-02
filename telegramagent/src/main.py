from telegramagent import TelegramAgent

if __name__ == '__main__':
    # print("Bot is running and waiting for messages...")
    # app.run_polling()
    # response=call_langflow_api("hello world!")
    telegram_agent=TelegramAgent()
    telegram_agent.run()