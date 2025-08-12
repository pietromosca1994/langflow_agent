from telegramagent import TelegramAgent
from webhook import BaseWebhook

if __name__ == '__main__':
    webhook=BaseWebhook() 
    telegram_agent=TelegramAgent(webhook=webhook)
    telegram_agent.listen()