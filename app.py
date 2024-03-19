from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv("var.env")
mistral_api_key = os.getenv("mistral_api_key")
# If mistral_api_key is not passed, default behavior is to use the `MISTRAL_API_KEY` environment variable.
chat = ChatMistralAI(mistral_api_key=mistral_api_key)

messages = [HumanMessage(content="knock knock")]
chat.invoke(messages)
