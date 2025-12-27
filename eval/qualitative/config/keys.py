import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_GPT4_KEY = os.getenv("AZURE_API_KEY")
ENDPOINT_OPENAI_GPT4 = os.getenv("AZURE_API_BASE")
CHAT_VERSION = "2024-08-01-preview"  # Update if needed
CHAT_DEPLOYMENT_NAME = "gpt-4o"  # Replace with your deployed model name
GEMINI_API = os.getenv("GEMINI_API")
AZURE_API_MINI = os.getenv("AZURE_API_MINI")
MINI_DEPLOYMENT_NAME = os.getenv("MINI_DEPLOYMENT_NAME")
MINI_VERSION = os.getenv("MINI_VERSION")
