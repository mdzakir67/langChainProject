from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# List all available models with details
models = client.models.list()

print(f"Total models available: {len(models.data)}\n")
for model in models.data:
    print(f"ID: {model.id}")
    print(f"Created: {model.created}")
    print(f"Owned by: {model.owned_by}")
    print("-" * 50)