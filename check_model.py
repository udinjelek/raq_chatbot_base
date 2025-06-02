import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure API
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
for model in genai.list_models():
    print(model.name, model.supported_generation_methods)
