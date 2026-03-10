import os
import google.generativeai as genai
from dotenv import dotenv_values

env_config = dotenv_values(".env")
api_key = env_config.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    print("Listing available models that support generateContent:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
else:
    print("API Key not found.")
