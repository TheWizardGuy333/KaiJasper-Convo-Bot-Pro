import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API Key is missing. Please set it in the .env file.")

BASE_URL = "https://deep-image.ai/rest_api/"
