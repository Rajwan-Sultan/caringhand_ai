import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME= os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)
    QUERY_WEIGHT = float(os.getenv("QUERY_WEIGHT", 0.8))
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

settings = Settings()