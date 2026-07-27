import os
from pathlib import Path

"""
Configuration constants for the hybrid PRED (RAG) heartchat-ai 
"""

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent # project root ./
MODELS_DIR = BASE_DIR/"dataset/models"
DATA_DIR = BASE_DIR/"data"
DOCS_DIR = DATA_DIR/"docs"

# Model paths
HEART_MODEL_PATH = MODELS_DIR/"heart_model.pkl"

# RAG / embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = DATA_DIR/"faiss_index.bin"

# Groq / LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Local LLM
LOCAL_LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# RAG parameters
TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Feature schema for prediction
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

URLS = ["https://pmc.ncbi.nlm.nih.gov/articles/PMC11262455",
        "https://www.heart.org/en/health-topics",
        "https://www.cdc.gov/heartdisease/about.htm",
        "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci"]

