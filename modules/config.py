import os
from dotenv import load_dotenv

load_dotenv()

# Set cache directory
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hf_models"))
os.environ["HF_HOME"] = CACHE_DIR

# Verify cache directory exists, create if not
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Created cache directory: {CACHE_DIR}")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env or secrets.toml")

# Configure AllVoiceLab API
allvoicelab_api_key = os.getenv("ALLVOICELAB_API_KEY")
allvoicelab_api_domain = os.getenv("ALLVOICELAB_API_DOMAIN")
if not allvoicelab_api_key or not allvoicelab_api_domain:
    raise ValueError("Missing AllVoiceLab API key or domain.")

# Import and configure AllVoiceLab client
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../allvoicelab_mcp")))
from client.all_voice_lab import AllVoiceLab
from allvoicelab_mcp.tools.base import set_client

client = AllVoiceLab(allvoicelab_api_key, allvoicelab_api_domain)
set_client(client)

# Import dependencies
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=api_key)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# System prompt
SYSTEM_PROMPT = """
You're a human presenter. Based on the provided document context and the user's question, answer naturally and briefly. Add some realistic voice-like fillers such as 'umm', 'haha', or 'you know?' where they make sense. Use sentences according to the tone you will decide the tone too by seeing the answer that what should be the tone ...'. Keep the response short, engaging, and like you're presenting live on stage. and also if there would be any digit use in answer, use them in alphabets like 76 -> seventy six.
"""

# Expose configurations
__all__ = ["client", "embeddings", "genai", "SYSTEM_PROMPT", "CACHE_DIR"]