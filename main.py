import os
from modules.config import client, vectorstore, embeddings, genai, SYSTEM_PROMPT, CACHE_DIR
from modules.rag import get_rag_response
from modules.tts import generate_tts, clone_new_voice

# Verify cache directory
print(f"🔍 Checking cache directory: {CACHE_DIR}")
if os.path.exists(CACHE_DIR):
    print(f"✅ Cache directory exists with contents: {os.listdir(CACHE_DIR)}")
else:
    print("❌ Cache directory not found (should have been created by config.py)")

# Clone a new voice first to get the voice_id
print("🎙️ Cloning new voice...")
new_voice_id = clone_new_voice("faiq.mp3", "MyCustomVoice", "This is my cloned voice")
if new_voice_id:
    print(f"🎙️ New voice ID: {new_voice_id}")
else:
    print("🎙️ Using default voice ID due to cloning failure.")
    new_voice_id = "293904040197095455"  # Fallback

# Sample query and tone
query = "What is the main topic of the document?"
tone = "informative"

# Run RAG pipeline
print("🟡 Starting RAG pipeline...")
text = get_rag_response(query, tone)
print(f"🟢 Answer: {text}")

# Generate TTS with the new voice_id
print("🔊 Generating text-to-speech...")
audio_path = generate_tts(text, voice_id=new_voice_id)
print(f"🔊 Audio saved to: {audio_path}")