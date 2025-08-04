import os
import streamlit as st
import time
import re
from scipy.io.wavfile import write
from pydub import AudioSegment
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from modules.config import client, embeddings, genai, SYSTEM_PROMPT, CACHE_DIR
from modules.rag import get_rag_response
from modules.tts import generate_tts, clone_new_voice
import shutil

# Directories for audio files
INPUT_AUDIO_PATH = os.path.join("input_audio")
OUTPUT_AUDIO_PATH = os.path.join("output")
os.makedirs(INPUT_AUDIO_PATH, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_PATH, exist_ok=True)

# State management
if "voice_id" not in st.session_state:
    st.session_state.voice_id = None
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "is_playing_tts" not in st.session_state:
    st.session_state.is_playing_tts = False
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

st.title("Voice-Cloned RAG Chatbot")

# CSS for chat
CHAT_CSS = """
<style>
.chat-container { display: flex; flex-direction: column; gap: 10px; padding: 20px; height: 500px; overflow-y: auto; background: #f5f7fa; border-radius: 10px; margin: 10px 0; }
.message { display: flex; align-items: flex-start; gap: 10px; max-width: 80%; }
.user-message { margin-right: auto; }
.assistant-message { margin-left: auto; flex-direction: row-reverse; }
.message-bubble { padding: 12px 16px; border-radius: 15px; font-size: 15px; line-height: 1.4; }
.user-bubble { background: #007AFF; color: white; }
.assistant-bubble { background: white; color: black; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.avatar { width: 32px; height: 32px; border-radius: 50%; background: #E0E0E0; display: flex; align-items: center; justify-content: center; font-size: 16px; }
</style>
"""

# Add small wave animation HTML
WAVE_HTML = """
<style>
.wave-container { display: flex; justify-content: center; align-items: center; gap: 3px; height: 20px; width: 120px; margin: 5px auto; background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%); border-radius: 8px; padding: 3px; box-shadow: 0 2px 3px rgba(0, 0, 0, 0.07); }
.wave { width: 3px; height: 14px; background: linear-gradient(45deg, #4776E6, #8E54E9); border-radius: 2px; animation: wave 1s infinite ease-in-out; display: inline-block; }
@keyframes wave { 0%, 100% { transform: scaleY(0.5); } 40% { transform: scaleY(1.5); } }
.wave:nth-child(1) { animation-delay: 0.0s; } .wave:nth-child(2) { animation-delay: 0.1s; } .wave:nth-child(3) { animation-delay: 0.2s; } .wave:nth-child(4) { animation-delay: 0.3s; }
.wave:nth-child(5) { animation-delay: 0.4s; } .wave:nth-child(6) { animation-delay: 0.5s; } .wave:nth-child(7) { animation-delay: 0.6s; } .wave:nth-child(8) { animation-delay: 0.7s; }
.wave:nth-child(9) { animation-delay: 0.8s; } .wave:nth-child(10) { animation-delay: 0.9s; }
</style>
<div class="wave-container">
    <div class="wave"></div><div class="wave"></div><div class="wave"></div><div class="wave"></div>
    <div class="wave"></div><div class="wave"></div><div class="wave"></div><div class="wave"></div>
    <div class="wave"></div><div class="wave"></div>
</div>
"""

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def update_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def display_chat_messages():
    messages_html = ""
    for i in range(max(len(st.session_state.transcriptions), len(st.session_state.responses))):
        if i < len(st.session_state.transcriptions):
            messages_html += f"""
            <div class="message user-message">
                <div class="avatar">👤</div>
                <div class="message-bubble user-bubble">
                    {st.session_state.transcriptions[i]}
                </div>
            </div>
            """
        if i < len(st.session_state.responses):
            messages_html += f"""
            <div class="message assistant-message">
                <div class="avatar">🤖</div>
                <div class="message-bubble assistant-bubble">
                    {st.session_state.responses[i]}
                </div>
            </div>
            """
    chat_html = f"{CHAT_CSS}<div class='chat-container'>{messages_html}</div>"
    return chat_html

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None and st.session_state.pdf_text != extract_text_from_pdf(uploaded_file):
    print(f"DEBUG: PDF uploaded, size: {uploaded_file.size} bytes")
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
    st.session_state.vectorstore = update_vector_store(st.session_state.pdf_text)
    print(f"DEBUG: Extracted text length: {len(st.session_state.pdf_text)} characters")
    print("DEBUG: Vector store updated with new PDF")
    st.success("PDF uploaded and vector store updated!")

# Audio input options and processing
st.subheader("Provide Audio Input")
audio_file_upload = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"], key="audio_file_upload")
audio_input = st.audio_input("🎙️ Record your voice (up to 15s)", key="audio_input", on_change=lambda: st.session_state.update({"recorded_audio": st.session_state.audio_input}))

submit_audio = st.button("Submit Audio")

if submit_audio and not st.session_state.audio_processed:
    st.session_state.audio_processed = True
    audio_source = audio_file_upload if audio_file_upload else st.session_state.recorded_audio

    print(f"DEBUG: Checking audio source - audio_file_upload: {audio_file_upload is not None}, recorded_audio: {st.session_state.recorded_audio is not None}")
    if audio_source:
        source_type = "upload" if audio_file_upload else "recording"
        print(f"DEBUG: Audio {source_type} received, ID: {id(audio_source)}")
        
        # Save uploaded or recorded audio as WAV
        wav_path = os.path.join(INPUT_AUDIO_PATH, f"audio_input_{id(audio_source)}.wav")
        if audio_file_upload:
            audio_segment = AudioSegment.from_file(audio_file_upload)
            audio_segment.export(wav_path, format="wav")
        else:
            with open(wav_path, "wb") as f:
                f.write(audio_source.getbuffer())
        print(f"DEBUG: WAV saved to {wav_path}")

        try:
            sound = AudioSegment.from_wav(wav_path)
            mp3_path = os.path.join(INPUT_AUDIO_PATH, f"audio_input_{id(audio_source)}.mp3")
            sound.export(mp3_path, format="mp3")
            print(f"DEBUG: MP3 saved to {mp3_path}")
            # Clone only if no voice_id exists
            if not st.session_state.voice_id:
                print("DEBUG: Attempting to clone new voice...")
                new_voice_id = clone_new_voice(mp3_path, "MyClonedVoice", "User's cloned voice")
                if new_voice_id:
                    st.session_state.voice_id = new_voice_id
                    print(f"DEBUG: Voice cloned with ID: {new_voice_id}")
                else:
                    print("DEBUG: Voice cloning failed, using fallback ID: 293904040197095455")
                    st.session_state.voice_id = "293904040197095455"
            else:
                print(f"DEBUG: Reusing existing voice ID: {st.session_state.voice_id}")
            os.remove(wav_path)
            print(f"DEBUG: Temporary WAV file removed")
            st.session_state.last_audio_path = mp3_path
        except Exception as e:
            print(f"DEBUG: MP3 conversion or cloning failed: {e}")
            st.session_state.last_audio_path = None
            st.warning("Audio conversion failed, please try again or check FFmpeg installation.")

        if st.session_state.last_audio_path and os.path.exists(st.session_state.last_audio_path):
            print(f"DEBUG: Playing {source_type} audio from {st.session_state.last_audio_path}")
            with open(st.session_state.last_audio_path, "rb") as f:
                st.audio(f, format="audio/mp3")
            st.success(f"Voice {source_type}ed and cloned!" if not st.session_state.voice_id else f"Voice {source_type}ed, reusing existing clone!")
        else:
            st.warning("Audio file not available due to conversion failure.")
    else:
        st.warning("No audio input detected. Please record or upload an audio file.")

# Chat and controls
with st.container():
    st.markdown('<div class="controls-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        tts_enabled = st.toggle("Enable Speech", value=True)
    with col2:
        if st.button("Play Audio", use_container_width=True):
            if "last_audio_bytes" in st.session_state:
                print(f"DEBUG: Replay path = {st.session_state.last_audio_path}")
                print(f"DEBUG: Exists = {os.path.exists(st.session_state.last_audio_path)}")

                if os.path.exists(st.session_state.last_audio_path):
                    st.audio(st.session_state["last_audio_bytes"], format="audio/mp3")
                    print("DEBUG: Replaying audio.")
                    st.success("Replaying audio.")
                else:
                    st.warning("Audio path stored is missing from disk.")
            else:
                st.warning("No audio has been played yet.")
    st.markdown('</div>', unsafe_allow_html=True)

container = st.container()
with container:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("Recording or upload handled above")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.transcriptions = []
            st.session_state.responses = []
            st.session_state.voice_id = None
            st.session_state.pdf_text = ""
            st.session_state.vectorstore = None
            st.session_state.is_playing_tts = False
            st.session_state.last_audio_path = None
            st.session_state.audio_processed = False
            st.session_state.recorded_audio = None

    if st.session_state.voice_id and st.session_state.vectorstore and st.session_state.last_audio_path:
        transcription = st.text_input("Enter your query or transcribe the recording:", key="transcription_input", value="")
        if st.button("Submit Query", key="submit_query"):
            st.session_state.transcriptions.append(transcription)
            response = get_rag_response(transcription, tone="informative", context=st.session_state.pdf_text)
            st.session_state.responses.append(response)
            if tts_enabled and st.session_state.voice_id:
                st.session_state.is_playing_tts = True
                tts_result = generate_tts(response, voice_id=st.session_state.voice_id)
                print(f"DEBUG: TTS result: {tts_result}")
                if hasattr(tts_result, 'text'):
                    tts_text = tts_result.text
                    match = re.search(r"file saved at:\s*([^\n]+)", tts_text)
                    mp3_path = match.group(1).strip() if match else None
                else:
                    mp3_path = None
                print(f"DEBUG: Extracted MP3 path: {mp3_path}")
                if mp3_path and os.path.exists(mp3_path):
                    local_mp3_path = os.path.join(OUTPUT_AUDIO_PATH, os.path.basename(mp3_path))
                    if not os.path.samefile(mp3_path, local_mp3_path):
                        try:
                            shutil.copy(mp3_path, local_mp3_path)
                            print(f"DEBUG: Copied MP3 to {local_mp3_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to copy MP3: {e}")
                    else:
                        print(f"DEBUG: MP3 already in output directory: {local_mp3_path}")
                    if os.path.exists(local_mp3_path):
                        print(f"DEBUG: Playing TTS audio from {local_mp3_path}")
                        print(f"DEBUG: os.path.exists: {os.path.exists(local_mp3_path)}")
                        st.session_state.last_audio_path = local_mp3_path
                        print(f"DEBUG: last_audio_path in session: {st.session_state.last_audio_path}")
                        print(f"DEBUG: Playing TTS audio from {local_mp3_path}")
                        
                        with open(local_mp3_path, "rb") as f:
                            audio_bytes = f.read()
                        st.session_state["last_audio_bytes"] = audio_bytes
                        st.audio(audio_bytes, format="audio/mp3")
                        print("DEBUG: TTS audio played successfully")
                        st.success("TTS generated and audio played!")
                        st.session_state.is_playing_tts = True
                    else:
                        st.warning("Copied audio file not found or generation failed.")
                else:
                    st.warning("TTS audio file not found or generation failed.")
            st.rerun() if st.session_state.is_playing_tts else None  # Avoid full rerun to reduce MediaFileStorageError

    st.components.v1.html(display_chat_messages(), height=600, scrolling=True)