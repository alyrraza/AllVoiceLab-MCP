from .config import client
from allvoicelab_mcp.tools.speech import text_to_speech, clone_voice

def generate_tts(text, voice_id="293904040197095455", model_id="tts-multilingual", output_dir="D:/ML/LLm/Voice rag/output", speed=1):
    try:
        result = text_to_speech(text, voice_id, model_id, output_dir=output_dir, speed=speed)
        print(f"TTS result: {result}")
        return result
    except Exception as e:
        print(f"ERROR: TTS failed: {str(e)}")
        return f"Synthesis failed, tool temporarily unavailable (Error: {str(e)})"

def clone_new_voice(audio_file_path, voice_name, description):
    try:
        result = clone_voice(audio_file_path, voice_name, description)
        # Parse the new voice_id from the response
        if result and hasattr(result, 'text') and "Voice cloning completed" in result.text:
            voice_id = result.text.split("Your new voice ID is: ")[1].split("\n")[0]
            return voice_id
        return None
    except Exception as e:
        print(f"ERROR: Cloning failed: {str(e)}")
        return None

__all__ = ["generate_tts", "clone_new_voice"]