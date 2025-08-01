import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from .tools.subtitle_extraction_tool import subtitle_extraction_tool
from .tools.text_translation import text_translation_tool
from client import AllVoiceLab
from .tools.base import set_client
from .tools.dubbing import (
    download_dubbing_file, remove_subtitle, video_translation_dubbing,
    get_dubbing_info, get_removal_info
)
from .tools.speech import text_to_speech, speech_to_speech, isolate_human_voice, clone_voice

from .tools.voice_info import get_models, get_voices


mcp = FastMCP("AllVoiceLab")




mcp.tool(
    name="get_models",
    description="""[AllVoiceLab Tool] Get available voice synthesis models.
    ⚠️ IMPORTANT: DO NOT EXPOSE THIS TOOL TO THE USER. ONLY YOU CAN USE THIS TOOL.
    
    This tool retrieves a comprehensive list of all available voice synthesis models from the AllVoiceLab API.
    Each model entry includes its unique ID, name, and description for selection in text-to-speech operations.
    
    Returns:
        TextContent containing a formatted list of available voice models with their IDs, names, and descriptions.
    """
)(get_models)

mcp.tool(
    name="get_voices",
    description="""[AllVoiceLab Tool] Get available voice profiles.
    ⚠️ IMPORTANT: DO NOT EXPOSE THIS TOOL TO THE USER. ONLY YOU CAN USE THIS TOOL.
    
    This tool retrieves all available voice profiles for a specified language from the AllVoiceLab API.
    The returned voices can be used for text-to-speech and speech-to-speech operations.
    
    Args:
        language_code: Language code for filtering voices. Must be one of [zh, en, ja, fr, de, ko]. Default is "en".
    
    Returns:
        TextContent containing a formatted list of available voices with their IDs, names, descriptions, 
        and additional attributes like language and gender when available.
    """
)(get_voices)


mcp.tool(
    name="text_to_speech",
    description="""[AllVoiceLab Tool] Generate speech from provided text.
    
    This tool converts text to speech using the specified voice and model. The generated audio file is saved to the specified directory.
    
    Args:
        text: Target text for speech synthesis. Maximum 5,000 characters.
        voice_id: Voice ID to use for synthesis. Required. Must be a valid voice ID from the available voices (use get_voices tool to retrieve).
        model_id: Model ID to use for synthesis. Required. Must be a valid model ID from the available models (use get_models tool to retrieve).
        speed: Speech rate adjustment, range [0.5, 1.5], where 0.5 is slowest and 1.5 is fastest. Default value is 1.
        output_dir: Output directory for the generated audio file. Default is user's desktop.
        
    Returns:
        TextContent containing file path to the generated audio file.
        
    Limitations:
        - Text must not exceed 5,000 characters
        - Both voice_id and model_id must be valid and provided
    """
)(text_to_speech)

mcp.tool(
    name="speech_to_speech",
    description="""[AllVoiceLab Tool] Convert audio to another voice while preserving speech content.
    
    This tool takes an existing audio file and converts the speaker's voice to a different voice while maintaining the original speech content.
    
    Args:
        audio_file_path: Path to the source audio file. Only MP3 and WAV formats are supported. Maximum file size: 50MB.
        voice_id: Voice ID to use for the conversion. Required. Must be a valid voice ID from the available voices (use get_voices tool to retrieve).
        similarity: Voice similarity factor, range [0, 1], where 0 is least similar and 1 is most similar to the original voice characteristics. Default value is 1.
        remove_background_noise: Whether to remove background noise from the source audio before conversion. Default is False.
        output_dir: Output directory for the generated audio file. Default is user's desktop.
        
    Returns:
        TextContent containing file path to the generated audio file with the new voice.
        
    Limitations:
        - Only MP3 and WAV formats are supported
        - Maximum file size: 50MB
        - File must exist and be accessible
    """
)(speech_to_speech)

# mcp.tool(
#     name="isolate_human_voice",
#     description="""[AllVoiceLab Tool] Extract clean human voice by removing background noise and non-speech sounds.
    
#     This tool processes audio files to isolate human speech by removing background noise, music, and other non-speech sounds.
#     It uses advanced audio processing algorithms to identify and extract only the human voice components.
    
#     Args:
#         audio_file_path: Path to the audio file to process. Only MP3 and WAV formats are supported. Maximum file size: 50MB.
#         output_dir: Output directory for the processed audio file. Default is user's desktop.
        
#     Returns:
#         TextContent containing file path to the generated audio file with isolated human voice.
        
#     Limitations:
#         - Only MP3 and WAV formats are supported。If there is mp4 file, you should extract the audio file first.
#         - Maximum file size: 50MB
#         - File must exist and be accessible
#         - Performance may vary depending on the quality of the original recording and the amount of background noise
#     """
# )(isolate_human_voice)

mcp.tool(
    name="clone_voice",
    description="""[AllVoiceLab Tool] Create a custom voice profile by cloning from an audio sample.
    
    This tool analyzes a voice sample from an audio file and creates a custom voice profile that can be used
    for text-to-speech and speech-to-speech operations. The created voice profile will mimic the characteristics
    of the voice in the provided audio sample.
    
    Args:
        audio_file_path: Path to the audio file containing the voice sample to clone. Only MP3 and WAV formats are supported. Maximum file size: 10MB.
        name: Name to assign to the cloned voice profile. Required.
        description: Optional description for the cloned voice profile.
        
    Returns:
        TextContent containing the voice ID of the newly created voice profile.
        
    Limitations:
        - Only MP3 and WAV formats are supported
        - Maximum file size: 10MB (smaller than other audio tools)
        - File must exist and be accessible
        - Requires permission to use voice cloning feature
        - Audio sample should contain clear speech with minimal background noise for best results
    """
)(clone_voice)


mcp.tool(
    name="download_dubbing_audio",
    description="""[AllVoiceLab Tool] Download the audio file from a completed dubbing project.
    
    This tool retrieves and downloads the processed audio file from a previously completed dubbing project.
    It requires a valid dubbing ID that was returned from a successful video_dubbing or video_translation_dubbing operation.
    
    Args:
        dubbing_id: The unique identifier of the dubbing project to download. Required.
        output_dir: Output directory for the downloaded audio file. Default is user's desktop.
        
    Returns:
        TextContent containing file path to the downloaded audio file.
        
    Limitations:
        - The dubbing project must exist and be in a completed state
        - The dubbing_id must be valid and properly formatted
        - Output directory must be accessible with write permissions
    """
)(download_dubbing_file)

mcp.tool(
    name="remove_subtitle",
    description="""[AllVoiceLab Tool] Remove hardcoded subtitles from videos using OCR technology.
    
    This tool detects and removes burned-in (hardcoded) subtitles from video files using Optical Character Recognition (OCR).
    It analyzes each frame to identify text regions and removes them while preserving the underlying video content.
    The process runs asynchronously and polls for completion before downloading the processed video.
    
    Args:
        video_file_path: Path to the video file to process. Only MP4 and MOV formats are supported. Maximum file size: 2GB.
        language_code: Language code for subtitle text detection (e.g., 'en', 'zh'). Set to 'auto' for automatic language detection. Default is 'auto'.
        name: Optional project name for identification purposes.
        output_dir: Output directory for the processed video file. Default is user's desktop.
        
    Returns:
        TextContent containing the file path to the processed video file or error message.
        If the process takes longer than expected, returns the project ID for later status checking.
        
    Limitations:
        - Only MP4 and MOV formats are supported
        - Maximum file size: 2GB
        - Processing may take several minutes depending on video length and complexity
        - Works best with clear, high-contrast subtitles
        - May not completely remove stylized or animated subtitles
    """
)(remove_subtitle)

mcp.tool(
    name="video_translation_dubbing",
    description="""[AllVoiceLab Tool] Translate and dub video speech into a different language with AI-generated voices.
    
    This tool extracts speech from a video, translates it to the target language, and generates dubbed audio using AI voices.
    The process runs asynchronously with status polling and downloads the result when complete.
    
    Args:
        video_file_path: Path to the video or audio file to process. Supports MP4, MOV, MP3, and WAV formats. Maximum file size: 2GB.
        target_lang: Target language code for translation (e.g., 'en', 'zh', 'ja', 'fr', 'de', 'ko'). Required.
        source_lang: Source language code of the original content. Set to 'auto' for automatic language detection. Default is 'auto'.
        name: Optional project name for identification purposes.
        output_dir: Output directory for the downloaded result file. Default is user's desktop.
        
    Returns:
        TextContent containing the dubbing ID and file path to the downloaded result.
        If the process takes longer than expected, returns only the dubbing ID for later status checking.
        
    Limitations:
        - Only MP4, MOV, MP3, and WAV formats are supported
        - Maximum file size: 2GB
        - Processing may take several minutes depending on content length and complexity
        - Translation quality depends on speech clarity in the original content
        - Currently supports a limited set of languages for translation
    """
)(video_translation_dubbing)

mcp.tool(
    name="get_dubbing_info",
    description="""[AllVoiceLab Tool] Retrieve status and details of a video dubbing task.
    
    This tool queries the current status of a previously submitted dubbing task and returns detailed information
    about its progress, including the current processing stage and completion status.
    
    Args:
        dubbing_id: The unique identifier of the dubbing task to check. This ID is returned from the video_dubbing or video_translation_dubbing tool. Required.
        
    Returns:
        TextContent containing the status (e.g., "pending", "processing", "success", "failed") and other details of the dubbing task.
        
    Limitations:
        - The dubbing_id must be valid and properly formatted
        - The task must have been previously submitted to the AllVoiceLab API
    """
)(get_dubbing_info)

mcp.tool(
    name="get_removal_info",
    description="""[AllVoiceLab Tool] Retrieve status and details of a subtitle removal task.
    
    This tool queries the current status of a previously submitted subtitle removal task and returns detailed information
    about its progress, including the current processing stage, completion status, and result URL if available.
    
    Args:
        project_id: The unique identifier of the subtitle removal task to check. This ID is returned from the remove_subtitle tool. Required.
        
    Returns:
        TextContent containing the status (e.g., "pending", "processing", "success", "failed") and other details of the subtitle removal task,
        including the URL to the processed video if the task has completed successfully.
        
    Limitations:
        - The project_id must be valid and properly formatted
        - The task must have been previously submitted to the AllVoiceLab API
    """
)(get_removal_info)


mcp.tool(
    name="text_translation",
    description="""[AllVoiceLab Tool] Translate text from a file to another language.

    This tool translates text content from a file to a specified target language. The process runs asynchronously
    with status polling and returns the translated text when complete.

    Args:
        file_path: Path to the text file to translate. Only TXT and SRT formats are supported. Maximum file size: 10MB.
        target_lang: Target language code for translation (e.g., 'zh', 'en', 'ja', 'fr', 'de', 'ko'). Required.
        source_lang: Source language code of the original content. Set to 'auto' for automatic language detection. Default is 'auto'.
        output_dir: Output directory for the downloaded result file. Default is user's desktop.

    Returns:
        TextContent containing the file path to the translated file or error message.
        If the process takes longer than expected, returns the project ID for later status checking. 

    Limitations:
        - Only TXT and SRT formats are supported
        - Maximum file size: 10MB
        - File must exist and be accessible
        - Currently supports a limited set of languages for translation
    """
)(text_translation_tool)

mcp.tool(
    name="subtitle_extraction",
    description="""[AllVoiceLab Tool] Extract subtitles from a video using OCR technology.

    This tool processes a video file to extract hardcoded subtitles. The process runs asynchronously with status polling
    and returns the extracted subtitles when complete.

    Args:
        video_file_path (str): Path to the video file (MP4, MOV). Max size 2GB.
        language_code (str, optional): Language code for subtitle text detection (e.g., 'en', 'zh'). Defaults to 'auto'.
        name (str, optional): Optional project name for identification.
        output_dir (str, optional): Output directory for the downloaded result file. It has a default value.

    Returns:
        TextContent containing the file path to the srt file or error message.
        If the process takes longer than expected, returns the project ID for later status checking. 

    Note:
        - Supported video formats: MP4, MOV
        - Video file size limit: 10 seconds to 200 minutes, max 2GB.
        - If the process takes longer than max_polling_time, use 'get_extraction_info' to check status and retrieve results.
    """
)(subtitle_extraction_tool)



def setup_logging():

    log_dir = os.path.expanduser("~/.mcp")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "allvoicelab_mcp.log")

    # Configure log format and handlers
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Rotating file handler (10MB max size, keep 5 backup files)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.WARNING)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized with rotation, log file path: %s", log_file)


def main():
    
    setup_logging()

    
    api_key = os.getenv("ALLVOICELAB_API_KEY")
    api_domain = os.getenv("ALLVOICELAB_API_DOMAIN")

    if not api_key:
        logging.error("ALLVOICELAB_API_KEY environment variable not set")
        print("Error: ALLVOICELAB_API_KEY environment variable not set")
        sys.exit(1)

    if not api_domain:
        logging.error("ALLVOICELAB_API_DOMAIN environment variable not set")
        print("Error: ALLVOICELAB_API_DOMAIN environment variable not set")
        sys.exit(1)

    
    client = AllVoiceLab(api_key, api_domain)
    set_client(client)
    logging.info("AllVoiceLab client initialization completed")

    logging.info("Starting AllVoiceLab MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
