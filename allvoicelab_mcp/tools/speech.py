import logging

from mcp.types import TextContent

from client.all_voice_lab import VoiceCloneNoPermissionError
from .base import get_client
from .utils import validate_audio_file, validate_output_directory, create_error_response, create_success_response


def text_to_speech(
    text: str,
    voice_id: str,
    model_id: str,
    speed: int = 1,
    output_dir: str = None
) -> TextContent:
    """
    Convert text to speech
    
    Args:
        text: Target text for speech synthesis. Maximum 5,000 characters.
        voice_id: Voice ID to use for synthesis. Required. Must be a valid voice ID from the available voices (use get_voices tool to retrieve).
        model_id: Model ID to use for synthesis. Required. Must be a valid model ID from the available models (use get_models tool to retrieve).
        speed: Speech rate adjustment, range [0.5, 1.5], where 0.5 is slowest and 1.5 is fastest. Default value is 1.
        output_dir: Output directory for the generated audio file. Default is user's desktop.
        
    Returns:
        TextContent: Contains the file path to the generated audio file.
    """
    all_voice_lab = get_client()
    output_dir = all_voice_lab.get_output_path(output_dir)
    logging.info(f"Tool called: text_to_speech, voice_id: {voice_id}, model_id: {model_id}, speed: {speed}")
    logging.info(f"Output directory: {output_dir}")

    # Validate parameters
    if not text:
        logging.warning("Text parameter is empty")
        return TextContent(
            type="text",
            text="text parameter cannot be empty"
        )
    if len(text) > 5000:
        logging.warning(f"Text parameter exceeds maximum length: {len(text)} characters")
        return TextContent(
            type="text",
            text="text parameter cannot exceed 5,000 characters"
        )
    if not voice_id:
        logging.warning("voice_id parameter is empty")
        return TextContent(
            type="text",
            text="voice_id parameter cannot be empty"
        )
    # Validate voice_id is numeric
    if not voice_id.isdigit():
        logging.warning(f"Invalid voice_id format: {voice_id}, not a numeric value")
        return TextContent(
            type="text",
            text="voice_id parameter must be a numeric value"
        )
    if not model_id:
        logging.warning("model_id parameter is empty")
        return TextContent(
            type="text",
            text="model_id parameter cannot be empty"
        )

    

    # Validate model_id against available models
    try:
        logging.info(f"Validating model_id: {model_id}")
        model_resp = all_voice_lab.get_supported_voice_model()
        available_models = model_resp.models
        valid_model_ids = [model.model_id for model in available_models]

        if model_id not in valid_model_ids:
            logging.warning(f"Invalid model_id: {model_id}, available models: {valid_model_ids}")
            return TextContent(
                type="text",
                text=f"Invalid model_id: {model_id}. Please use a valid model ID."
            )
        logging.info(f"Model ID validation successful: {model_id}")
    except Exception as e:
        logging.error(f"Failed to validate model_id: {str(e)}")
        # Continue with the process even if validation fails
        # to maintain backward compatibility

    try:
        logging.info(f"Starting text-to-speech processing, text length: {len(text)} characters")
        file_path = all_voice_lab.text_to_speech(text, voice_id, model_id, output_dir, speed)
        logging.info(f"Text-to-speech successful, file saved at: {file_path}")
        return TextContent(
            type="text",
            text=f"Speech generation completed, file saved at: {file_path}\n"
        )
    except Exception as e:
        logging.error(f"Text-to-speech failed: {str(e)}")
        return TextContent(
            type="text",
            text=f"Synthesis failed, tool temporarily unavailable"
        )


def speech_to_speech(
    audio_file_path: str,
    voice_id: str,
    similarity: float = 1,
    remove_background_noise: bool = False,
    output_dir: str = None
) -> TextContent:
    """
    Convert audio to another voice while preserving speech content
    
    Args:
        audio_file_path: Path to the source audio file. Only MP3 and WAV formats are supported. Maximum file size: 50MB.
        voice_id: Voice ID to use for the conversion. Required. Must be a valid voice ID from the available voices (use get_voices tool to retrieve).
        similarity: Voice similarity factor, range [0, 1], where 0 is least similar and 1 is most similar to the original voice characteristics. Default value is 1.
        remove_background_noise: Whether to remove background noise from the source audio before conversion. Default is False.
        output_dir: Output directory for the generated audio file. Default is user's desktop.
        
    Returns:
        TextContent: Contains the file path to the generated audio file with the new voice.
    """
    all_voice_lab = get_client()
    output_dir = all_voice_lab.get_output_path(output_dir)
    logging.info(f"Tool called: speech_to_speech, voice_id: {voice_id}, similarity: {similarity}")
    logging.info(f"Audio file path: {audio_file_path}, remove background noise: {remove_background_noise}")
    logging.info(f"Output directory: {output_dir}")

    # Validate audio file
    is_valid, error_message = validate_audio_file(audio_file_path)
    if not is_valid:
        return create_error_response(error_message)

    # Validate voice_id parameter
    if not voice_id:
        logging.warning("voice_id parameter is empty")
        return create_error_response("voice_id parameter cannot be empty")

    # Validate voice_id format (basic check)
    if not isinstance(voice_id, str) or len(voice_id.strip()) == 0:
        logging.warning(f"Invalid voice_id format: {voice_id}")
        return create_error_response("Invalid voice_id format")

    # Validate similarity range
    if similarity < 0 or similarity > 1:
        logging.warning(f"Similarity parameter {similarity} is out of range [0, 1]")
        return create_error_response("similarity parameter must be between 0 and 1")

    # Validate and create output directory
    is_valid, error_message = validate_output_directory(output_dir)
    if not is_valid:
        return create_error_response(error_message)

    try:
        logging.info("Starting speech conversion processing")
        file_path = all_voice_lab.speech_to_speech(audio_file_path, voice_id, output_dir, similarity,
                                                   remove_background_noise)
        logging.info(f"Speech conversion successful, file saved at: {file_path}")
        return create_success_response(f"Audio conversion completed, file saved at: {file_path}\n")
    except FileNotFoundError as e:
        logging.error(f"Audio file does not exist: {audio_file_path}, error: {str(e)}")
        return create_error_response(f"Audio file does not exist: {audio_file_path}")
    except Exception as e:
        logging.error(f"Speech conversion failed: {str(e)}")
        return create_error_response("Conversion failed, tool temporarily unavailable")


def isolate_human_voice(
    audio_file_path: str,
    output_dir: str = None
) -> TextContent:
    """
    Extract clean human voice by removing background noise and non-speech sounds
    
    Args:
        audio_file_path: Path to the audio file to process. Only MP3 and WAV formats are supported. Maximum file size: 50MB.
        output_dir: Output directory for the processed audio file. Default is user's desktop.
        
    Returns:
        TextContent: Contains the file path to the generated audio file with isolated human voice.
    """
    all_voice_lab = get_client()
    output_dir = all_voice_lab.get_output_path(output_dir)
    logging.info(f"Tool called: isolate_human_voice")
    logging.info(f"Audio file path: {audio_file_path}")
    logging.info(f"Output directory: {output_dir}")

    # Validate audio file
    is_valid, error_message = validate_audio_file(audio_file_path)
    if not is_valid:
        return create_error_response(error_message)

    # Validate and create output directory
    is_valid, error_message = validate_output_directory(output_dir)
    if not is_valid:
        return create_error_response(error_message)

    try:
        logging.info("Starting human voice isolation processing")
        file_path = all_voice_lab.audio_isolation(audio_file_path, output_dir)
        logging.info(f"Human voice isolation successful, file saved at: {file_path}")
        return create_success_response(f"Voice isolation completed, file saved at: {file_path}\n")
    except FileNotFoundError as e:
        logging.error(f"Audio file does not exist: {audio_file_path}, error: {str(e)}")
        return create_error_response(f"Audio file does not exist: {audio_file_path}")
    except Exception as e:
        logging.error(f"Human voice isolation failed: {str(e)}")
        return create_error_response("Voice isolation failed, tool temporarily unavailable")


def clone_voice(
    audio_file_path: str,
    name: str,
    description: str = None
) -> TextContent:
    """
    Create a custom voice profile by cloning from an audio sample
    
    Args:
        audio_file_path: Path to the audio file containing the voice sample to clone. Only MP3 and WAV formats are supported. Maximum file size: 10MB.
        name: Name to assign to the cloned voice profile. Required.
        description: Optional description for the cloned voice profile.
        
    Returns:
        TextContent: Contains the voice ID of the newly created voice profile.
    """
    all_voice_lab = get_client()
    logging.info(f"Tool called: clone_voice")
    logging.info(f"Audio file path: {audio_file_path}")
    logging.info(f"Voice name: {name}")
    if description:
        logging.info(f"Voice description: {description}")

    # Validate audio file, using 10MB size limit
    is_valid, error_message = validate_audio_file(audio_file_path, max_size_mb=10)
    if not is_valid:
        return create_error_response(error_message)

    # Validate name parameter
    if not name:
        logging.warning("Name parameter is empty")
        return create_error_response("name parameter cannot be empty")

    try:
        logging.info("Starting voice cloning process")
        voice_id = all_voice_lab.add_voice(name, audio_file_path, description)
        logging.info(f"Voice cloning successful, voice ID: {voice_id}")
        return TextContent(
            type="text",
            text=f"Voice cloning completed. Your new voice ID is: {voice_id}\n"
        )
    except VoiceCloneNoPermissionError as e:
        logging.error(f"Voice cloning failed: {str(e)}")
        return TextContent(
            type="text",
            text=f"Voice cloning failed, you don't have permission to clone voice. Please contact AllVoiceLab com."
        )
    except FileNotFoundError as e:
        logging.error(f"Audio file does not exist: {audio_file_path}, error: {str(e)}")
        return TextContent(
            type="text",
            text=f"Audio file does not exist: {audio_file_path}"
        )
    except Exception as e:
        logging.error(f"Voice cloning failed: {str(e)}")
        return TextContent(
            type="text",
            text=f"Voice cloning failed, tool temporarily unavailable"
        )
