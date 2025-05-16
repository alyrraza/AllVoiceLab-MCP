import logging

from mcp.types import TextContent

from .base import get_client


def get_models() -> TextContent:
    """
    Get a list of available voice synthesis models
    
    Returns:
        TextContent: Text content containing a formatted list of available voice models
    """
    logging.info("Tool called: get_models")
    all_voice_lab = get_client()
    
    try:
        logging.info("Getting supported voice model list")
        resp = all_voice_lab.get_supported_voice_model()
        models = resp.models
        logging.info(f"Retrieved {len(models)} voice models")

        if len(models) == 0:
            logging.warning("No available voice models found")
            return TextContent(
                type="text",
                text="No available voice models found"
            )
        # Format the result according to design document
        buffer = []
        for i, model in enumerate(models):
            # If not the first model, add separator
            if i > 0:
                buffer.append("---------------------\n")

            buffer.append(f"- id: {model.model_id}\n")
            buffer.append(f"- Name: {model.name}\n")
            buffer.append(f"- Description: {model.description}\n")

        # Add the final separator
        buffer.append("---------------------\n")

        # Join the list into a string
        result = "".join(buffer)
        logging.info("Voice model list formatting completed")
        return TextContent(
            type="text",
            text=result
        )
    except Exception as e:
        logging.error(f"Failed to get voice models: {str(e)}")
        return TextContent(
            type="text",
            text=f"Failed to get models, tool temporarily unavailable"
        )


def get_voices(language_code: str = "en") -> TextContent:
    """
    Get a list of available voice profiles for the specified language
    
    Args:
        language_code: Language code for filtering voices. Must be one of: [zh, en, ja, fr, de, ko]. Default is "en".
    
    Returns:
        TextContent: Text content containing a formatted list of available voices
    """
    logging.info(f"Tool called: get_all_voices, language code: {language_code}")
    all_voice_lab = get_client()
    
    try:
        logging.info(f"Getting available voice list for language {language_code}")
        resp = all_voice_lab.get_all_voices(language_code=language_code)
        voices = resp.voices
        logging.info(f"Retrieved {len(voices)} voices")

        if len(voices) == 0:
            logging.warning(f"No available voices found for language {language_code}")
            return TextContent(
                type="text",
                text="No available voices found"
            )

        # Format the result according to design document
        buffer = []
        for i, voice in enumerate(voices):
            # If not the first voice, add separator
            if i > 0:
                buffer.append("---------------------\n")

            buffer.append(f"- id: {voice.voice_id}\n")
            buffer.append(f"- Name: {voice.name}\n")
            buffer.append(f"- Description: {voice.description}\n")

            # Add language and gender information (if exists)
            if "language" in voice.labels:
                buffer.append(f"- Language: {voice.labels['language']}\n")
            if "gender" in voice.labels:
                buffer.append(f"- Gender: {voice.labels['gender']}\n")

        # Add the final separator
        buffer.append("---------------------\n")

        # Join the list into a string
        result = "".join(buffer)
        logging.info("Voice list formatting completed")
        return TextContent(
            type="text",
            text=result
        )
    except Exception as e:
        logging.error(f"Failed to get voice list: {str(e)}")
        return TextContent(
            type="text",
            text=f"Failed to get voices, tool temporarily unavailable"
        )