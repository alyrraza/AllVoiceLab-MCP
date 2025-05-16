import logging
import os
from typing import Tuple, Optional, List, Dict, Any

from mcp.types import TextContent

# Constants definition
AUDIO_FORMATS = [".mp3", ".wav"]


def validate_output_directory(output_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Validate and create output directory
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Tuple[bool, Optional[str]]: (success status, error message)
    """
    if not output_dir:
        logging.warning("Output directory parameter is empty")
        return False, "output_dir parameter cannot be empty"
        
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True, None
    except Exception as e:
        logging.error(f"Failed to create output directory: {output_dir}, error: {str(e)}")
        return False, f"Failed to create output directory: {output_dir}"


def validate_audio_file(
    audio_file_path: str, 
    allowed_formats: List[str] = None,
    max_size_mb: int = 50
) -> Tuple[bool, Optional[str]]:
    """
    Validate if the audio file exists, has a supported format, and is within the size limit
    
    Args:
        audio_file_path: Audio file path
        allowed_formats: List of allowed file formats, defaults to ['.mp3', '.wav']
        max_size_mb: Maximum file size (MB), defaults to 50MB
        
    Returns:
        Tuple[bool, Optional[str]]: (success status, error message)
    """
    if allowed_formats is None:
        allowed_formats = AUDIO_FORMATS
        
    # Check if parameter is empty
    if not audio_file_path:
        logging.warning("Audio file path parameter is empty")
        return False, "audio_file_path parameter cannot be empty"
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        logging.warning(f"Audio file does not exist: {audio_file_path}")
        return False, f"Audio file does not exist: {audio_file_path}"
    
    # Check file format
    _, file_extension = os.path.splitext(audio_file_path)
    file_extension = file_extension.lower()
    if file_extension not in allowed_formats:
        formats_str = ", ".join(allowed_formats)
        logging.warning(f"Unsupported audio file format: {file_extension}")
        return False, f"Unsupported audio file format. Only {formats_str} formats are supported."
    
    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(audio_file_path)
    if file_size > max_size_bytes:
        logging.warning(f"Audio file size exceeds limit: {file_size} bytes, max allowed: {max_size_bytes} bytes")
        return False, f"Audio file size exceeds the maximum limit of {max_size_mb}MB. Please use a smaller file."
    
    return True, None


def format_list_with_separator(items: List[Dict[str, Any]], key_mapping: Dict[str, str]) -> str:
    """
    Format list items with separators
    
    Args:
        items: List of items to format
        key_mapping: Key mapping in the format {original_key: display_name}
        
    Returns:
        str: Formatted string
    """
    buffer = []
    
    for i, item in enumerate(items):
        # Add separator if not the first item
        if i > 0:
            buffer.append("---------------------\n")
        
        # Add each mapped key-value pair
        for original_key, display_name in key_mapping.items():
            if original_key in item.__dict__:
                buffer.append(f"- {display_name}: {getattr(item, original_key)}\n")
        
        # Handle special cases, such as labels
        if hasattr(item, 'labels') and isinstance(item.labels, dict):
            for label_key, label_value in item.labels.items():
                # Capitalize first letter of label name
                buffer.append(f"- {label_key.capitalize()}: {label_value}\n")
    
    # Add final separator
    buffer.append("---------------------\n")
    
    # Join the list into a string
    return "".join(buffer)


def create_error_response(error_message: str) -> TextContent:
    """
    Create standard error response
    
    Args:
        error_message: Error message
        
    Returns:
        TextContent: Response containing the error message
    """
    return TextContent(
        type="text",
        text=error_message
    )


def create_success_response(message: str) -> TextContent:
    """
    Create standard success response
    
    Args:
        message: Success message
        
    Returns:
        TextContent: Response containing the success message
    """
    return TextContent(
        type="text",
        text=message
    )