import logging
from typing import Optional

from client import AllVoiceLab

# Global variable, will be initialized in the main function
all_voice_lab = None


def set_client(client: AllVoiceLab) -> None:
    """
    Set the global client instance
    
    Args:
        client: AllVoiceLab client instance
    """
    global all_voice_lab
    all_voice_lab = client
    logging.info("AllVoiceLab client set in tools module")


def get_client() -> Optional[AllVoiceLab]:
    """
    Get the global client instance
    
    Returns:
        Optional[AllVoiceLab]: AllVoiceLab client instance
    """
    return all_voice_lab
