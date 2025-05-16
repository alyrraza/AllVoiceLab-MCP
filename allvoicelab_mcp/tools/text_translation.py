import logging
import os
import random
import time

import requests
from mcp.types import TextContent

from .base import get_client


def text_translation_tool(
    file_path: str,
    target_lang: str,
    source_lang: str = "auto",
    output_dir: str = None
) -> TextContent:
    all_voice_lab = get_client()
    output_dir = all_voice_lab.get_output_path(output_dir)
    max_polling_time = 600
    polling_interval = 10
    logging.info(
        f"Tool called: text_translation, file_path: {file_path}, target_lang: {target_lang}, source_lang: {source_lang}")
    logging.info(f"Max polling time: {max_polling_time}s, Polling interval: {polling_interval}s")

    
    if not file_path:
        logging.warning("File path parameter is empty")
        return TextContent(
            type="text",
            text="file_path parameter cannot be empty"
        )

    
    if not os.path.exists(file_path):
        logging.warning(f"File does not exist: {file_path}")
        return TextContent(
            type="text",
            text=f"File does not exist: {file_path}"
        )

    
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension not in [".txt", ".srt"]:
        logging.warning(f"Unsupported file format: {file_extension}")
        return TextContent(
            type="text",
            text=f"Unsupported file format. Only TXT and SRT formats are supported."
        )

    
    max_size_bytes = 10 * 1024 * 1024  # 10MB in bytes
    file_size = os.path.getsize(file_path)
    if file_size > max_size_bytes:
        logging.warning(f"File size exceeds limit: {file_size} bytes, max allowed: {max_size_bytes} bytes")
        return TextContent(
            type="text",
            text=f"File size exceeds the maximum limit of 10MB. Please use a smaller file."
        )

    try:
        if all_voice_lab is None:
            logging.error("all_voice_lab client is not initialized.")
            return TextContent(type="text",
                               text="Error: AllVoiceLab client not initialized. Please check server setup.")

        
        logging.info("Starting text translation process")
        project_id = all_voice_lab.text_translation(
            file_path=file_path,
            target_lang=target_lang,
            source_lang=source_lang
        )
        logging.info(f"Text translation task submitted. Project ID: {project_id}")

        
        logging.info(f"Starting to poll translation status for Project ID: {project_id}")
        start_time = time.time()
        completed = False

        
        while time.time() - start_time < max_polling_time:
            try:
                
                translation_result = all_voice_lab.get_text_translation_result(project_id)
                if translation_result is None:
                    logging.warning(f"Failed to get translation result for Project ID: {project_id}")
                    time.sleep(polling_interval)
                    continue

                logging.info(f"Translation status: {translation_result.status} for Project ID: {project_id}")

                
                if translation_result.status.lower() == "success":
                    logging.info(f"Text translation completed for Project ID: {project_id}")
                    completed = True

                    
                    if translation_result.result and translation_result.result.startswith("http"):
                        try:
                            
                            os.makedirs(output_dir, exist_ok=True)

                            
                            # Get original filename without extension
                            original_filename = os.path.splitext(os.path.basename(file_path))[0]
                            timestamp = int(time.time())
                            random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
                            filename = f"{original_filename}_translation_{timestamp}_{random_suffix}.txt"
                            file_path = os.path.join(output_dir, filename)

                            
                            logging.info(f"Downloading translation result from URL: {translation_result.result}")
                            response = requests.get(translation_result.result, timeout=30)
                            response.raise_for_status()

                            
                            with open(file_path, 'wb') as f:
                                f.write(response.content)

                            
                            with open(file_path, 'r', encoding='utf-8') as f:
                                translated_text = f.read()

                            logging.info(f"Translation result downloaded and saved to: {file_path}")

                            
                            result_text = f"Translation completed successfully.\n\n"
                            result_text += f"Source Language: {translation_result.source_lang}\n"
                            result_text += f"Target Language: {translation_result.target_lang}\n\n"
                            result_text += f"Translated Text:\n{translated_text}\n\n"
                            result_text += f"Result file saved at: {file_path}"

                            return TextContent(
                                type="text",
                                text=result_text
                            )
                        except Exception as e:
                            logging.error(f"Error downloading translation result: {str(e)}")
                            result_text = f"Translation completed, but failed to download result: {str(e)}\n\n"
                            result_text += f"Source Language: {translation_result.source_lang}\n"
                            result_text += f"Target Language: {translation_result.target_lang}\n\n"
                            result_text += f"Result URL: {translation_result.result}"

                            return TextContent(
                                type="text",
                                text=result_text
                            )
                    else:
                        
                        result_text = f"Translation completed successfully.\n\n"
                        result_text += f"Source Language: {translation_result.source_lang}\n"
                        result_text += f"Target Language: {translation_result.target_lang}\n\n"
                        result_text += f"No result URL available."

                        return TextContent(
                            type="text",
                            text=result_text
                        )
                elif translation_result.status.lower() == "failed":
                    logging.error(
                        f"Translation failed for Project ID: {project_id}")
                    return TextContent(
                        type="text",
                        text=f"Translation failed"
                    )

            except Exception as e:
                logging.error(f"Error while polling translation status: {str(e)}")

            
            time.sleep(polling_interval)

        
        if not completed:
            logging.warning(f"Translation not completed within {max_polling_time} seconds for Project ID: {project_id}")
            return TextContent(
                type="text",
                text=f"Translation is still in progress. Please check the status later using the project ID: {project_id}"
            )

    except Exception as e:
        logging.error(f"Text translation failed: {str(e)}")
        return TextContent(
            type="text",
            text=f"Translation failed: {str(e)}"
        )
