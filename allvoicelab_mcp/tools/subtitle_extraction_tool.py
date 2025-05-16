import logging
import os
import random
import time

import requests
from mcp.types import TextContent

from .base import get_client


def subtitle_extraction_tool(
    video_file_path: str,
    language_code: str = "auto",
    name: str = None,
    output_dir: str = None
) -> TextContent:
    all_voice_lab = get_client()
    output_dir = all_voice_lab.get_output_path(output_dir)
    max_polling_time = 600
    polling_interval = 10
    logging.info(
        f"Tool called: subtitle_extraction, video_file_path: {video_file_path}, language_code: {language_code}, name: {name}")
    logging.info(f"Max polling time: {max_polling_time}s, Polling interval: {polling_interval}s")

    
    if not video_file_path:
        logging.warning("Video file path parameter is empty")
        return TextContent(
            type="text",
            text="video_file_path parameter cannot be empty"
        )

    
    if not os.path.exists(video_file_path):
        logging.warning(f"Video file does not exist: {video_file_path}")
        return TextContent(
            type="text",
            text=f"Video file does not exist: {video_file_path}"
        )

    
    _, file_extension = os.path.splitext(video_file_path)
    file_extension = file_extension.lower()
    if file_extension not in [".mp4", ".mov"]:
        logging.warning(f"Unsupported video file format: {file_extension}")
        return TextContent(
            type="text",
            text=f"Unsupported video file format. Only MP4 and MOV formats are supported."
        )

    
    max_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    file_size = os.path.getsize(video_file_path)
    if file_size > max_size_bytes:
        logging.warning(f"Video file size exceeds limit: {file_size} bytes, max allowed: {max_size_bytes} bytes")
        return TextContent(
            type="text",
            text=f"Video file size exceeds the maximum limit of 2GB. Please use a smaller file."
        )

    try:
        if all_voice_lab is None:
            logging.error("all_voice_lab client is not initialized.")
            return TextContent(type="text",
                               text="Error: AllVoiceLab client not initialized. Please check server setup.")

        
        logging.info("Starting subtitle extraction process")
        project_id = all_voice_lab.subtitle_extraction(
            video_file_path=video_file_path,
            language_code=language_code,
            name=name
        )
        logging.info(f"Subtitle extraction task submitted. Project ID: {project_id}")

        
        logging.info(f"Starting to poll extraction status for Project ID: {project_id}")
        start_time = time.time()
        completed = False

        
        while time.time() - start_time < max_polling_time:
            try:
                
                extraction_info = all_voice_lab.get_extraction_info(project_id)
                logging.info(f"Extraction status: {extraction_info.status} for Project ID: {project_id}")

                
                if extraction_info.status.lower() == "success":
                    logging.info(f"Subtitle extraction completed for Project ID: {project_id}")
                    completed = True

                    
                    if hasattr(extraction_info, 'result') and extraction_info.result:
                        result_url = extraction_info.result
                        logging.info(f"Downloading subtitle file from: {result_url}")

                        try:
                            
                            url = result_url

                            
                            headers = all_voice_lab._get_headers(content_type="", accept="*/*")

                            
                            response = requests.get(url, headers=headers, stream=True)

                            
                            response.raise_for_status()

                            
                            # Get original filename without extension
                            original_filename = os.path.splitext(os.path.basename(video_file_path))[0]
                            timestamp = int(time.time())
                            random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
                            filename = f"{original_filename}_subtitle_{timestamp}_{random_suffix}.srt"

                            
                            os.makedirs(output_dir, exist_ok=True)
                            file_path = os.path.join(output_dir, filename)

                            
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)

                            logging.info(f"Subtitle file saved to: {file_path}")

                            
                            info_parts = []
                            info_parts.append(f"Subtitle extraction completed successfully.")
                            info_parts.append(f"Project ID: {project_id}")
                            info_parts.append(f"Subtitle file saved to: {file_path}")

                            return TextContent(
                                type="text",
                                text="\n".join(info_parts)
                            )

                        except Exception as e:
                            logging.error(f"Failed to download subtitle file: {str(e)}")
                            
                            info_parts = []
                            info_parts.append(f"Subtitle extraction completed successfully.")
                            info_parts.append(f"Project ID: {project_id}")
                            info_parts.append(f"Result URL: {extraction_info.result}")
                            info_parts.append(f"Failed to download subtitle file: {str(e)}")

                            return TextContent(
                                type="text",
                                text="\n".join(info_parts)
                            )
                    else:
                        info_parts = []
                        info_parts.append(f"Subtitle extraction completed successfully.")
                        info_parts.append(f"Project ID: {project_id}")
                        info_parts.append("No subtitle file URL available.")

                        return TextContent(
                            type="text",
                            text="\n".join(info_parts)
                        )

                elif extraction_info.status.lower() in ["failed", "error"]:
                    logging.error(f"Subtitle extraction failed for Project ID: {project_id}")
                    error_message = "Subtitle extraction failed."
                    if hasattr(extraction_info, 'message') and extraction_info.message:
                        error_message += f" Message: {extraction_info.message}"
                    return TextContent(
                        type="text",
                        text=f"{error_message}\nProject ID: {project_id}"
                    )

                logging.info(f"Waiting {polling_interval} seconds before next poll")
                time.sleep(polling_interval)

            except Exception as e:
                logging.error(f"Error while polling extraction status: {str(e)}")
                time.sleep(polling_interval)


        if not completed:
            logging.warning(f"Polling timed out after {max_polling_time} seconds for Project ID: {project_id}")
            return TextContent(
                type="text",
                text=f"Subtitle extraction is still in progress. Please check the status later using the 'get_extraction_info' tool.\n"
                     f"Project ID: {project_id}"
            )

    except FileNotFoundError as e:
        logging.error(f"Error in subtitle_extraction_tool: {str(e)}")
        return TextContent(type="text", text=str(e))
    except Exception as e:
        logging.error(f"Failed to extract subtitles: {str(e)}")
        error_message = f"Failed to extract subtitles. Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_message = f"Failed to extract subtitles: API Error - {error_detail.get('message', str(e))}"
            except ValueError:  # Not a JSON response
                error_message = f"Failed to extract subtitles: API Error - {e.response.status_code} {e.response.text}"
        return TextContent(
            type="text",
            text=error_message
        )
