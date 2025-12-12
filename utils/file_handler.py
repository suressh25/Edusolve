"""
File Handler - Manages file uploads and downloads
"""

import asyncio
from typing import Optional
from pathlib import Path
import shutil
from config.settings import settings
from utils.logger import logger


class FileHandler:
    """Handle file operations for EduSolve"""

    @staticmethod
    async def save_uploaded_file(uploaded_file, destination_dir: str) -> str:
        """Save uploaded file to destination directory"""

        try:
            dest_path = Path(destination_dir) / uploaded_file.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Save file
            def save():
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            await asyncio.to_thread(save)

            logger.info(f"Saved uploaded file: {dest_path}")
            return str(dest_path)

        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise

    @staticmethod
    def validate_file(uploaded_file) -> bool:
        """Validate uploaded file"""

        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.2f}MB (max: {settings.MAX_FILE_SIZE_MB}MB)"
            )

        # Check extension
        file_extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return True

    @staticmethod
    async def cleanup_old_files(directory: str, max_age_hours: int = 24):
        """Clean up old files from directory"""

        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return

            import time

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            def cleanup():
                count = 0
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            count += 1
                return count

            removed_count = await asyncio.to_thread(cleanup)
            logger.info(f"Cleaned up {removed_count} old files from {directory}")

        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")
