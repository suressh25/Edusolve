"""
Document Converter - Handles DOCX to PDF conversion with multiple fallback methods
Supports Windows (docx2pdf), Linux (LibreOffice), and cross-platform solutions
"""

import asyncio
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from utils.logger import logger


class DocumentConverter:
    """Convert documents between formats (primarily DOCX to PDF)"""

    def __init__(self):
        self.system = platform.system()
        self.conversion_method = self._detect_conversion_method()
        logger.info(
            f"Document converter initialized with method: {self.conversion_method}"
        )

    def _detect_conversion_method(self) -> str:
        """Detect available conversion method based on system"""

        # Check for LibreOffice (works on all platforms)
        if self._check_libreoffice():
            return "libreoffice"

        # Check for docx2pdf (Windows/macOS with MS Word)
        if self.system in ["Windows", "Darwin"]:
            try:
                import docx2pdf

                return "docx2pdf"
            except ImportError:
                pass

        # Fallback: return DOCX as-is
        logger.warning(
            "No PDF conversion method available. PDFs will not be generated."
        )
        return "none"

    def _check_libreoffice(self) -> bool:
        """Check if LibreOffice is installed and accessible"""

        commands = ["soffice", "libreoffice"]

        for cmd in commands:
            if shutil.which(cmd):
                logger.info(f"Found LibreOffice command: {cmd}")
                return True

        return False

    async def docx_to_pdf(self, docx_path: str, pdf_path: Optional[str] = None) -> str:
        """
        Convert DOCX file to PDF

        Args:
            docx_path: Path to input DOCX file
            pdf_path: Optional path to output PDF file (defaults to same name with .pdf)

        Returns:
            Path to generated PDF file or original DOCX if conversion fails
        """

        docx_file = Path(docx_path)

        if not docx_file.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        # Determine output path
        if pdf_path is None:
            pdf_path = str(docx_file.with_suffix(".pdf"))

        pdf_file = Path(pdf_path)

        try:
            if self.conversion_method == "libreoffice":
                return await self._convert_with_libreoffice(
                    str(docx_file), str(pdf_file)
                )

            elif self.conversion_method == "docx2pdf":
                return await self._convert_with_docx2pdf(str(docx_file), str(pdf_file))

            else:
                logger.warning("No conversion method available, returning DOCX file")
                return str(docx_file)

        except Exception as e:
            logger.error(f"PDF conversion failed: {str(e)}")
            logger.warning("Returning original DOCX file")
            return str(docx_file)

    async def _convert_with_libreoffice(self, docx_path: str, pdf_path: str) -> str:
        """Convert using LibreOffice (cross-platform)"""

        try:
            docx_file = Path(docx_path)
            pdf_file = Path(pdf_path)
            output_dir = pdf_file.parent

            # Determine LibreOffice command
            soffice_cmd = "soffice" if shutil.which("soffice") else "libreoffice"

            # LibreOffice command
            cmd = [
                soffice_cmd,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(docx_file),
            ]

            logger.info(f"Running LibreOffice conversion: {' '.join(cmd)}")

            # Run conversion in thread to avoid blocking
            def run_conversion():
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(timeout=60)

                if process.returncode != 0:
                    raise subprocess.SubprocessError(
                        f"LibreOffice conversion failed: {stderr.decode()}"
                    )

                return stdout.decode()

            await asyncio.to_thread(run_conversion)

            # LibreOffice creates file with same name as input
            generated_pdf = output_dir / f"{docx_file.stem}.pdf"

            # Rename if needed
            if generated_pdf != pdf_file:
                if pdf_file.exists():
                    pdf_file.unlink()
                generated_pdf.rename(pdf_file)

            if pdf_file.exists():
                logger.info(f"Successfully converted to PDF: {pdf_file}")
                return str(pdf_file)
            else:
                raise FileNotFoundError("PDF file was not created")

        except Exception as e:
            logger.error(f"LibreOffice conversion error: {str(e)}")
            raise

    async def _convert_with_docx2pdf(self, docx_path: str, pdf_path: str) -> str:
        """Convert using docx2pdf (Windows/macOS with MS Word)"""

        try:
            from docx2pdf import convert

            logger.info(f"Converting with docx2pdf: {docx_path} -> {pdf_path}")

            # Run conversion in thread
            await asyncio.to_thread(convert, docx_path, pdf_path)

            pdf_file = Path(pdf_path)
            if pdf_file.exists():
                logger.info(f"Successfully converted to PDF: {pdf_file}")
                return str(pdf_file)
            else:
                raise FileNotFoundError("PDF file was not created")

        except Exception as e:
            logger.error(f"docx2pdf conversion error: {str(e)}")
            raise

    async def batch_convert(
        self, docx_files: list, output_dir: Optional[str] = None
    ) -> list:
        """
        Convert multiple DOCX files to PDF

        Args:
            docx_files: List of DOCX file paths
            output_dir: Optional output directory (defaults to same as input files)

        Returns:
            List of converted PDF file paths
        """

        converted_files = []

        for docx_file in docx_files:
            try:
                docx_path = Path(docx_file)

                if output_dir:
                    pdf_path = Path(output_dir) / f"{docx_path.stem}.pdf"
                else:
                    pdf_path = docx_path.with_suffix(".pdf")

                result = await self.docx_to_pdf(str(docx_path), str(pdf_path))
                converted_files.append(result)

            except Exception as e:
                logger.error(f"Failed to convert {docx_file}: {str(e)}")
                converted_files.append(docx_file)  # Keep original on failure

        return converted_files

    def get_conversion_info(self) -> dict:
        """Get information about available conversion methods"""

        return {
            "system": self.system,
            "conversion_method": self.conversion_method,
            "libreoffice_available": self._check_libreoffice(),
            "docx2pdf_available": self.conversion_method == "docx2pdf",
        }
