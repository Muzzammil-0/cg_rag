import pymupdf
from pathlib import Path
import json
import logging
from typing import Dict, Any, List

# --- DOCKER PATH FIX ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extract text from a PDF file with robust error handling."""
    try:
        doc = pymupdf.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        logging.info(f"Successfully extracted text from {pdf_path.name}")
        return text
    except pymupdf.PdfError as e:
        logging.error(f"Failed to read PDF {pdf_path.name} due to a PDF error: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {pdf_path.name}: {e}")
        return None

def extract_text_from_txt(txt_path: Path) -> str | None:
    """Extract text from a TXT file with robust error handling."""
    try:
        return txt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error(f"TXT file not found at path: {txt_path}")
        return None
    except UnicodeDecodeError as e:
        logging.error(f"Could not decode TXT file {txt_path.name}. Check encoding. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {txt_path.name}: {e}")
        return None

def run_extraction():
    """Main function to discover and extract text from all PDF and TXT files."""
    # --- DOCKER PATH FIX APPLIED HERE ---
    raw_dirs: List[tuple[Path, str]] = [
        (PROJECT_ROOT / "data/raw/central_acts", "act"),
        (PROJECT_ROOT / "data/raw/cg_acts", "act"),
        (PROJECT_ROOT / "data/raw/judgments", "judgment"),
    ]

    output_file: Path = PROJECT_ROOT / "data/processed/extracted_text.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted: Dict[str, Any] = {}

    for raw_dir, doc_type in raw_dirs:
        if not raw_dir.exists():
            logging.warning(f"Directory {raw_dir} does not exist. Skipping.")
            continue

        logging.info("-" * 30)
        logging.info(f"Scanning {raw_dir} (doc_type={doc_type})...")

        # Process PDFs (match extension case-insensitively so .PDF works on Linux)
        pdf_files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        for pdf in pdf_files:
            text = extract_text_from_pdf(pdf)
            if text:
                extracted[pdf.name] = {"text": text, "doc_type": doc_type}

        # Process TXT files
        txt_files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
        for txt in txt_files:
            text = extract_text_from_txt(txt)
            if text:
                extracted[txt.name] = {"text": text, "doc_type": doc_type}

    # Write aggregated data to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        logging.info(f"\n✅ Extraction complete. Processed {len(extracted)} files. Results saved to {output_file}")
    except IOError as e:
        logging.critical(f"CRITICAL: Could not write to output file {output_file}: {e}")

if __name__ == "__main__":
    run_extraction()