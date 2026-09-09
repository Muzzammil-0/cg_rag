import pymupdf
from pathlib import Path
import json
import re

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_txt(txt_path):
    return txt_path.read_text(encoding="utf-8")

def filter_non_english(text):
    """Remove lines containing Devanagari or other non-ASCII Indic scripts (optional)."""
    filtered = []
    for line in text.split('\n'):
        if not re.search(r'[\u0900-\u097F]', line):
            filtered.append(line)
    return '\n'.join(filtered)

if __name__ == "__main__":
    # Define all raw directories to scan
    raw_dirs = [
        Path("data/raw/central_acts"),
        Path("data/raw/cg_acts"),
        # Add future folders here, e.g., Path("data/raw/judgments")
    ]

    output_file = Path("data/processed/extracted_text.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted = {}

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            print(f"Warning: {raw_dir} does not exist. Skipping.")
            continue

        print(f"\nScanning {raw_dir}...")
        print("All files:", list(raw_dir.glob("*")))

        # Extract PDFs
        for pdf in raw_dir.glob("*.pdf"):
            print(f"Extracting PDF: {pdf.name}")
            text = extract_text_from_pdf(pdf)
            # Optional: filter out Hindi lines (uncomment if needed)
            # text = filter_non_english(text)
            extracted[pdf.name] = text

        # Extract TXTs
        for txt in raw_dir.glob("*.txt"):
            print(f"Extracting TXT: {txt.name}")
            text = extract_text_from_txt(txt)
            # Optional: filter out Hindi lines
            # text = filter_non_english(text)
            extracted[txt.name] = text

    # Save all extracted text
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted {len(extracted)} files. Saved to {output_file}")