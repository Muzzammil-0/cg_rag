import pymupdf 
from pathlib import Path
import json

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

if __name__ == "__main__":
    raw_dir = Path("data/raw/central_acts")
    print("Looking in:", raw_dir.resolve())
    print("Folder exists:", raw_dir.exists())
    print("All files in folder:", list(raw_dir.glob("*")))
    print("PDF files:", list(raw_dir.glob("*.pdf")))

    output_file = Path("data/processed/extracted_text.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted = {}
    for pdf in raw_dir.glob("*.pdf"):
        print(f"Extracting {pdf.name}...")
        text = extract_text_from_pdf(pdf)
        extracted[pdf.name] = text

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(extracted)} files. Saved to {output_file}")