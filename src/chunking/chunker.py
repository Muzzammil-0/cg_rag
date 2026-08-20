import json
import re
from pathlib import Path

def clean_text(text):
    # Remove page numbers and repeated headers (adjust as needed)
    text = re.sub(r'\n{3,}', '\n\n', text)           # collapse 3+ newlines
    text = re.sub(r'Contents\s*\n\s*ARTICLES\s*\([ivx]+\)', '', text)
    text = re.sub(r'\bARTICLES\b', '', text)         # remove "ARTICLES" headers
    text = re.sub(r'\bContents\b', '', text)         # remove "Contents"
    # Remove lines that are just page numbers (simple heuristic)
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = [line for line in lines if not re.fullmatch(r'\d{1,4}', line)]
    return '\n'.join(cleaned_lines)

def chunk_constitution(text, source):
    # Find the actual start of Article 1 using a known body phrase
    body_marker = "India, that is Bharat"
    marker_index = text.find(body_marker)
    start_pos = 0

    if marker_index != -1:
        # Search backward from the marker to find the line starting with "1."
        line_start = text.rfind('\n', 0, marker_index)
        while line_start != -1:
            line_end = text.find('\n', line_start + 1)
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            if re.match(r'^1\.\s', line):
                start_pos = line_start
                break
            line_start = text.rfind('\n', 0, line_start)
    else:
        # Fallback: use second occurrence of "THE CONSTITUTION OF INDIA"
        occurrences = [m.start() for m in re.finditer(r'THE CONSTITUTION OF INDIA', text)]
        if len(occurrences) >= 2:
            start_pos = occurrences[1]

    body_text = text[start_pos:]

    # Article heading pattern: number (optionally followed by capital letters) then dot and space
    pattern = re.compile(r'^\s*(\d+[A-Z]*)\.\s+(?=[A-Z(])', re.MULTILINE)
    matches = list(pattern.finditer(body_text))

    chunks = []
    for i, match in enumerate(matches):
        article_num = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(body_text)
        chunk_text = body_text[start:end].strip()
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_type": "constitution",
                "article": article_num
            }
        })
    return chunks

def chunk_ipc_crpc(text, source):
    # Pattern: "Section 1." or "1." (adjust based on actual text)
    pattern = re.compile(r'^\s*(?:Section\s+)?(\d+[A-Z]*)\.\s+', re.MULTILINE)
    matches = list(pattern.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        section_num = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_type": "act",
                "section": section_num
            }
        })
    return chunks

def main():
    input_file = Path("data/processed/extracted_text.json")
    output_file = Path("data/processed/chunks.jsonl")

    with open(input_file, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    all_chunks = []
    for filename, text in extracted.items():
        cleaned = clean_text(text)
        if "constitution" in filename.lower():
            chunks = chunk_constitution(cleaned, filename)
        else:
            chunks = chunk_ipc_crpc(cleaned, filename)
        all_chunks.extend(chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(all_chunks)} chunks -> {output_file}")

if __name__ == "__main__":
    main()