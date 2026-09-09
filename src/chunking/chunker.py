import json
import re
from pathlib import Path

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'Contents\s*\n\s*ARTICLES\s*\([ivx]+\)', '', text)
    text = re.sub(r'\bARTICLES\b', '', text)
    text = re.sub(r'\bContents\b', '', text)
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = [line for line in lines if not re.fullmatch(r'\d{1,4}', line)]
    return '\n'.join(cleaned_lines)

def chunk_constitution(text, source):
    body_marker = "India, that is Bharat"
    marker_index = text.find(body_marker)
    start_pos = 0

    if marker_index != -1:
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
        occurrences = [m.start() for m in re.finditer(r'THE CONSTITUTION OF INDIA', text)]
        if len(occurrences) >= 2:
            start_pos = occurrences[1]

    body_text = text[start_pos:]
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

def chunk_generic_act(text, source):
    """
    Chunk any act/rule document.
    Pattern matches:
      - "Section 1. Title..."
      - "Rule 1. Title..."
      - "1. Title..." (plain)
    """
    # First try explicit "Section" or "Rule" headings
    pattern = re.compile(r'^\s*(?:(?:Section|Rule)\s+)?(\d+[A-Z]*)\.\s+', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        # Fallback: any number followed by dot and space (might catch footnotes)
        pattern = re.compile(r'^\s*(\d+[A-Z]*)\.\s+', re.MULTILINE)
        matches = list(pattern.finditer(text))

    chunks = []
    for i, match in enumerate(matches):
        number = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        # Try to determine type from the text (optional)
        # if "section" in chunk_text.lower():
        #     doc_type = "act_section"
        # elif "rule" in chunk_text.lower():
        #     doc_type = "rule"
        # else:
        #     doc_type = "generic"
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_type": "act",   # generic act/rule chunk
                "section": number   # use "section" key for compatibility
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
            chunks = chunk_generic_act(cleaned, filename)  # now handles acts and rules
        all_chunks.extend(chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(all_chunks)} chunks -> {output_file}")

if __name__ == "__main__":
    main()