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

def is_toc_chunk(chunk):
    if chunk["metadata"].get("doc_type") != "constitution":
        return False
    text = chunk["text"].strip()
    if len(text) < 150 and re.match(r'^\d+[A-Z]?\.\s+[A-Z][^.]*\.?$', text):
        return True
    return False

def is_noise_chunk(chunk):
    text = chunk["text"].strip()
    if len(text) < 60:
        return True
    if re.match(r'^\d+\.\s+(Subs\.|Ins\.|Omitted|Repealed|w\.e\.f)', text):
        if len(text) < 200:
            return True
    return False

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

def split_constitution_subsections(chunks):
    result = []
    for c in chunks:
        if c["metadata"].get("doc_type") != "constitution":
            result.append(c)
            continue

        article = c["metadata"].get("article", "")
        text = c["text"]

        heading_match = re.match(r'^\s*\d+[A-Z]*\.\s+', text)
        if not heading_match:
            result.append(c)
            continue

        search_start = heading_match.end()
        sub_match = re.search(r'(\d+)([A-Z])\.\s+(?=[A-Z(])', text[search_start:])

        if not sub_match:
            result.append(c)
            continue

        sub_abs_start = search_start + sub_match.start()
        parent_num = sub_match.group(1)
        sub_letter = sub_match.group(2)
        sub_article = f"{parent_num}{sub_letter}"

        if article != parent_num:
            result.append(c)
            continue

        parent_text = text[:sub_abs_start].strip()
        sub_text = text[sub_abs_start:].strip()

        if parent_text:
            result.append({
                "text": parent_text,
                "metadata": {**c["metadata"]}
            })
        if sub_text:
            result.append({
                "text": sub_text,
                "metadata": {
                    **c["metadata"],
                    "article": sub_article
                }
            })

    return result


# ------------------------------------------------------------
# UPDATED: handles "2[304B." amendment-insertion markers
# ------------------------------------------------------------
def chunk_generic_act(text, source):
    # Allow an optional footnote marker like "1[" or "2[" or "12[" before the
    # section number. This is how Indian PDFs mark sections inserted by
    # subsequent amendments (e.g. IPC 304B, IPC 498A, IPC 376A-D).
    pattern = re.compile(
        r'^\s*(?:\d{1,3}\[)?(?:(?:Section|Rule)\s+)?(\d+[A-Z]*)\.\s+',
        re.MULTILINE
    )
    matches = list(pattern.finditer(text))
    if not matches:
        pattern = re.compile(r'^\s*(\d+[A-Z]*)\.\s+', re.MULTILINE)
        matches = list(pattern.finditer(text))

    chunks = []
    for i, match in enumerate(matches):
        number = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        # Strip a leading footnote marker from the chunk text so the
        # citation reads cleanly as "304B. Dowry death.—..." not "2[304B..."
        chunk_text = re.sub(r'^\d{1,3}\[\s*', '', chunk_text)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_type": "act",
                "section": number
            }
        })
    return chunks


# ------------------------------
# NEW: judgment chunker
# ------------------------------
def chunk_judgment(text, source):
    """
    Split a judgment by numbered paragraphs.
    """
    patterns = [
        re.compile(r'^\s*(\d+)\.\s+(?=[A-Z(])', re.MULTILINE),
        re.compile(r'^\s*Para\s+(\d+)\.\s+', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\s*(\d+)\s{2,}(?=[A-Z(])', re.MULTILINE),
    ]

    matches = []
    for pat in patterns:
        m = list(pat.finditer(text))
        if len(m) > len(matches):
            matches = m

    chunks = []
    if not matches:
        chunks.append({
            "text": text.strip(),
            "metadata": {
                "source": source,
                "doc_type": "judgment",
                "para": "unknown"
            }
        })
        return chunks

    for i, match in enumerate(matches):
        para_num = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 100:
            continue

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "doc_type": "judgment",
                "para": para_num
            }
        })
    return chunks


def main():
    input_file = Path("data/processed/extracted_text.json")
    output_file = Path("data/processed/chunks_v2.jsonl")

    with open(input_file, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    all_chunks = []
    stats = {
        "total": 0,
        "toc_filtered": 0,
        "noise_filtered": 0,
        "acts_processed": 0,
        "judgments_processed": 0,
    }

    for filename, entry in extracted.items():
        if isinstance(entry, dict):
            text = entry.get("text", "")
            doc_type = entry.get("doc_type", "act")
        else:
            text = entry
            doc_type = "act"

        stats["total"] += 1

        if doc_type == "judgment":
            stats["judgments_processed"] += 1
            chunks = chunk_judgment(text, filename)
        else:
            stats["acts_processed"] += 1
            cleaned = clean_text(text)
            if "constitution" in filename.lower():
                chunks = chunk_constitution(cleaned, filename)
            else:
                chunks = chunk_generic_act(cleaned, filename)

        for c in chunks:
            if is_toc_chunk(c):
                stats["toc_filtered"] += 1
                continue
            if is_noise_chunk(c):
                stats["noise_filtered"] += 1
                continue
            all_chunks.append(c)

    before_count = len(all_chunks)
    all_chunks = split_constitution_subsections(all_chunks)
    after_count = len(all_chunks)
    split_extra = after_count - before_count

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Total documents scanned: {stats['total']}")
    print(f"Acts processed: {stats['acts_processed']}")
    print(f"Judgments processed: {stats['judgments_processed']}")
    print(f"Filtered TOC: {stats['toc_filtered']}")
    print(f"Filtered noise: {stats['noise_filtered']}")
    print(f"Sub-articles split out: {split_extra}")
    print(f"Kept chunks: {len(all_chunks)} -> {output_file}")


if __name__ == "__main__":
    main()