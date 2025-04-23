# structure_analyzer.py
import re
from openai import OpenAI

client = OpenAI()


def extract_toc_and_sample(doc,
                           toc_identifiers=("Table of Contents", "Contents"),
                           chapter_start=1,
                           chapter_length=3):
    toc_pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        if any(identifier in text for identifier in toc_identifiers):
            toc_pages.append(i)
    toc_text = "\n\n".join([doc[i].get_text() for i in toc_pages])
    sample_text = "\n\n".join([
        doc[i].get_text() for i in range(
            chapter_start, min(len(doc), chapter_start + chapter_length))
    ])
    return toc_text, sample_text


def learn_textbook_structure(toc_text,
                             sample_chapter_text,
                             model="gpt-4o-mini"):
    prompt = f"""
You are a document structure analyst. Based on the Table of Contents and the first chapter of a textbook, create a structure map that explains:

1. How chapters and sections are numbered (e.g., \"Chapter 6\", \"6.1\", \"6.1.1\")
2. What kind of content appears in sections (e.g., examples, definitions, theorems)
3. Any consistent format or layout across chapters.
4. Record ALL chapter ranges (e.g., Chapter 6: 6.1-6.4),

Output a JSON structure like:
```
{{
  "numbering_scheme": "Chapter.Section.Subsection", 
  "section_pattern": "Each section starts with a bold header like '6.1 Motivation'",
  "common_elements": ["definitions", "examples", "summary"],
  "sample_structure": {{
    "6.1": "Motivation",
    "6.2": "Key Concepts"
  }},
  "chapter_ranges": {{
    "6": ["6.1", "6.4"],
    "7": ["7.1", "7.3"],
    "VI": ["VI.1", "VI.3"],
    "Appendix": ["A.1", "A.2"]
  }},
}}
```

TOC:
{toc_text}

Chapter Sample:
{sample_chapter_text}
"""
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.3,
    )
    return completion.choices[0].message.content


def smart_chunk_text(text, max_chunk_size=8000):
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if re.match(r"^\d+\.\d+.*", line) and len(current_chunk) > 2000:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += line + "\n"
        if len(current_chunk) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks
