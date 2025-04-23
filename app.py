import streamlit as st
import fitz  # PyMuPDF
import re
import openai
import os
from dotenv import load_dotenv
import tempfile
from openai import OpenAI
import io

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# app.py (Updated)
import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import re
from dotenv import load_dotenv
from openai import OpenAI
from structure_analyzer import extract_toc_and_sample, learn_textbook_structure, smart_chunk_text

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# --- Utility Functions ---


def process_large_pdf(file, max_pages_per_chunk=50):
    try:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        st.info("Starting PDF processing...")
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        full_text = ""

        for i in range(0, total_pages, max_pages_per_chunk):
            if hasattr(st.session_state, 'processing_active'
                       ) and not st.session_state.processing_active:
                st.warning("PDF processing interrupted")
                doc.close()
                os.unlink(tmp_path)
                return ""
            end_page = min(i + max_pages_per_chunk, total_pages)
            chunk_text = "".join(
                [doc[page].get_text() for page in range(i, end_page)])
            full_text += chunk_text

        doc.close()
        os.unlink(tmp_path)
        st.success(f"Finished processing {total_pages} pages")
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""


def summarize_with_llm(text, model="gpt-4o-mini", language="English"):
    if not text or len(text.strip()) < 10:
        st.error("Text is too short or empty to summarize")
        return None

    st.info(
        f"Sending {len(text):,} characters to {model} for summarization in {language}..."
    )
    prompt = f"""
You are an expert explainer and summarizer for textbooks.
Summarize the following section clearly and comprehensively in {language}. Highlight key terms and concepts in bold.
- Use **LaTeX** for math expressions ($$...$$).
- Format **code** using Markdown blocks (e.g., ```python).
- Retain headings/subheadings and use markdown tables if needed.

Text to summarize:
{text}
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


def chunk_text(text, max_chunk_size=8000):
    return smart_chunk_text(text, max_chunk_size)


def extract_sections_with_llm(doc,
                              section_range="",
                              structure_insight=None,
                              model="gpt-4o-mini",
                              chapter_prefix=None):
    st.info("Attempting to locate sections using AI...")

    if hasattr(st.session_state,
               'processing_active') and not st.session_state.processing_active:
        st.warning("Section extraction interrupted")
        return {}, False

    if not section_range and chapter_prefix:
        if not st.session_state.get("structure_insight"):
            raise ValueError("Structure analysis not completed")

        chapter_ranges = st.session_state.structure_insight["chapter_ranges"]
        if chapter_prefix not in chapter_ranges:
            st.error(
                f"Chapter {chapter_prefix} not found. Available chapters: {list(chapter_ranges.keys())}"
            )
            return {}, False

        start, end = chapter_ranges[chapter_prefix]
        section_range = f"{start}–{end}"
        st.info(
            f"Auto-generated range for Chapter {chapter_prefix}: {section_range}"
        )

    if "–" in section_range:
        start_sec, end_sec = section_range.split("–")
    elif "-" in section_range:
        start_sec, end_sec = section_range.split("-")
    else:
        start_sec = end_sec = section_range.strip()

    toc_pages = [
        i for i in range(len(doc)) if any(
            x in doc[i].get_text() for x in ["Contents", "Table of Contents"])
    ]
    toc_text = "\n\n".join([doc[i].get_text() for i in toc_pages])

    prompt = f"""
You are helping extract specific sections from a textbook using its Table of Contents.

Identify the starting pages for the sections below:
Sections: {start_sec}, {end_sec}

Follow rules:
- Match exact section numbers (e.g., 6.1 not 5.6.1).
- Use TOC layout patterns from below.
- Include page numbers only if confident.

STRUCTURE INSIGHT:
{structure_insight or 'No structure insight provided'}

TOC:
{toc_text}

Return JSON:
```
section_number: start_page
section_number: start_page
```
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.3,
    )
    result = completion.choices[0].message.content
    st.text_area("LLM TOC Extraction", result, height=300)

    section_pages = {}
    matches = re.findall(r'([0-9.]+):\s*(\d+)', result)
    for sec, page in matches:
        if sec in [start_sec, end_sec]:
            section_pages[sec] = int(page)

    if not section_pages:
        st.warning("No valid section pages extracted.")
        return {}, False

    # ✅ VALIDATE PAGE ACCURACY
    validated_pages = validate_and_adjust_pages(doc, section_pages, start_sec)
    if not validated_pages:
        st.warning("Page validation failed. Using LLM-suggested TOC pages.")
        validated_pages = section_pages

    sections = {}
    for sec, page in validated_pages.items():
        section_text = doc[page].get_text()
        sections[sec] = {
            "title": f"Section {sec}",
            "content": section_text,
            "pdf_start_page": page + 1
        }

    return sections, True


def validate_and_adjust_pages(doc, section_pages, start_section):
    """
    Validates and adjusts page numbers by searching for actual section headers in the PDF.
    Uses binary search approach to find the correct starting page.
    """
    validated_pages = {}
    total_pages = len(doc)

    for section, toc_page in section_pages.items():
        # Check if processing should be stopped
        if hasattr(st.session_state, 'processing_active'
                   ) and not st.session_state.processing_active:
            st.warning("Page validation interrupted")
            return validated_pages if validated_pages else None

        st.info(f"Validating section {section} (ToC suggests page {toc_page})")

        # Create more precise regex pattern to find this specific section number and avoid subsections
        # This uses word boundaries and negative lookbehind to ensure we don't match subsections
        pattern = fr"(?<![0-9\.])(?:\b|^)({re.escape(section)})(?:\s+|\.\s+)"

        # Also create a header pattern to look for typical section heading formats
        header_pattern = fr"(?:Section|Chapter)?\s*{re.escape(section)}(?:\s+|\.\s+)([^\n\r]+)"

        # Initial search range: +/- 20 pages from ToC suggestion
        search_start = max(0, toc_page - 20)
        search_end = min(total_pages - 1, toc_page + 20)

        # Binary search to narrow down the correct page
        found_page = None
        while search_start <= search_end:
            # Check if processing should be stopped
            if hasattr(st.session_state, 'processing_active'
                       ) and not st.session_state.processing_active:
                st.warning("Page validation interrupted")
                return validated_pages if validated_pages else None

            mid = (search_start + search_end) // 2

            # Check current page and adjacent pages
            search_range = range(max(0, mid - 2), min(total_pages, mid + 3))
            for page_num in search_range:
                page_text = doc[page_num].get_text()

                # First try the stricter pattern
                match = re.search(pattern, page_text)
                if match:
                    matched_section = match.group(1)
                    # Double check that we matched the exact section number
                    if matched_section == section:
                        st.success(
                            f"Found section {section} on page {page_num+1} (PDF page)"
                        )
                        found_page = page_num
                        break

                # If stricter pattern didn't match, try the header pattern
                match = re.search(header_pattern, page_text)
                if match:
                    # We found something that looks like a section header
                    st.success(
                        f"Found section {section} header on page {page_num+1} (PDF page)"
                    )
                    found_page = page_num
                    break

            if found_page is not None:
                break

            # If we haven't found it yet, continue searching
            # Get text from mid page to determine search direction
            mid_text = doc[mid].get_text()

            # Try to determine if we need to search higher or lower pages
            # This is a heuristic - if we don't find pattern, check if page has
            # lower-numbered sections (search higher) or higher-numbered sections (search lower)
            if section == start_section:
                # Extract all section-like numbers from the page
                # We're looking for patterns like "X.Y" where X and Y are numbers
                section_matches = re.findall(r'\b(\d+\.\d+)\b', mid_text)

                if section_matches:
                    # Check if any extracted sections match our target exactly
                    exact_match = section in section_matches
                    if exact_match:
                        # We found our section but didn't catch it earlier - check this page more carefully
                        st.info(
                            f"Found section number but missed the heading pattern, rechecking page {mid+1}"
                        )
                        found_page = mid
                        break

                    # Get numeric representation of our target section (e.g., 6.1 -> 6.1)
                    section_parts = [
                        float(part) for part in section.split('.')
                    ]
                    section_num = section_parts[0] + 0.1 * section_parts[
                        1] if len(section_parts) > 1 else float(
                            section_parts[0])

                    # Compare with sections on the current page
                    page_sections = []
                    for sec in section_matches:
                        # Skip subsections (like 5.6.1) by checking the format
                        if sec.count('.') > 1:
                            continue

                        sec_parts = [float(part) for part in sec.split('.')]
                        if len(sec_parts) > 1:
                            page_sections.append(sec_parts[0] +
                                                 0.1 * sec_parts[1])
                        else:
                            page_sections.append(float(sec_parts[0]))

                    if page_sections:
                        max_page_section = max(page_sections)
                        min_page_section = min(page_sections)

                        if min_page_section > section_num:
                            # All sections on this page come after our target
                            search_end = mid - 1  # Search lower
                        elif max_page_section < section_num:
                            # All sections on this page come before our target
                            search_start = mid + 1  # Search higher
                        else:
                            # Our section should be on this page or nearby
                            # Try a more detailed linear search of adjacent pages
                            found = False
                            for adj_page in range(max(0, mid - 3),
                                                  min(total_pages, mid + 4)):
                                adj_text = doc[adj_page].get_text()
                                if re.search(pattern, adj_text) or re.search(
                                        header_pattern, adj_text):
                                    found_page = adj_page
                                    found = True
                                    break

                            if found:
                                break

                            # If still not found, expand search range
                            search_start = max(0, toc_page - 40)
                            search_end = min(total_pages - 1, toc_page + 40)
                    else:
                        # No valid sections on this page, try linear search
                        search_start = max(0, mid - 40)
                        search_end = min(total_pages - 1, mid + 40)
                else:
                    # No sections found on this page, try a wider search
                    # First try higher pages (more common)
                    higher_found = False
                    for page_num in range(mid + 3, min(total_pages, mid + 10)):
                        page_text = doc[page_num].get_text()
                        if re.search(pattern, page_text) or re.search(
                                header_pattern, page_text):
                            found_page = page_num
                            higher_found = True
                            break

                    if not higher_found:
                        # Then try lower pages
                        for page_num in range(max(0, mid - 10), mid - 2):
                            page_text = doc[page_num].get_text()
                            if re.search(pattern, page_text) or re.search(
                                    header_pattern, page_text):
                                found_page = page_num
                                break

                    if found_page is not None:
                        break
                    else:
                        # If still not found, expand search range
                        search_start = max(0, toc_page - 40)
                        search_end = min(total_pages - 1, toc_page + 40)
            else:
                # For other sections, consider relative position to first section
                if found_page is None:
                    # If we haven't found the page yet, try a linear search as fallback
                    for page_num in range(search_start, search_end + 1):
                        page_text = doc[page_num].get_text()
                        if re.search(pattern, page_text) or re.search(
                                header_pattern, page_text):
                            found_page = page_num
                            break
                break  # Exit while loop if we're not on the first section

        if found_page is not None:
            # Store the validated page number
            validated_pages[section] = found_page
        else:
            # Fallback to ToC suggestion if validation fails
            st.warning(
                f"Couldn't validate section {section}. Using ToC page {toc_page}"
            )
            validated_pages[section] = toc_page

    # Calculate offset between ToC page and actual page
    if start_section in validated_pages and start_section in section_pages:
        offset = validated_pages[start_section] - section_pages[
            start_section] + 1
        st.info(f"Page offset detected: {offset} pages")

        # Apply offset to all pages that weren't directly validated
        for section, page in section_pages.items():
            if section not in validated_pages:
                validated_pages[section] = page + offset

    return validated_pages


def render_summary(summary):
    """Display the summary with proper Markdown formatting"""
    if not summary:
        st.warning("No summary content to display")
        return

    with st.container():
        st.markdown(summary, unsafe_allow_html=False)

        # Add some visual separation
        st.markdown("---")
        st.caption("Summary generated by Fast Reader")


def extract_pdf_range(pdf_file, start_page, end_page):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    output = fitz.open()  # New empty PDF

    for i in range(start_page - 1, end_page):  # Pages are 0-indexed
        output.insert_pdf(doc, from_page=i, to_page=i)

    buffer = io.BytesIO()
    output.save(buffer)
    output.close()
    buffer.seek(0)
    return buffer


# --- Streamlit UI ---

if __name__ == "__main__":
    import ui
    ui.main_ui()
