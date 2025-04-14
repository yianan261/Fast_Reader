import streamlit as st
import fitz  # PyMuPDF
import re
import openai
import os
from dotenv import load_dotenv
import tempfile
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# --- Utility Functions ---


def extract_sections(text, chapter_pattern=None):
    """
    Extract sections from text based on chapter patterns.
    If no pattern is provided, it will try to detect common chapter numbering patterns.
    """
    sections = {}

    # Common patterns to try in order of specificity
    patterns = [
        # Pattern 1: Standard chapter-section format (e.g., "16.1 Title")
        r"\b(?:Chapter\s+)?(\d{1,3}\.\d{1,2}(?![0-9]))\s+([^\n]+)",  # Prevents matching decimals

        # Pattern 2: Section numbers with letters (e.g., "16.1a Title")
        r"\b(?:Chapter\s+)?(\d{1,3}\.\d{1,2}[a-z]?(?![0-9]))\s+([^\n]+)",

        # Pattern 3: Chapter and section separately numbered (e.g., "Chapter 16 Section 1")
        r"\bChapter\s+(\d{1,3})\s+Section\s+(\d{1,2})(?:\s+|:)([^\n]+)",

        # Pattern 4: Appendix format (e.g., "Appendix A.1")
        r"\bAppendix\s+([A-Z]\.\d{1,2})\s+([^\n]+)",

        # Pattern 5: Simple numbered sections (e.g., "Section 16")
        r"\bSection\s+(\d{1,3})(?:\s+|:)([^\n]+)"
    ]

    # If chapter pattern provided, create a custom pattern
    if chapter_pattern:
        # Ensure the pattern matches typical chapter.section format (e.g., 16.1) and not decimals
        custom_pattern = rf"\b({chapter_pattern}\.\d{{1,2}}(?![0-9]))\s+([^\n]+)"
        patterns.insert(0, custom_pattern)  # Try this pattern first

    # Try each pattern until we find matches
    found_matches = False
    matched_pattern = None

    for pattern_str in patterns:
        pattern = re.compile(pattern_str)
        matches = list(pattern.finditer(text))

        if matches:
            found_matches = True
            matched_pattern = pattern_str
            st.info(
                f"Found {len(matches)} sections using pattern: {pattern_str}")

            # Different handling based on pattern
            if "Chapter" in pattern_str and "Section" in pattern_str:
                # Handle "Chapter X Section Y" format
                for i in range(len(matches)):
                    start = matches[i].start()
                    end = matches[i + 1].start() if i + 1 < len(
                        matches) else len(text)
                    chapter_num = matches[i].group(1)
                    section_num = matches[i].group(2)
                    key = f"{chapter_num}.{section_num}"
                    title = matches[i].group(3).strip()
                    content = text[start:end].strip()
                    # Calculate approximate page numbers based on character position
                    approx_start_page = start // 3000  # Rough estimate of chars per page
                    approx_end_page = end // 3000
                    sections[key] = {
                        "title":
                        title,
                        "content":
                        content,
                        "start_pos":
                        start,
                        "end_pos":
                        end,
                        "approx_pages":
                        f"{approx_start_page+1}-{approx_end_page+1}"
                    }
            else:
                # Standard handling for other patterns
                for i in range(len(matches)):
                    start = matches[i].start()
                    end = matches[i + 1].start() if i + 1 < len(
                        matches) else len(text)
                    key = matches[i].group(1)
                    title = matches[i].group(2).strip()
                    content = text[start:end].strip()
                    # Calculate approximate page numbers
                    approx_start_page = start // 3000  # Rough estimate of chars per page
                    approx_end_page = end // 3000
                    sections[key] = {
                        "title":
                        title,
                        "content":
                        content,
                        "start_pos":
                        start,
                        "end_pos":
                        end,
                        "approx_pages":
                        f"{approx_start_page+1}-{approx_end_page+1}"
                    }

            # Found sections, no need to try other patterns
            break

    return sections, found_matches


def process_large_pdf(file, max_pages_per_chunk=50):
    """Process large PDFs by chunking to avoid memory issues"""
    try:
        # Create a temporary copy of the file since we need to read it twice
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # Debug message
        st.info("Starting PDF processing...")

        # Open the file from the temp path
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        full_text = ""

        # Simple processing without progress bar to avoid NoneType issues
        for i in range(0, total_pages, max_pages_per_chunk):
            # Check if processing should be stopped
            if hasattr(st.session_state, 'processing_active'
                       ) and not st.session_state.processing_active:
                st.warning("PDF processing interrupted")
                doc.close()
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return ""

            end_page = min(i + max_pages_per_chunk, total_pages)
            chunk_text = ""
            for page_num in range(i, end_page):
                chunk_text += doc[page_num].get_text()
            full_text += chunk_text

        # Clean up and return
        doc.close()
        os.unlink(tmp_path)  # Remove the temporary file
        st.success(f"Finished processing {total_pages} pages")
        return full_text
    except Exception as e:
        # Make sure we clean up even if there's an error
        st.error(f"Error processing PDF: {str(e)}")
        try:
            doc.close()
        except:
            pass
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""


def summarize_with_llm(text, model="gpt-4o-mini", language="English"):
    """Summarize text using LLM with support for math, code, tables."""
    try:
        if not text or len(text.strip()) < 10:
            st.error("Text is too short or empty to summarize")
            return None

        st.info(
            f"Sending {len(text):,} characters to {model} for summarization in {language}..."
        )

        prompt = f"""
You are an expert explainer and summarizer for textbooks. 

Summarize the following section clearly and comprehensively in {language}. Preserve the key concepts, examples, and explanations.
Highlight key terms and concepts in bold.

- Use **LaTeX** for math expressions (enclose with $$...$$).
- Format **code snippets** using Markdown code blocks (e.g., ```python).
- Retain headings and subheadings if present.
- Use Markdown tables if needed to summarize comparisons.
- Highlight important definitions and theorems.

Text to summarize:
{text}
"""

        # Use the new OpenAI client interface
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
        )

        summary = completion.choices[0].message.content
        st.success(f"Generated summary ({len(summary):,} characters)")
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None


def chunk_text(text, max_chunk_size=8000):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def summarize_chunks(chunks, model="gpt-4o-mini"):
    summaries = []
    for i, chunk in enumerate(chunks):
        st.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
        summary = summarize_with_llm(chunk, model)
        if summary:
            summaries.append(summary)
    return summaries


def summarize_merged_summary(summaries, model="gpt-4"):
    combined = "\n\n".join(
        [f"Part {i+1}:\n{s}" for i, s in enumerate(summaries)])
    st.info("Merging chunk summaries into final summary...")
    return summarize_with_llm(
        f"Here are summaries of several sections:\n\n{combined}\n\nPlease combine them into a single cohesive summary.",
        model)


def render_summary(summary_text):
    """
    Renders summary text with special handling for math expressions and code blocks
    """
    # Split by display math blocks ($$...$$)
    parts = re.split(r"(\$\$.*?\$\$)", summary_text, flags=re.DOTALL)
    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            math_expr = part.strip("$").strip()
            st.latex(math_expr)
        else:
            st.markdown(part, unsafe_allow_html=True)


def extract_sections_with_llm(doc, section_range, model="gpt-3.5-turbo"):
    """
    Fallback method that uses an LLM to identify section page ranges by analyzing the Table of Contents.
    Then validates and adjusts the page numbers to match the actual content in the PDF.
    """
    st.info("Attempting to locate sections using AI...")

    # Check if processing should be stopped
    if hasattr(st.session_state,
               'processing_active') and not st.session_state.processing_active:
        st.warning("Section extraction interrupted")
        return {}, False

    # Parse section range
    if "–" in section_range:  # en dash
        start_sec, end_sec = section_range.split("–")
    elif "-" in section_range:  # regular hyphen
        start_sec, end_sec = section_range.split("-")
    else:
        start_sec = end_sec = section_range  # Single section

    # Identify potential ToC pages
    toc_pages = []
    for i in range(len(doc)):
        # Check if processing should be stopped
        if hasattr(st.session_state, 'processing_active'
                   ) and not st.session_state.processing_active:
            st.warning("Section extraction interrupted")
            return {}, False

        text = doc[i].get_text()
        if "Contents" in text or "Table of Contents" in text:
            toc_pages.append(i)

    if not toc_pages:
        st.error("No Table of Contents found in the document.")
        return {}, False

    # Extract text from ToC pages
    toc_text = "\n\n".join([doc[i].get_text() for i in toc_pages])

    # Create the prompt
    prompt = f"""The following is the Table of Contents from a textbook PDF.

Please identify the starting pages for the following sections:
{', '.join([start_sec, end_sec])}

Return a mapping in this exact format:
```
section_number: start_page
section_number: start_page
...
```

If you can't find a section, indicate "Not found"

Text:
{toc_text}
"""

    try:
        # Log the prompt for debugging
        st.text_area("AI Prompt", prompt, height=200)

        # Use the new OpenAI client interface
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
        )

        result = completion.choices[0].message.content

        # Log the AI response for debugging
        st.text_area("AI Response", result, height=200)

        st.info("AI response received. Extracting section locations...")

        # Parse the LLM response to extract starting pages
        section_pages = {}

        # Look for patterns like "16.1: start_page"
        section_pattern = re.compile(r'([0-9.]+):\s*(\d+)')
        matches = section_pattern.findall(result)

        for match in matches:
            section = match[0]
            start_page = int(match[1])
            section_pages[section] = start_page

        if section_pages:
            st.success(f"AI found starting pages: {section_pages}")

            # Store original ToC page numbers for later calculations
            toc_start_page = section_pages.get(start_sec, None)
            toc_end_page = section_pages.get(end_sec, None)

            if toc_start_page is None or toc_end_page is None:
                st.warning("Missing page information for one or more sections")
                return section_pages, True

            # Calculate the length of the section in ToC pages
            toc_section_length = toc_end_page - toc_start_page

            # Validate and search for correct pages
            validated_pages = validate_and_adjust_pages(
                doc, section_pages, start_sec)
            if validated_pages:
                # Get the actual starting page that was found
                actual_start_page = validated_pages.get(start_sec, None)

                if actual_start_page is not None:
                    # Calculate actual end page using offset and section length
                    actual_end_page = actual_start_page + toc_section_length
                    st.success(
                        f"Calculated actual page range: {actual_start_page+1} to {actual_end_page+1}"
                    )

                    # Build proper section data structure
                    sections = {}
                    for section in [start_sec, end_sec]:
                        if section == start_sec:
                            start_pos = 0  # Placeholder
                            end_pos = 1  # Placeholder
                            page = actual_start_page
                            end_page = actual_end_page if section == end_sec else None
                        else:
                            page = actual_end_page
                            end_page = None
                            start_pos = 0  # Placeholder
                            end_pos = 1  # Placeholder

                        # Extract text for this section
                        section_text = ""
                        if section == start_sec and end_sec != start_sec:
                            # For start section, get text from this page to end section
                            for p in range(actual_start_page,
                                           actual_end_page + 1):
                                section_text += doc[p].get_text()
                        else:
                            # For other sections, just get text from their page
                            section_text = doc[page].get_text()

                        # Create section entry with required fields for summarization
                        sections[section] = {
                            "title":
                            f"Section {section}",
                            "content":
                            section_text,
                            "start_pos":
                            start_pos,
                            "end_pos":
                            end_pos,
                            "pdf_start_page":
                            page + 1,  # +1 for 1-indexed display
                            "pdf_end_page":
                            (end_page + 1) if end_page is not None else None,
                            "approx_pages":
                            f"{page+1}-{end_page+1}"
                            if end_page is not None else f"{page+1}"
                        }

                    return sections, True
                else:
                    st.warning(
                        "Couldn't find actual start page. Using AI-suggested pages instead."
                    )
            else:
                st.warning(
                    "Couldn't validate section pages. Using AI-suggested pages instead."
                )

            # Fallback: Use the original section_pages but convert to proper format
            sections = {}
            for section, page in section_pages.items():
                if section == end_sec and start_sec in section_pages:
                    # Calculate end page based on ToC
                    start_page = section_pages[start_sec]
                    section_length = page - start_page
                    end_page = page
                else:
                    end_page = None

                sections[section] = {
                    "title":
                    f"Section {section}",
                    "content":
                    doc[page].get_text(),
                    "start_pos":
                    0,  # Placeholder
                    "end_pos":
                    1,  # Placeholder
                    "pdf_start_page":
                    page + 1,  # +1 for 1-indexed display
                    "pdf_end_page":
                    (end_page + 1) if end_page is not None else None,
                    "approx_pages":
                    f"{page+1}-{end_page+1}"
                    if end_page is not None else f"{page+1}"
                }

            return sections, True
        else:
            st.warning("AI couldn't identify section starting pages")
            return {}, False

    except Exception as e:
        st.error(f"Error using AI to identify sections: {str(e)}")
        return {}, False


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

        # Create regex pattern to find this specific section number
        pattern = fr"\b{re.escape(section)}\b\s+"

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
                if re.search(pattern, page_text):
                    st.success(
                        f"Found section {section} on page {page_num+1} (PDF page)"
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
                # For first section, look for any section patterns
                mid_section_match = re.search(r"\b(\d+\.\d+)\b", mid_text)
                if mid_section_match:
                    mid_section = mid_section_match.group(1)
                    # Compare section numbers numerically
                    if float(mid_section.replace(' ', '')) < float(
                            section.replace(' ', '')):
                        search_start = mid + 1  # Search higher
                    else:
                        search_end = mid - 1  # Search lower
                else:
                    # If no section found, try both directions
                    # First try higher pages (more common)
                    higher_found = False
                    for page_num in range(mid + 3, min(total_pages, mid + 10)):
                        page_text = doc[page_num].get_text()
                        if re.search(pattern, page_text):
                            found_page = page_num
                            higher_found = True
                            break

                    if not higher_found:
                        # Then try lower pages
                        for page_num in range(max(0, mid - 10), mid - 2):
                            page_text = doc[page_num].get_text()
                            if re.search(pattern, page_text):
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
                        if re.search(pattern, page_text):
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


# --- Streamlit UI ---

if __name__ == "__main__":
    import ui
    ui.main_ui()
