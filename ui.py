# ui.py (Updated)
import streamlit as st
import fitz
from app import process_large_pdf, extract_sections_with_llm, summarize_with_llm, chunk_text, render_summary
from structure_analyzer import extract_toc_and_sample, learn_textbook_structure
import time
import pdfkit

if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False


def summary_to_pdf(summary_text):
    try:
        return pdfkit.from_string(summary_text, False)
    except ImportError:
        st.warning(
            "PDF generation requires pdfkit. Install with: pip install pdfkit")
        return None


def download_pdf(summary, file_name):
    # Markdown download
    st.download_button(label="📝 Download Summary (Markdown)",
                       data=summary,
                       file_name=f"{file_name}.md",
                       mime="text/markdown")

    # PDF download
    try:
        pdf_data = summary_to_pdf(summary)
        st.download_button(label="📄 Download Summary (PDF)",
                           data=pdf_data,
                           file_name=f"{file_name}.pdf",
                           mime="application/pdf")
    except Exception as e:
        st.warning("PDF generation failed. Is wkhtmltopdf installed?")


def main_ui():
    st.title("Fast Reader")
    st.write(
        "Upload a textbook PDF and specify the section range you want to summarize."
    )

    with st.sidebar:
        st.header("Settings")
        chapter_prefix = st.text_input(
            "Chapter prefix (optional)",
            help=
            "Enter the chapter number (e.g., '6') to summarize the full chapter."
        )
        llm_model = st.selectbox(
            "LLM Model", ["gpt-4o-mini", "gpt-3.5-turbo"],
            help="Select the language model for summarization")

    uploaded_file = st.file_uploader("Upload a textbook PDF", type="pdf")
    section_range = st.text_input(
        "Enter section range (e.g. 16.1–16.3), end section is non-inclusive:",
        help=
        "Use format 'start–end' with en dash or hyphen. Leave blank to summarize full chapter if chapter prefix is set."
    )

    section_range_valid = section_range or chapter_prefix

    start_page = st.number_input("Start Page", min_value=1, step=1)
    end_page = st.number_input("End Page", min_value=1, step=1)

    if uploaded_file and start_page and end_page and end_page >= start_page:
        if st.button("📄 Download Selected Chapter (PDF)"):
            from app import extract_pdf_range  # Make sure this is defined in `app.py`
            pdf_bytes = extract_pdf_range(uploaded_file, int(start_page),
                                          int(end_page))
            st.download_button(label="Download Split PDF",
                               data=pdf_bytes,
                               file_name="chapter_extract.pdf",
                               mime="application/pdf")

    if uploaded_file:
        st.info(
            "Large PDFs will be processed in chunks to avoid memory issues.")
        col1, col2 = st.columns([1, 1])
        process_button = col1.button("Process PDF",
                                     disabled=not section_range_valid)
        stop_button = col2.button(
            "Stop Processing", disabled=not st.session_state.processing_active)

        if stop_button:
            st.session_state.processing_active = False
            st.warning("Processing stopped by user")
            st.experimental_rerun()

        if process_button:
            st.session_state.processing_active = True
            with st.spinner("Extracting text from PDF..."):
                try:
                    doc = fitz.open(stream=uploaded_file.read(),
                                    filetype="pdf")
                    st.session_state.doc = doc

                    full_text = process_large_pdf(uploaded_file)
                    st.session_state.full_text = full_text

                    toc_text, sample_text = extract_toc_and_sample(doc)
                    structure_insight = learn_textbook_structure(
                        toc_text, sample_text)
                    st.session_state.structure_insight = structure_insight

                    if not section_range and chapter_prefix:
                        st.info(
                            f"📘 You requested to summarize the entire Chapter {chapter_prefix}."
                        )

                    sections, llm_success = extract_sections_with_llm(
                        doc,
                        section_range=section_range,
                        structure_insight=structure_insight,
                        model=llm_model,
                        chapter_prefix=chapter_prefix)

                    if llm_success:
                        st.success("Successfully extracted sections using AI!")
                        st.session_state.sections = sections
                    else:
                        st.error(
                            "AI extraction failed. Try entering a different section range or chapter prefix."
                        )

                    st.session_state.processing_active = False

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.session_state.processing_active = False

    if 'sections' in st.session_state and st.session_state.sections:
        st.write("---")
        with st.form("summary_form"):
            st.write("## Generate Summary")
            summarize_method = st.radio("Summarization Method",
                                        ["Single Section", "All Sections"],
                                        horizontal=True)
            section_keys = list(st.session_state.sections.keys())
            section_keys.sort()

            if summarize_method == "Single Section":
                selected_section = st.selectbox("Section to summarize",
                                                section_keys)
            else:
                selected_section = None

            language = st.selectbox(
                "Language", ["English", "Spanish", "French", "Chinese"])
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
            submitted = st.form_submit_button("Generate Summary")

        if submitted:
            if summarize_method == "Single Section":
                text = st.session_state.sections[selected_section]['content']
                summary = summarize_with_llm(text, model, language)
                st.write(f"### Summary of Section {selected_section}")
                render_summary(summary)
                download_pdf(summary, f"summary_{selected_section}")
            else:
                all_summaries = []
                for key in section_keys:
                    text = st.session_state.sections[key]['content']
                    summary = summarize_with_llm(text, model, language)
                    if summary:
                        all_summaries.append(f"## Section {key}\n\n{summary}")

                combined_summary = "\n\n---\n\n".join(all_summaries)
                st.write("### All Sections Summary")
                render_summary(combined_summary)
                download_pdf(combined_summary, "all_sections_summary")


def process_single_section(sections, section_key, model, language):
    """Process a single section and generate its summary"""
    progress_container = st.empty()
    progress_container.info("⏳ Initializing summarization process...")

    try:
        selected_section = sections.get(section_key)

        if not selected_section:
            st.error(
                f"❌ Section {section_key} not found in sections dictionary")
            return
        elif not isinstance(selected_section, dict):
            st.error(
                f"❌ Section {section_key} has invalid format: {type(selected_section)}"
            )
            return

        # Update progress
        progress_container.info(
            f"⏳ Preparing to summarize section {section_key}...")

        # Get content
        if "content" not in selected_section:
            st.error(f"❌ No content found in section {section_key}")
            st.write(f"Available keys: {', '.join(selected_section.keys())}")
            return

        section_text = selected_section.get("content", "")

        if not section_text or len(section_text) < 10:
            st.error("❌ Text content is too short to summarize")
            return

        # Content preview
        with st.expander(f"📄 Section {section_key} Content Preview",
                         expanded=False):
            st.text(section_text[:500] +
                    "..." if len(section_text) > 500 else section_text)

        # Update progress
        progress_container.info(
            f"⏳ Analyzing {len(section_text):,} characters of text...")
        time.sleep(0.5)  # Brief pause for visual feedback

        # Results container
        results_container = st.container()
        with results_container:
            st.write(f"### 📚 Section {section_key} Summary")

            # Generate summary with progress updates
            progress_container.info(
                "⏳ Sending to AI model for summarization...")
            generate_and_display_summary(section_text,
                                         section_key,
                                         model,
                                         language,
                                         container=results_container)

        # Final progress update
        progress_container.success(
            f"✅ Successfully summarized section {section_key}")

    except Exception as e:
        st.error(f"❌ Error during summarization: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def process_all_sections(sections, section_keys, model, language):
    """Process all sections in the extracted range and generate summaries for each"""
    st.write("## 📚 Processing All Sections")
    status_container = st.empty()
    status_container.info(
        f"⏳ Preparing to summarize {len(section_keys)} sections...")

    # Create a progress bar
    progress_bar = st.progress(0)

    # Create containers for each section
    section_containers = {}
    for i, key in enumerate(section_keys):
        section_containers[key] = st.container()
        with section_containers[key]:
            st.write(f"### Section {key}")
            st.info("⏳ Queued for processing...")

    # Process each section
    for i, key in enumerate(section_keys):
        # Update overall status
        status_container.info(
            f"⏳ Processing section {key} ({i+1} of {len(section_keys)})...")

        with section_containers[key]:
            st.empty()  # Clear the waiting message
            st.write(f"### 📄 Section {key}")

            try:
                section = sections.get(key)
                if not section or not isinstance(
                        section, dict) or "content" not in section:
                    st.error(f"❌ Invalid section data for {key}")
                    continue

                section_text = section.get("content", "")
                if not section_text or len(section_text) < 10:
                    st.warning(
                        f"⚠️ Section {key} has insufficient text content")
                    continue

                with st.expander("Content Preview", expanded=False):
                    st.text(section_text[:500] +
                            "..." if len(section_text) > 500 else section_text)

                st.info(f"⏳ Processing {len(section_text):,} characters...")

                # Generate summary for this section
                generate_and_display_summary(section_text,
                                             key,
                                             model,
                                             language,
                                             container=section_containers[key])

            except Exception as e:
                st.error(f"❌ Error processing section {key}: {str(e)}")

            # Update progress
            progress = (i + 1) / len(section_keys)
            progress_bar.progress(progress)

            # Brief pause for visual feedback
            time.sleep(0.3)

    # Complete
    progress_bar.progress(1.0)
    status_container.success(
        f"✅ Completed summarization of all {len(section_keys)} sections!")

    # Add a download all button
    if hasattr(st.session_state, 'all_summaries'):
        combined_text = "\n\n---\n\n".join([
            f"# Section {key}\n\n{summary}"
            for key, summary in st.session_state.all_summaries.items()
        ])
        st.download_button("📥 Download All Summaries",
                           combined_text,
                           file_name="all_section_summaries.md",
                           mime="text/markdown")


def generate_and_display_summary(text,
                                 section_key,
                                 model,
                                 language,
                                 container=None):
    """Generate and display a summary for the given text"""
    ctx = container or st

    # Initialize all_summaries in session state if not present
    if 'all_summaries' not in st.session_state:
        st.session_state.all_summaries = {}

    if len(text) > 12000:
        ctx.info("⏳ Using chunked summarization for large text...")
        chunks = chunk_text(text)

        # Create a status element for updates
        status = ctx.empty()
        status.info(
            f"⏳ Created {len(chunks)} chunks. Beginning summarization...")

        # Use status for updates during summarization
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            status.info(f"⏳ Summarizing chunk {i+1} of {len(chunks)}...")
            summary = summarize_with_llm(chunk, model=model, language=language)
            if summary:
                chunk_summaries.append(summary)

        if chunk_summaries:
            # Simply combine all chunk summaries with section breaks
            summary = "\n\n---\n\n".join(chunk_summaries)
        else:
            summary = "No summary generated"

        # Clear the status when done
        status.empty()
    else:
        with ctx.status("⏳ Generating summary...") as status:
            summary = summarize_with_llm(text, model=model, language=language)

    if summary:
        # Store in session state for download all feature
        st.session_state.all_summaries[section_key] = summary

        ctx.success("✅ Summary generated successfully!")

        # Display the summary
        summary_container = ctx.container()
        with summary_container:
            st.write("#### Summary")
            render_summary(summary)

            # Add download button for individual summary
            cols = st.columns([3, 1])
            cols[0].download_button("📥 Download This Summary",
                                    summary,
                                    file_name=f"summary_{section_key}.md",
                                    mime="text/markdown")
            cols[1].write(f"({len(summary):,} characters)")
    else:
        ctx.error("❌ Failed to generate summary")

    # Show helpful information
    with st.expander("Help & Tips"):
        st.markdown("""
        ### How to use this app:
        1. Upload your textbook PDF
        2. Enter a section range (e.g., "16.1–16.3")
        3. Click "Process PDF" to extract the sections
        4. Click "Get Summary" to generate a summary of the selected sections
        
        ### Tips:
        - If sections aren't detected correctly, try specifying a chapter prefix in the sidebar
        - For very large PDFs, processing might take a few minutes
        - Use the "Stop Processing" button if you need to interrupt a long-running process
        - The summary can be downloaded as a text file
        """)
