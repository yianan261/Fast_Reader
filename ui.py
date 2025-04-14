import streamlit as st
import fitz  # PyMuPDF
from app import process_large_pdf, extract_sections_with_llm, summarize_with_llm, chunk_text, summarize_chunks, summarize_merged_summary, render_summary
import threading
import time

# Create a flag for controlling processing
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False


def main_ui():
    st.title("Fast Reader")
    st.write(
        "Upload a textbook PDF and specify the section range you want to summarize."
    )

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        chapter_prefix = st.text_input(
            "Chapter prefix (optional)",
            help=
            "Enter only the chapter number (e.g., '16' for chapter 16) to help with section detection. Do not enter a range here."
        )
        llm_model = st.selectbox(
            "LLM Model", ["gpt-4o-mini", "gpt-3.5-turbo"],
            help="Select the language model for summarization")

    # Main area
    uploaded_file = st.file_uploader("Upload a textbook PDF", type="pdf")
    section_range = st.text_input(
        "Enter section range (e.g. 16.1–16.3), end section is non-inclusive:",
        help="Use the format 'start–end' with an en dash or hyphen")

    # Validate section range
    section_range_valid = bool(
        section_range and
        ('-' in section_range or '–' in section_range or '.' in section_range))

    if uploaded_file:
        # Process PDF
        with st.expander("PDF Processing Details", expanded=False):
            st.info(
                "Large PDFs will be processed in chunks to avoid memory issues."
            )

        # Column layout for process and stop buttons
        col1, col2 = st.columns([1, 1])

        process_button = col1.button("Process PDF",
                                     disabled=not section_range_valid)
        stop_button = col2.button(
            "Stop Processing", disabled=not st.session_state.processing_active)

        # Show warning if section range is invalid
        if not section_range_valid and uploaded_file:
            st.warning(
                "Please enter a valid section range before processing (e.g., 16.1-16.3)"
            )

        if stop_button:
            st.session_state.processing_active = False
            st.warning("Processing stopped by user")
            st.experimental_rerun()

        if process_button and section_range_valid:
            # Set the processing flag to active
            st.session_state.processing_active = True

            with st.spinner("Extracting text from PDF..."):
                try:
                    # Open the document
                    doc = fitz.open(stream=uploaded_file.read(),
                                    filetype="pdf")
                    st.session_state.doc = doc  # Store in session state for later use

                    # Process PDF with stop check
                    full_text = process_large_pdf(uploaded_file)

                    if not st.session_state.processing_active:
                        st.warning("Processing stopped by user")
                        return

                    st.session_state.full_text = full_text

                    # Extract sections using LLM and ToC
                    sections, llm_success = extract_sections_with_llm(
                        doc, section_range, model="gpt-3.5-turbo")

                    if not st.session_state.processing_active:
                        st.warning("Processing stopped by user")
                        return

                    if llm_success:
                        st.success("Successfully extracted sections using AI!")
                    else:
                        st.error(
                            "AI extraction failed. Consider manually extracting the sections."
                        )

                    st.session_state.sections = sections

                    if not sections:
                        st.error(
                            "No sections found matching the pattern. Try a different chapter prefix."
                        )
                    else:
                        st.success(
                            f"Found {len(sections)} sections in the PDF.")
                        # Display available sections with page ranges
                        section_keys = list(sections.keys())
                        section_keys.sort()

                        # Create a formatted display of sections
                        st.subheader("Available Sections")
                        cols = st.columns([2, 3, 2])
                        cols[0].write("**Section**")
                        cols[1].write("**Title**")
                        cols[2].write("**PDF Page**")

                        # Display first 10 sections with details
                        for key in section_keys[:10]:
                            section = sections[key]
                            cols[0].write(key)

                            # Check if section has a title or use "Section {key}" as fallback
                            if isinstance(section,
                                          dict) and "title" in section:
                                section_title = section["title"][:50] + (
                                    "..."
                                    if len(section["title"]) > 50 else "")
                            else:
                                section_title = f"Section {key}"
                            cols[1].write(section_title)

                            # Display page numbers
                            if isinstance(section,
                                          dict) and "approx_pages" in section:
                                cols[2].write(section["approx_pages"])
                            elif isinstance(section, int):
                                # If section is just an integer page number
                                cols[2].write(
                                    f"{section+1}"
                                )  # +1 because PDF pages are 1-indexed
                            else:
                                cols[2].write("Unknown")

                        if len(section_keys) > 10:
                            st.info(
                                f"... and {len(section_keys) - 10} more sections"
                            )

                        # Add a note about page numbers
                        st.caption(
                            "Note: Page numbers are exact PDF page numbers found in the document."
                        )

                        # Store sections in session state for later use
                        st.session_state.sections = sections
                        st.session_state.section_keys = section_keys

                    # Reset processing flag when complete
                    st.session_state.processing_active = False

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.processing_active = False

        # Add summarize button OUTSIDE the process button condition
        if 'sections' in st.session_state and st.session_state.sections:
            st.write("---")

            # Use a form to avoid async issues
            with st.form("summary_form", clear_on_submit=False):
                st.write("## Generate Summary")
                st.write("Generate summaries for your extracted sections.")

                sections = st.session_state.sections
                section_keys = list(sections.keys())
                section_keys.sort()

                # Model selection
                selected_model = st.selectbox("Model",
                                              ["gpt-4o-mini", "gpt-3.5-turbo"],
                                              index=0)

                # Summarization method selection
                summarize_method = st.radio(
                    "Summarization Method:",
                    ["Single Section", "All Sections in Range"],
                    horizontal=True,
                    help=
                    "Choose to summarize one section or all sections in the extracted range"
                )

                # Section selection (only show if single section is selected)
                if summarize_method == "Single Section":
                    selected_section_key = st.selectbox("Section to summarize",
                                                        section_keys,
                                                        index=0)
                else:
                    # If all sections selected, no need for section dropdown
                    st.info(
                        f"Will summarize all {len(section_keys)} sections: {', '.join(section_keys)}"
                    )
                    selected_section_key = None

                # Submit button with clear indication
                submitted = st.form_submit_button("Generate Summary",
                                                  use_container_width=True,
                                                  type="primary")

            # Create a placeholder for submission feedback
            submission_feedback = st.empty()

            # Process outside the form
            if submitted:
                # Display prominent visual feedback
                submission_feedback.success(
                    "✅ Summary request received! Processing will begin shortly..."
                )

                # Add a divider
                st.divider()

                # Clear feedback after 3 seconds
                time.sleep(1)
                submission_feedback.empty()

                # Header for results
                st.subheader("💡 Generating Summaries")

                if summarize_method == "Single Section":
                    # Process single section
                    process_single_section(sections, selected_section_key,
                                           selected_model)
                else:
                    # Process all sections
                    process_all_sections(sections, section_keys,
                                         selected_model)


def process_single_section(sections, section_key, model):
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
                                         container=results_container)

        # Final progress update
        progress_container.success(
            f"✅ Successfully summarized section {section_key}")

    except Exception as e:
        st.error(f"❌ Error during summarization: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def process_all_sections(sections, section_keys, model):
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


def generate_and_display_summary(text, section_key, model, container=None):
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
            summary = summarize_with_llm(chunk, model=model)
            if summary:
                chunk_summaries.append(summary)

        if len(chunk_summaries) > 1:
            status.info("⏳ Merging summaries from all chunks...")
            summary = summarize_merged_summary(chunk_summaries, model=model)
        else:
            summary = chunk_summaries[
                0] if chunk_summaries else "No summary generated"

        # Clear the status when done
        status.empty()
    else:
        with ctx.status("⏳ Generating summary...") as status:
            summary = summarize_with_llm(text, model=model)

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
