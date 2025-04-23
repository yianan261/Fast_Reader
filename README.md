# Fast Reader

Fast Reader is a Streamlit application designed to process and summarize PDF documents efficiently. It leverages OpenAI's language models to extract and summarize sections from textbooks or other structured documents.

## Features

- **Smart Structure Analysis**: Analyzes textbook structure first for more accurate section detection
- **Precise Section Extraction**: Uses AI and binary search to accurately locate section ranges with validation
- **Intelligent Chunking**: Smart text segmentation that avoids breaking paragraphs or mathematical expressions
- **Hierarchical Summarization**: Tree-based approach enables summarizing very long sections without hitting token limits
- **Multi-format Support**: Handles math expressions (LaTeX), code blocks, and tables in summaries
- **Batch Processing**: Summarize single sections or entire chapter ranges
- Extracts sections from PDFs using AI and binary search to accurately locate section ranges.
- Summarizes text using OpenAI's language models with support for math, code, and tables.
- Provides a user-friendly interface for uploading PDFs and selecting sections to summarize.
- Supports batch summarization of multiple sections.

## Key Improvements

| Feature                  | Impact                                                              |
| ------------------------ | ------------------------------------------------------------------- |
| Structure analysis first | More accurate section detection across different textbooks          |
| Smarter chunking         | Avoids breaking paragraphs or math, improves summarization quality  |
| Tree summarization       | Enables summarizing very long sections without hitting token limits |
| Section range validation | Catches embedded sub-sections you might have missed                 |

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- [OpenAI Python Client](https://github.com/openai/openai-python)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Fast_Reader
   ```

2. Create a virtual environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

   On Windows use:

   ```bash
   myenv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501` to access the app.

3. Upload a PDF document using the file uploader.

4. Enter the section range you wish to summarize and click "Process PDF".

5. Once sections are extracted, choose to summarize a single section or all sections in the range.

6. View and download the generated summaries.

## Contribution Guidelines

We welcome contributions to improve Fast Reader! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- Prototyped by Cursor.
- Thanks to OpenAI for providing the language models.
- Thanks to the Streamlit community for their support and resources.
