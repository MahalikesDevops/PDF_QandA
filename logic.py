import os
import tempfile
import re
from llama_index.core import SimpleDirectoryReader
from llama_cloud_services import LlamaParse
import ollama

CACHE_PATH = "parsed_doc.md"

# Initialize the LlamaParse parser
parser = LlamaParse(
    result_type="markdown",
    complemental_formatting_instruction="""
    The provided document is a research article with Images, Tables, and Text.
    Parse all the data in the file as it is.
    Perform as best as possible to extract all the flow diagrams, figures, and tables by performing OCR if required.
    """,
    api_key=os.getenv("LLAMAPARSE_API_KEY") or "llx-uympZGcliIFtYFwc5v9ybDITKhviYv4AizAyKwfETrQHm2SG"
)

async def parse_pdf_content(file_content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    file_extractor = {".pdf": parser}
    parsed_docs = SimpleDirectoryReader(
        input_files=[tmp_path],
        file_extractor=file_extractor
    ).load_data()

    output = ""
    for doc in parsed_docs:
        text = doc.text.strip()
        metadata = doc.metadata or {}
        page = metadata.get("page_label", "unknown")

        if re.search(r"\|.+\|", text) and re.search(r"-{3,}", text):
            lines = text.splitlines()
            table_title = lines[0] if not lines[0].startswith("|") else "Untitled Table"
            table_lines = [line for line in lines if line.startswith("|")]

            headings = [h.strip() for h in table_lines[0].strip("|").split("|")]
            values = [
                [col.strip() for col in row.strip("|").split("|")]
                for row in table_lines[2:]
            ]

            output += f"""
========================================
Content-Type: table
Table Title: {table_title}
No Rows: {len(values)}
No Columns: {len(headings)}
Column Headings: {", ".join(headings)}
Column Values:
""" + "\n".join([", ".join(row) for row in values]) + f"""
Page: {page}
========================================
"""
        else:
            lines = text.splitlines()
            title = lines[0] if len(lines) > 1 and len(lines[0]) < 100 else "Untitled Section"
            body = "\n".join(lines[1:]) if len(lines) > 1 else text

            output += f"""
========================================
Content-Type: plain-text
Title: {title}
Content: {body.strip()}
Page: {page}
========================================
"""

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(output.strip())

    return output.strip()


def answer_question(question: str) -> str:
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError("Please parse a PDF first.")

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        context = f.read()

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions from research papers."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    return response["message"]["content"]
