from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import fitz  # PyMuPDF
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import os
import uuid

app = FastAPI()

# Initialize LLM and Vector Store
embedding_model = OpenAIEmbeddings()
llm_model = OpenAI(temperature=0)
collection_name = "pdf_collection"
vector_store = None

class SearchRequest(BaseModel):
    question: str

def print_chunk_metadata(content_type, title, content, metadata, table_data=None):
    print("========================================")
    print(f"Content-Type: {content_type}")
    print(f"Title: {title if title else 'N/A'}")

    if content_type == "table" and table_data:
        print(f"Table Title: {title if title else 'N/A'}")
        print(f"No Rows: {table_data['n_rows']}")
        print(f"No Columns: {table_data['n_cols']}")
        print(f"Column Headings: {', '.join(table_data['headers'])}")
        for i, row in enumerate(table_data['rows']):
            print(f"Row {i+1} Values: {', '.join(row)}")
    else:
        print(f"Text: {content.strip()}")

    print(f"Page: {metadata['page']}")
    print("========================================\n")

def extract_pdf_chunks(file_path: str):
    doc = fitz.open(file_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # text block
                text = "".join([line["spans"][0]["text"] for line in block["lines"] if line["spans"]])
                if text.strip():
                    metadata = {"page": page_num + 1}
                    print_chunk_metadata("plain-text", None, text, metadata)
                    chunks.append({
                        "content": text.strip(),
                        "metadata": metadata
                    })
            elif block["type"] == 1:  # image (possible table image – skip or OCR needed)
                continue
            elif block["type"] == 2:  # assume this might be a table-like text block (rare in fitz)
                continue  # advanced parsing needed

    return chunks

def ingest_to_vector_store(chunks: List[dict]):
    global vector_store
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=collection_name
    )

def search_from_vector_store(question: str):
    if not vector_store:
        return {"error": "No PDF has been ingested yet."}

    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([f"[Page {doc.metadata['page']}] {doc.page_content}" for doc in docs])

    prompt = (
        f"Answer this question: {question}\n\n"
        f"Using the context below:\n\n{context}\n\n"
        f"Be specific and mention page numbers in the response."
    )

    answer = llm_model(prompt)
    return {
        "answer": answer,
        "sources": [
            {
                "page": doc.metadata["page"],
                "excerpt": doc.page_content[:200]
            } for doc in docs
        ]
    }

@app.post("/api/v1/ingest")
async def ingest(file: UploadFile):
    file_ext = os.path.splitext(file.filename)[-1]
    if file_ext.lower() != ".pdf":
        return {"error": "Only PDF files are supported."}

    temp_filename = f"/tmp/{uuid.uuid4()}.pdf"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    chunks = extract_pdf_chunks(temp_filename)
    ingest_to_vector_store(chunks)

    return {"message": "PDF successfully ingested and logged."}

@app.post("/api/v1/search")
def search(req: SearchRequest):
    return search_from_vector_store(req.question)
