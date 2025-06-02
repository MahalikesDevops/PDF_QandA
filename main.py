from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from logic import parse_pdf_content, answer_question
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

app = FastAPI()

# to parse PDF content and return structured output
@app.post("/api/v1/ingest")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        structured_output = await parse_pdf_content(content)
        return PlainTextResponse(content=structured_output)
    except Exception as e:
        return PlainTextResponse(content=f"Error: {str(e)}", status_code=500)

#to answer a question based on the parsed PDF content
@app.post("/api/v1/search")
async def ask_question(question: str = Form(...)):
    try:
        answer = answer_question(question)
        return PlainTextResponse(content=answer)
    except Exception as e:
        return PlainTextResponse(content=f"Error: {str(e)}", status_code=500)
