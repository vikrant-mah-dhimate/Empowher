from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from rag_chatbot import store_document, generate_response

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload a PDF and store its content in ChromaDB"""
    file_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    store_document(file_path)
    return {"message": f"Stored {file.filename} in ChromaDB"}


@app.post("/chat")
def chat(request: QueryRequest):
    """Endpoint to generate a response from ChromaDB"""
    response = generate_response(request.query)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

