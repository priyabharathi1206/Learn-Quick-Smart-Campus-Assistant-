from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
from fastapi.responses import JSONResponse


from services import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    clean_text,
    split_into_chunks,
    create_vector_store,
    ask_question,
    get_mcqs,
    check_answer,
    summarize_text,
    extract_topics_and_keywords,
    extract_topics_hierarchy
)

app = FastAPI(title="Smart Campus Assistant")

# Static + Templates 
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Global Store 
ALL_TEXT = ""
CHUNKS = []
INDEX = None
EMBEDDINGS = None
TREE_DATA=None


# PAGES 

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("learnquick.html", {"request": request})


@app.get("/features")
async def features_page(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})


@app.get("/ask-page")
async def ask_page(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})


@app.get("/mcq-page")
async def mcq_page(request: Request):
    return templates.TemplateResponse("mcq.html", {"request": request})


@app.get("/summary-page")
async def summary_page(request: Request):
    return templates.TemplateResponse("summarization.html", {"request": request})



# FILE UPLOAD 
@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global ALL_TEXT, CHUNKS, INDEX, EMBEDDINGS

    ALL_TEXT = ""

    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        file_path = f"{upload_dir}/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".pdf"):
            ALL_TEXT += extract_text_from_pdf(file_path)

        elif file.filename.endswith(".docx"):
            ALL_TEXT += extract_text_from_docx(file_path)

        elif file.filename.endswith(".pptx"):
            ALL_TEXT += extract_text_from_pptx(file_path)

    cleaned = clean_text(ALL_TEXT)
    CHUNKS = split_into_chunks(cleaned)

    EMBEDDINGS, INDEX = create_vector_store(CHUNKS)
    print("Successfull")
     # ---- FAISS Index creation ----
    # INDEX = create_faiss_index(CHUNKS)
    

    return {
        "message": "Files processed successfully",
        "total_characters": len(ALL_TEXT),
        "total_chunks": len(CHUNKS)
    }


#  ASK QUESTION 
@app.post("/ask")
async def ask(question: str = Form(...)):
    if INDEX is None:
        return {"error": "Upload files first!"}

    answer = ask_question(question, CHUNKS, INDEX)
    return {"question": question, "answer": answer}


# GENERATE MCQs 

@app.post("/mcq")
async def generate_mcq(num_questions: int = Form(5)):
    if INDEX is None:
        return JSONResponse({"error": "Upload files first!"})

    try:
        mcqs = get_mcqs(CHUNKS, INDEX, num_q=num_questions)

        return JSONResponse({
            "mcqs": mcqs
        })

    except Exception as e:
        print("MCQ ERROR:", e)
        return JSONResponse({"error": "MCQ generation failed"})
#  CHECK MCQ ANSWER 
@app.post("/check")
async def check_mcq_answer(
    mcq_text: str = Form(...),
    q_no: int = Form(...),
    user_answer: str = Form(...)
):
    result = check_answer(mcq_text, q_no, user_answer)
    return {"result": result}


#  SUMMARY
@app.post("/summary")
async def get_summary():
    if ALL_TEXT == "":
        return {"error": "Upload files first!"}

    summary = summarize_text(ALL_TEXT)
    return {"summary": summary}

#  EXTRACT TOPICS & KEYWORDS 
@app.get("/topics")
async def topics_page(request: Request):
    topics_keywords = extract_topics_and_keywords(ALL_TEXT) if ALL_TEXT else []
    error_msg = None if topics_keywords else "Upload files first or extraction failed"

    return templates.TemplateResponse(
        "topicskey.html", 
        {"request": request, "topics": topics_keywords, "error": error_msg}
    )


# Mind Map Endpoint 
@app.get("/mindmap")
async def mindmap_page(request: Request):
    if ALL_TEXT == "":
        return templates.TemplateResponse(
            "mindmap.html",
            {"request": request, "tree_data": [], "error": "Upload files first!"}
        )

    return templates.TemplateResponse(
        "mindmap.html",
        {"request": request, "tree_data": TREE_DATA, "error": None}
    )
