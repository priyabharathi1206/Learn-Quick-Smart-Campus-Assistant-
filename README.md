# Smart Campus Assistant – LearnQuick

Smart Campus Assistant (LearnQuick) is an AI-powered learning platform designed to help students understand study materials more effectively. The system processes uploaded documents and provides intelligent features such as question answering, topic-wise keywords, MCQ generation, summarization, and mind map visualization using modern NLP and LLM techniques.

---

## 🚀 Key Features

- 📄 Upload study materials (PDF, DOCX, PPTX)
- ❓ Ask questions using Retrieval-Augmented Generation (RAG)
- 📝 Automatic MCQ generation with correct answers
- 🧠 Topic-wise keyword extraction
- ✍️ Automatic text summarization
- 🌐 Web-based interface using FastAPI and Jinja2

---

## 🛠 Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn

### AI & NLP
- Sentence Transformers
- FAISS (Vector Similarity Search)
- Groq LLM API (LLaMA models)

### Frontend
- HTML
- Jinja2 Templates
- JavaScript

### File Processing
- PDF: pdfplumber
- DOCX: python-docx
- PPTX: python-pptx

---

## 📁 Project Structure

smart-campus-assistant/
│
├── main.py # FastAPI application & API routes
├── server.py # Core AI logic (RAG, MCQ, topics, mind map)
├── requirements.txt # Project dependencies
│
├── templates/ # HTML templates
│ ├── learnquick.html
│ ├── ask.html
│ ├── mcq.html
│ ├── summarization.html
│ ├── topicskey.html
│ └── feature.html
│
├── uploaded_files/ # Uploaded study materials
│ 
│
├── .env # Environment variables (API keys)
├── .gitignore
└── README.md