# Chat with Multiple PDFs 📚

A sophisticated PDF interaction tool that allows users to engage in conversations with multiple PDF documents simultaneously using the power of Gemini AI and LangChain.

![Chat with Multiple PDFs Interface](UI.jpg)

## Features

- Upload and process multiple PDF documents
- Interactive chat interface with AI-powered responses
- Real-time document processing
- Efficient text chunking and vector storage
- Session-based chat history

## Technical Architecture

**Core Components:**
- Frontend: Streamlit
- Language Model: Google's Gemini Pro
- Vector Store: FAISS
- Embeddings: Google Generative AI
- Text Processing: LangChain, PyPDF2

## Workflow

**Document Processing Flow:**
PDF upload → Text extraction → Chunking → Embedding generation → Vector storage

**Query Processing Flow:**
Question input → Vector search → Context retrieval → Response generation → Display

### Document Upload & Processing
- Users upload PDF files through the sidebar interface
- System extracts text using PyPDF2
- Text is split into manageable chunks using RecursiveCharacterTextSplitter
- Chunks are converted to embeddings using Google's embedding model
- Embeddings are stored in FAISS vector store

### Question-Answering System
- User inputs questions about the PDFs
- System performs similarity search on stored vectors
- Relevant context is retrieved and processed
- Gemini Pro generates detailed, context-aware responses

## Installation
```
git clone [repository-url]
cd chat-with-pdfs
pip install -r requirements.txt
```

## Environment Setup
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage
```
streamlit run ChatWithPDFs.py
```

## Key Components Explained
**PDF Processing:**
```
def get_pdf_text(pdf_docs):
text = ""
for pdf in pdf_docs:
pdf_reader = PdfReader(pdf)
for page in pdf_reader.pages:
text += page.extract_text()
```
**Text Chunking:**
```
def get_text_chunks(text):
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=10000,
chunk_overlap=1000
)
return text_splitter.split_text(text)
```
**Vector Store Implementation:**
```
def get_vector_store(text_chunks):
embeddings = GoogleGenerativeAIEmbeddings(
model = "models/embedding-001"
)
vector_store = FAISS.from_texts(
text_chunks,
embedding=embeddings
)
```
## Dependencies
- streamlit
- google-generativeai
- python-dotenv
- langchain
- PyPDF2
- chromadb
- faiss-cpu
- langchain_google_genai

## Special Thanks
Shoutout to my friend, Vedanth Sirimalla (@VedSirimalla) for the collaborative effort in bringing this project to life! 

Thank You. Let’s keep learning and growing together!
