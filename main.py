from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import json
import shutil
import chromadb
from chromadb.utils import embedding_functions
import uuid
from datetime import datetime

app = FastAPI(title="Diagram RAG Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings and configurations
DIAGRAMS_FOLDER = "diagrams_store"
VECTOR_DB_FOLDER = "vector_db"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Ensure storage directories exist
os.makedirs(DIAGRAMS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Set up Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Setup ChromaDB
def get_chroma_client():
    client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)
    
    # Get embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="diagram_descriptions",
        embedding_function=embedding_function
    )
    
    return collection

# Models
class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    session_id: Optional[str] = None

class DiagramSearchRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatResponse(BaseModel):
    response: str
    diagram_path: Optional[str] = None
    chat_id: str

# Search diagrams function
def search_diagrams(query: str, top_k: int = 3):
    try:
        collection = get_chroma_client()
        
        # Search for diagrams
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                'diagram_path': results['metadatas'][0][i]['diagram_path'],
                'description': results['documents'][0][i],
                'source_pdf': results['metadatas'][0][i]['source_pdf'],
                'page_number': results['metadatas'][0][i]['page_number'],
                'relevance_score': results.get('distances', [[0]*len(results['ids'][0])])[0][i]
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Error searching vector database: {e}")
        return []

# Simple in-memory chat history (for production, use Redis or DB)
chat_sessions = {}

# Routes
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Get or create chat session
    chat_id = request.chat_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Store chat history
    history = chat_sessions[session_id]
    
    # Initialize Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    # Determine if query needs a diagram
    needs_diagram_prompt = f"""
    Analyze this user query: "{request.query}"
    
    Does this query potentially need or reference a diagram, chart, architecture, 
    or visual representation? Answer with YES or NO only.
    """
    
    needs_diagram_response = model.generate_content(needs_diagram_prompt).text.strip()
    
    # Search for relevant diagrams if needed
    retrieved_context = ""
    diagram_path = None
    
    if "YES" in needs_diagram_response:
        diagram_results = search_diagrams(request.query)
        
        if diagram_results:
            best_match = diagram_results[0]
            diagram_path = best_match['diagram_path']
            
            retrieved_context = f"""
            Relevant diagram found:
            Description: {best_match['description']}
            Source: {best_match['source_pdf']}, page {best_match['page_number']}
            """
    
    # Generate response with RAG context
    chat = model.start_chat(history=history)
    
    # Construct prompt with retrieved diagram info
    user_message = f"""
    {request.query}
    
    {retrieved_context if retrieved_context else ""}
    """
    
    # Get response
    response = chat.send_message(user_message)
    
    # Update chat history
    chat_sessions[session_id] = chat.history
    
    # Prepare response
    return ChatResponse(
        response=response.text,
        diagram_path=diagram_path,
        chat_id=chat_id
    )

@app.post("/search-diagrams")
async def search_diagrams_endpoint(request: DiagramSearchRequest):
    results = search_diagrams(request.query, request.top_k)
    return {"results": results}

@app.get("/diagram/{diagram_id}")
async def get_diagram(diagram_id: str):
    diagram_path = os.path.join(DIAGRAMS_FOLDER, diagram_id)
    
    # Check if file exists with different extensions
    for ext in ['.png', '.jpg', '.jpeg', '.gif']:
        full_path = f"{diagram_path}{ext}"
        if os.path.exists(full_path):
            return FileResponse(full_path)
    
    raise HTTPException(status_code=404, detail="Diagram not found")

# New endpoint for uploading diagrams
@app.post("/upload-diagram")
async def upload_diagram(file: UploadFile = File(...)):
    try:
        # Get the filename and create the file path
        filename = file.filename
        file_path = os.path.join(DIAGRAMS_FOLDER, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "message": f"Diagram uploaded: {filename}", "path": file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading diagram: {str(e)}")

@app.post("/upload-processed-data")
async def upload_processed_data(file: UploadFile = File(...)):
    # Create temp file
    temp_file_path = f"temp_{uuid.uuid4()}.json"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read json data
    try:
        with open(temp_file_path, "r") as f:
            data = json.load(f)
        
        # Process and store in ChromaDB
        collection = get_chroma_client()
        
        ids = []
        documents = []
        metadatas = []
        
        for item in data:
            ids.append(item["id"])
            documents.append(item["description"])
            metadatas.append({
                "source_pdf": item["source_pdf"],
                "page_number": str(item["page_number"]), 
                "diagram_path": item["diagram_path"],
                "source_page": f"{item['source_pdf']} - Page {item['page_number']}"
            })
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        # Clean up
        os.remove(temp_file_path)
        
        return {"status": "success", "message": f"Processed {len(ids)} diagram entries"}
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "online", "message": "Diagram RAG Chatbot server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)