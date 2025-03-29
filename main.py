from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
import io

# Load environment variables
load_dotenv()

# Initialize API
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Set up Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Supabase client
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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

# Generate embeddings
def generate_embedding(text: str) -> List[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# Search diagrams function
def search_diagrams(query: str, top_k: int = 3):
    try:
        supabase = get_supabase()
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # SQL to search for similar diagrams using cosine similarity
        sql = """
        SELECT 
            d.id, 
            d.filename, 
            d.description, 
            d.source_pdf, 
            d.page_number, 
            d.storage_path,
            1 - (e.embedding <=> ?) as similarity
        FROM 
            diagrams d
        JOIN 
            diagram_embeddings e ON d.id = e.diagram_id
        ORDER BY 
            similarity DESC
        LIMIT ?
        """
        
        # Execute query
        response = supabase.table("diagram_embeddings").select(
            "*, diagrams(id, filename, description, source_pdf, page_number, storage_path)"
        ).execute()
        
        # Get storage URL for the bucket
        bucket_url = f"{SUPABASE_URL}/storage/v1/object/public/diagrams"
        
        # Perform vector similarity search (we're doing this in Python since RPC may be more complex)
        # In production, consider using a Supabase Edge Function or RPC
        results = []
        for item in response.data:
            embedding = np.array(item['embedding'])
            query_emb = np.array(query_embedding)
            similarity = 1 - np.dot(embedding, query_emb) / (np.linalg.norm(embedding) * np.linalg.norm(query_emb))
            
            diagram = item['diagrams']
            results.append({
                'id': diagram['id'],
                'diagram_path': f"{bucket_url}/{diagram['storage_path']}",
                'description': diagram['description'],
                'source_pdf': diagram['source_pdf'],
                'page_number': diagram['page_number'],
                'similarity': similarity
            })
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        print(f"Error searching diagrams: {e}")
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
    try:
        supabase = get_supabase()
        
        # Get diagram details
        response = supabase.table("diagrams").select("storage_path").eq("id", diagram_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Diagram not found")
        
        storage_path = response.data[0]["storage_path"]
        
        # Redirect to Supabase Storage
        bucket_url = f"{SUPABASE_URL}/storage/v1/object/public/diagrams"
        return RedirectResponse(f"{bucket_url}/{storage_path}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving diagram: {str(e)}")

@app.post("/upload-diagram")
async def upload_diagram(file: UploadFile = File(...)):
    try:
        supabase = get_supabase()
        
        # Read file content
        file_content = await file.read()
        
        # Generate a unique filename to avoid collisions
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Upload to Supabase Storage
        supabase.storage.from_("diagrams").upload(
            path=unique_filename,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        # Return the path information
        return {
            "status": "success", 
            "message": f"Diagram uploaded: {file.filename}", 
            "storage_path": unique_filename,
            "url": f"{SUPABASE_URL}/storage/v1/object/public/diagrams/{unique_filename}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading diagram: {str(e)}")

@app.post("/upload-processed-data")
async def upload_processed_data(file: UploadFile = File(...)):
    try:
        # Read JSON data
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        supabase = get_supabase()
        
        # Process and store in Supabase
        success_count = 0
        for item in data:
            try:
                # Extract filename from diagram_path
                filename = os.path.basename(item["diagram_path"])
                
                # Insert diagram metadata
                response = supabase.table("diagrams").insert({
                    "filename": filename,
                    "description": item["description"],
                    "source_pdf": item["source_pdf"],
                    "page_number": item["page_number"],
                    "storage_path": filename  # assuming the file will be uploaded with the same name
                }).execute()
                
                if not response.data:
                    continue
                
                # Get the diagram ID
                diagram_id = response.data[0]["id"]
                
                # Generate embedding for the description
                embedding = generate_embedding(item["description"])
                
                # Store embedding
                supabase.table("diagram_embeddings").insert({
                    "diagram_id": diagram_id,
                    "embedding": embedding
                }).execute()
                
                success_count += 1
            
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
        
        return {"status": "success", "message": f"Processed {success_count} out of {len(data)} diagram entries"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "online", "message": "Diagram RAG Chatbot server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)