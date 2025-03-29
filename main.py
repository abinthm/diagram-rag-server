from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np

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
    prompt: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    diagram_path: Optional[str] = None
    session_id: str

# Generate embeddings for vector search
def generate_embedding(text: str) -> List[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# Search diagrams function
# Search diagrams function
def search_diagrams(query: str, top_k: int = 1):
    try:
        supabase = get_supabase()
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Fetch all diagram embeddings
        response = supabase.table("diagram_embeddings").select(
            "diagram_id, embedding"
        ).execute()
        
        # Manual vector similarity calculation
        results = []
        for item in response.data:
            try:
                # Get the embedding data
                embedding_data = item['embedding']
                
                # Check if embedding is a string and convert it to a list if needed
                if isinstance(embedding_data, str):
                    # If it starts with '[' and ends with ']', it's likely a string representation of a list
                    if embedding_data.startswith('[') and embedding_data.endswith(']'):
                        # Convert string representation to actual list of floats
                        import ast
                        embedding_data = ast.literal_eval(embedding_data)
                    else:
                        # Skip this item if the embedding format is unexpected
                        print(f"Unexpected embedding format for diagram {item.get('diagram_id')}")
                        continue
                
                # Convert embeddings to numpy arrays of float type
                db_embedding = np.array(embedding_data, dtype=np.float32)
                query_emb = np.array(query_embedding, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(db_embedding, query_emb) / (np.linalg.norm(db_embedding) * np.linalg.norm(query_emb))
                
                # Get diagram details
                diagram_response = supabase.table("diagrams").select(
                    "id, filename, description, source_pdf, page_number, storage_path"
                ).eq("id", item['diagram_id']).execute()
                
                if diagram_response.data:
                    diagram = diagram_response.data[0]
                    
                    # Get storage URL
                    bucket_url = f"{SUPABASE_URL}/storage/v1/object/public/diagrams"
                    diagram_path = f"{bucket_url}/{diagram['storage_path']}"
                    
                    results.append({
                        'id': diagram['id'],
                        'diagram_path': diagram_path,
                        'description': diagram.get('description', ''),
                        'source_pdf': diagram.get('source_pdf', ''),
                        'page_number': diagram.get('page_number', ''),
                        'similarity': float(similarity)
                    })
            except Exception as e:
                print(f"Error processing embedding for diagram {item.get('diagram_id')}: {e}")
                continue
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        print(f"Error searching diagrams: {e}")
        return []

# Simple in-memory chat history (for production, use Redis or DB)
chat_sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Get chat history
        history = chat_sessions[session_id]
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # Determine if the query might need a diagram
        needs_diagram_prompt = f"""
        Analyze this educational query: "{request.prompt}"
        
        Does this query potentially need or reference a diagram, chart, architecture, 
        or visual representation to properly explain the concept? Answer with YES or NO only.
        """
        
        needs_diagram_response = model.generate_content(needs_diagram_prompt).text.strip()
        
        # Search for relevant diagrams if needed
        retrieved_context = ""
        diagram_path = None
        
        if "YES" in needs_diagram_response.upper():
            diagram_results = search_diagrams(request.prompt)
            
            if diagram_results and diagram_results[0]['similarity'] > 0.6:
                best_match = diagram_results[0]
                diagram_path = best_match['diagram_path']
                
                retrieved_context = f"""
                I found a relevant diagram that helps explain this concept.
                Description: {best_match['description']}
                Source: {best_match.get('source_pdf', 'Educational material')}, 
                page {best_match.get('page_number', 'N/A')}
                
                Please refer to the attached diagram while reading my explanation.
                """
        
        # Start chat with history
        chat = model.start_chat(history=history)
        
        # Add system instructions to the user message instead
        system_instructions = """
        You are an educational assistant that helps explain concepts clearly.
        When diagrams are available, refer to them in your explanation to enhance understanding.
        Keep explanations clear, concise, and tailored to educational purposes.
        """
        
        # Construct prompt with system instructions and retrieved diagram info
        user_message = f"""
        {system_instructions}
        
        User query: {request.prompt}
        
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
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "online", "message": "Diagram RAG Chatbot server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)