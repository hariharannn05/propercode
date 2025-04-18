from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

load_dotenv()

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load embedding model (low memory)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Load FAISS and doc_map
index = faiss.read_index("faiss_index.idx")
with open("doc_map.json") as f:
    doc_map = json.load(f)

class QueryRequest(BaseModel):
    query: str

PROMPT_TEMPLATE = """
You are an AI assistant providing accurate information from the company website.
Use the following website content to answer the user's question:

{context}

User Query: {query}

AI Response:
"""

# Keywords
COURSE_QUERY_KEYWORDS = ["courses", "training", "learning", "education", "curriculum"]
FOUNDER_QUERY_KEYWORDS = ["founder", "ceo"]
COURSE_RESPONSE = "Available courses: Azure, GenAI, Chip Design, 5G, Cyber Security, HPC, Quantum Computing, Data Science."
FOUNDER_RESPONSE = "Ganesan Narayanasamy"

def retrieve_relevant_docs(query, top_k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [doc_map[str(idx)] for idx in indices[0] if str(idx) in doc_map]

def generate_response(query, context):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in COURSE_QUERY_KEYWORDS):
        return COURSE_RESPONSE
    if any(keyword in query_lower for keyword in FOUNDER_QUERY_KEYWORDS):
        return FOUNDER_RESPONSE
    if not context.strip():
        return "The given website content does not provide information."

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    headers = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

@app.post("/chat")
def chat(request: QueryRequest):
    try:
        query = request.query
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in COURSE_QUERY_KEYWORDS):
            return {"response": COURSE_RESPONSE}
        if any(keyword in query_lower for keyword in FOUNDER_QUERY_KEYWORDS):
            return {"response": FOUNDER_RESPONSE}

        docs = retrieve_relevant_docs(query)
        if not docs:
            return {"response": "This content is not in the website."}
        context = " ".join([d["text"] for d in docs])
        answer = generate_response(query, context)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
