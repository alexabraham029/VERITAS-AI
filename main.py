from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, structured AI assistant. Respond clearly using markdown. Use bullet points or numbered lists where helpful. Avoid hallucinating information."),
    ("user", "{question}")
])

# Request schema
class Query(BaseModel):
    question: str
    temperature: float = 0.7
    max_tokens: int = 512
    groq_api_key: str  # ✅ User must provide their Groq key

@app.post("/generate")
def generate_response(payload: Query):
    if not payload.groq_api_key:
        raise HTTPException(status_code=400, detail="Groq API key is required.")

    try:
        # Dynamically use user's Groq key
        chat = ChatGroq(
            model="gemma-7b-it",  # Or mixtral-8x7b, llama3-8b-8192 etc.
            api_key=payload.groq_api_key,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )
        chain = prompt | chat | StrOutputParser()
        response = chain.invoke({"question": payload.question})
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
