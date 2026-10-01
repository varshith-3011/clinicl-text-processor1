from fastapi import FastAPI
from models import ClinicalInput
from agent import ClinicalTextProcessingAgent

app = FastAPI(
    title="Clinical Text Processing Agent",
    description="Input Agent for Graph-RAG Clinical Decision Support System",
    version="1.0.0"
)

# Create an instance of our agent
agent = ClinicalTextProcessingAgent()


# Home Endpoint
@app.get("/")
def home():
    return {
        "message": "Clinical Text Processing Agent is running!"
    }


# Health Check Endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agent": "Clinical Text Processing Agent",
        "version": "1.0.0"
    }


# Main Processing Endpoint
@app.post("/process-text")
def process_text(data: ClinicalInput):
    return agent.process(data.text)