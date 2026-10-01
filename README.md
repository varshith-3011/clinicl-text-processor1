# Clinical Text Processing Agent

## Overview

The Clinical Text Processing Agent processes unstructured clinical text and extracts important medical information such as diseases, symptoms, medications, and clinical tests.

It is designed as the input-processing component of the Graph-RAG Clinical Decision Support System.

## Features

- Clinical text preprocessing
- Disease extraction
- Symptom extraction
- Medication extraction
- Clinical test extraction
- Clinical assertion and negation detection
- Structured JSON output
- FastAPI-based REST API

## Installation

### 1. Create and activate the virtual environment

```bash
python -m venv venv311
```

On Windows:

```bash
venv311\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The application will run at:

```text
http://127.0.0.1:8000
```

## API Endpoints

### Home

```text
GET /
```

Returns a message confirming that the Clinical Text Processing Agent is running.

### Health Check

```text
GET /health
```

Returns the health status and version of the agent.

### Process Clinical Text

```text
POST /process-text
```

Processes the provided clinical text and returns structured medical entities.

## Example Input

```json
{
  "text": "Patient has fever and cough. The patient is taking aspirin."
}
```

## Example Output

```json
{
  "clinical_text": "Patient has fever and cough. The patient is taking aspirin.",
  "diseases": [],
  "symptoms": [
    "fever",
    "cough"
  ],
  "medications": [
    "aspirin"
  ],
  "tests": []
}
```

## API Documentation

When the application is running, interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```
