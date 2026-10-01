from pydantic import BaseModel

class ClinicalInput(BaseModel):
    text: str