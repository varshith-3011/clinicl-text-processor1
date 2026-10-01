from preprocess import clean_text
from entity_extractor import extract_entities


class ClinicalTextProcessingAgent:

    def process(self, text: str):

        cleaned_text = clean_text(text)

        entities = extract_entities(cleaned_text)

        return {
            "clinical_text": text,
            "diseases": entities["diseases"],
            "symptoms": entities["symptoms"],
            "medications": entities["medications"],
            "tests": entities["tests"]
        }