from preprocess import clean_text
from entity_extractor import extract_entities


class ClinicalTextProcessingAgent:

    def process(self, text: str):

        print("\n========== FASTAPI DEBUG ==========")

        cleaned_text = clean_text(text)

        print("CLEANED TEXT:")
        print(repr(cleaned_text))

        entities = extract_entities(cleaned_text)

        print("\nENTITIES FROM EXTRACTOR:")
        print(entities)

        print("===================================\n")

        return {
            "clinical_text": text,
            "diseases": entities["diseases"],
            "symptoms": entities["symptoms"],
            "medications": entities["medications"],
            "tests": entities["tests"]
        }