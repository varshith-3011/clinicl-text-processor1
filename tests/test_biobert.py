from entity_extractor import extract_entities
import json


tests = [
    "patient is having fever and that he is not having diabetes but he says that he is having hypertension. He says that he is using dolo and paracetamol since a week and did a mri and shortness of breath but denies abdominal pain. She is currently taking furosemide and iron supplements, while a previously prescribed antibiotic has been discontinued. The doctor ordered a CBC, serum creatinine, electrolyte panel, urinalysis, and renal ultrasound, but no CT scan was performed."
]


for text in tests:

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

    result = extract_entities(text)

    # Print only the public API fields
    public_result = {
        "clinical_text": result["clinical_text"],
        "diseases": result["diseases"],
        "symptoms": result["symptoms"],
        "medications": result["medications"],
        "tests": result["tests"]
    }

    print(json.dumps(public_result, indent=2))