from entity_extractor import (
    disease_ner,
    symptom_ner,
    medication_ner,
    test_ner
)


text = (
    "The patient has atrial fibrillation and chronic kidney disease "
    "but no history of asthma. She reports palpitations, dizziness, "
    "and fatigue but denies chest pain. She is currently taking "
    "apixaban and amlodipine, previously used warfarin but stopped it, "
    "and is not taking aspirin. The physician ordered an ECG, "
    "transthoracic echocardiogram, renal function test, lipid panel, "
    "and chest X-ray, but no CT scan was performed. "
    "Pulmonary embolism cannot be ruled out."
)


print("\n" + "=" * 70)
print("DISEASE RAW OUTPUT")
print("=" * 70)

for item in disease_ner(text):
    print(item)


print("\n" + "=" * 70)
print("SYMPTOM RAW OUTPUT")
print("=" * 70)

for item in symptom_ner(text):
    print(item)


print("\n" + "=" * 70)
print("MEDICATION RAW OUTPUT")
print("=" * 70)

for item in medication_ner(text):
    print(item)


print("\n" + "=" * 70)
print("TEST RAW OUTPUT")
print("=" * 70)

for item in test_ner(text):
    print(item)