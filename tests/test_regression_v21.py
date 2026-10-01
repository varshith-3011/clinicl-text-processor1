from entity_extractor import extract_entities


tests = [

    "The patient is taking metformin but is not taking aspirin.",

    "The patient denies fever but reports cough.",

    "The patient has hypertension but no history of diabetes.",

    "The patient was prescribed amoxicillin but is not currently taking it.",

    "No MRI was performed but a CBC was completed.",

    "The patient is currently taking apixaban and amlodipine, previously used warfarin but stopped it, and is not taking aspirin.",

    "The doctor requested a transthoracic echocardiogram, pulmonary function test, liver function test, fasting blood glucose, and lipid panel.",

    "The patient has atrial fibrillation and chronic kidney disease but no history of asthma. She reports palpitations, dizziness, and fatigue but denies chest pain. She is currently taking apixaban and amlodipine, previously used warfarin but stopped it, and is not taking aspirin. The physician ordered an ECG, transthoracic echocardiogram, renal function test, lipid panel, and chest X-ray, but no CT scan was performed. Pulmonary embolism cannot be ruled out.",

    "The patient reports nausea and vomiting without diarrhea.",

    "The patient may have pulmonary embolism, but deep vein thrombosis has been ruled out.",

]


for i, text in enumerate(tests, 1):

    print("\n" + "=" * 80)
    print(f"TEST {i}")
    print("=" * 80)
    print(text)
    print("-" * 80)

    result = extract_entities(text)

    print("Diseases   :", result["diseases"])
    print("Symptoms   :", result["symptoms"])
    print("Medications:", result["medications"])
    print("Tests      :", result["tests"])