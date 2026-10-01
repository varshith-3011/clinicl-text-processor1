"""
Compare NER results with lowercased vs original-cased input
to determine whether lowercasing helps or hurts each model.
"""
import sys, io
sys.stderr = io.StringIO()

from transformers import pipeline

disease_ner = pipeline("token-classification", model="alvaroalon2/biobert_diseases_ner", aggregation_strategy="simple")
symptom_ner = pipeline("token-classification", model="BSC-NLP4BIA/multiclinner-en-symptom-BiomedBERT", aggregation_strategy="first")
medication_ner = pipeline("token-classification", model="jackleejm/distilbert-medication-ner", aggregation_strategy="simple")
test_ner = pipeline("token-classification", model="sschet/bert-base-uncased_clinical-ner", aggregation_strategy="simple")

clinical_texts = [
    "The patient has Hypertension and Type 2 Diabetes.",
    "Patient presents with Pneumonia and Chronic Obstructive Pulmonary Disease.",
    "The patient is taking Metformin and Lisinopril.",
    "The patient was prescribed Amoxicillin.",
    "An MRI and CBC were ordered.",
    "The patient denies Fever but reports Cough.",
    "Patient has a history of Congestive Heart Failure.",
    "The doctor suspects Tuberculosis.",
]

lines = []
def log(msg):
    lines.append(str(msg))

for text in clinical_texts:
    text_lower = text.lower()
    log(f"\n{'='*70}")
    log(f"ORIGINAL: {text}")
    log(f"LOWERED:  {text_lower}")
    
    # Disease NER
    d_orig = [(e['word'], round(float(e['score']),3)) for e in disease_ner(text) if e['entity_group'] == 'DISEASE']
    d_low  = [(e['word'], round(float(e['score']),3)) for e in disease_ner(text_lower) if e['entity_group'] == 'DISEASE']
    if d_orig or d_low:
        log(f"  Disease (original): {d_orig}")
        log(f"  Disease (lower):    {d_low}")
    
    # Symptom NER
    s_orig = [(e['word'], round(float(e['score']),3)) for e in symptom_ner(text) if e['entity_group'] == 'SYMPTOM']
    s_low  = [(e['word'], round(float(e['score']),3)) for e in symptom_ner(text_lower) if e['entity_group'] == 'SYMPTOM']
    if s_orig or s_low:
        log(f"  Symptom (original): {s_orig}")
        log(f"  Symptom (lower):    {s_low}")
    
    # Medication NER
    m_orig = [(e['word'], round(float(e['score']),3)) for e in medication_ner(text) if e['entity_group'] == 'DRUG']
    m_low  = [(e['word'], round(float(e['score']),3)) for e in medication_ner(text_lower) if e['entity_group'] == 'DRUG']
    if m_orig or m_low:
        log(f"  Medication (original): {m_orig}")
        log(f"  Medication (lower):    {m_low}")
    
    # Test NER
    t_orig = [(e['word'], round(float(e['score']),3)) for e in test_ner(text) if e['entity_group'] == 'test']
    t_low  = [(e['word'], round(float(e['score']),3)) for e in test_ner(text_lower) if e['entity_group'] == 'test']
    if t_orig or t_low:
        log(f"  Test (original): {t_orig}")
        log(f"  Test (lower):    {t_low}")

with open('casing_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("DONE - results in casing_results.txt")
