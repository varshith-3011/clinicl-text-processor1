from entity_extractor import extract_tests

text = (
    "The doctor requested a transthoracic echocardiogram, "
    "pulmonary function test, liver function test, "
    "fasting blood glucose, and lipid panel."
)

print("\nRAW EXTRACTED TEST ENTITIES")
print("=" * 70)

results = extract_tests(text)

for item in results:
    print(item)

print("=" * 70)