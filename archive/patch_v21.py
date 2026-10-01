from pathlib import Path
import shutil
import re


FILE = Path("entity_extractor.py")
BACKUP = Path("entity_extractor_V2_1_PATCH_BACKUP.py")


if not FILE.exists():
    raise FileNotFoundError(
        "entity_extractor.py was not found in the current directory."
    )


# ------------------------------------------------------------
# Safety backup
# ------------------------------------------------------------

shutil.copy2(FILE, BACKUP)

print(f"Backup created: {BACKUP}")


source = FILE.read_text(encoding="utf-8")


# ============================================================
# FIX 1 + FIX 2
# Replace medication status function
# ============================================================

start_marker = "def detect_medication_status("
end_marker = "\n\n# -------------------------------------------\n# Assertion detection (Task 3)"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start == -1 or end == -1:
    raise RuntimeError(
        "Could not safely locate detect_medication_status(). "
        "No changes were made."
    )


new_medication_status = r'''def detect_medication_status(
    text: str,
    med_text: str,
    med_start: int,
    med_end: int,
    all_medications: list
) -> str:
    """
    Determine whether a medication is currently active.

    Handles:
        - explicit negation
        - stopped/discontinued medication
        - previously used medication
        - local pronoun references such as:
              "warfarin ... stopped it"

    Returns:
        "ABSENT"
        "PRESENT"
        None
    """

    text_lower = text.lower()
    med_lower = med_text.lower().strip()

    # --------------------------------------------------------
    # Local context
    #
    # We intentionally avoid searching the entire document
    # for medication status.  Status should be determined from
    # the local clause/sentence containing the medication.
    # --------------------------------------------------------

    sentence_start = max(
        text_lower.rfind(".", 0, med_start) + 1,
        text_lower.rfind(";", 0, med_start) + 1
    )

    sentence_end_candidates = [
        p for p in [
            text_lower.find(".", med_end),
            text_lower.find(";", med_end)
        ]
        if p != -1
    ]

    sentence_end = (
        min(sentence_end_candidates)
        if sentence_end_candidates
        else len(text_lower)
    )

    local_before = text_lower[sentence_start:med_start]
    local_after = text_lower[med_end:sentence_end]

    # --------------------------------------------------------
    # Explicit medication-name negation anywhere in the
    # local sentence/clause.
    # --------------------------------------------------------

    negation_patterns = [
        r"\bnot\s+(?:currently\s+)?taking\s+",
        r"\bnot\s+(?:currently\s+)?using\s+",
        r"\bno\s+longer\s+(?:taking|using|on)\s+",
        r"\bstopped\s+(?:taking|using)\s+",
        r"\bdiscontinued\s+",
        r"\bheld\s+",
        r"\bwithheld\s+",
    ]

    for pattern in negation_patterns:

        full_pattern = pattern + re.escape(med_lower)

        if re.search(full_pattern, text_lower[sentence_start:sentence_end]):
            return "ABSENT"

    # --------------------------------------------------------
    # Medication followed immediately by:
    #
    #   "was stopped"
    #   "has been discontinued"
    #   "was held"
    # --------------------------------------------------------

    post_med_patterns = [
        r"^\s*(?:was|has been|had been)\s+"
        r"(?:stopped|discontinued|held|withheld)",

        r"^\s*(?:but\s+)?(?:is|was|has been)\s+"
        r"(?:not\s+taking|not\s+using)"
    ]

    for pattern in post_med_patterns:

        if re.search(pattern, local_after):
            return "ABSENT"

    # --------------------------------------------------------
    # Pronoun-based status.
    #
    # Example:
    #
    #   "previously used warfarin but stopped it"
    #
    # We only apply this when the pronoun occurs locally after
    # THIS medication and before another medication name.
    #
    # This allows multiple medications in the same sentence
    # while avoiding the old global len(all_medications)==1
    # restriction.
    # --------------------------------------------------------

    pronoun_negation_patterns = [
        r"\bstopped\s+(?:taking\s+)?it\b",
        r"\bdiscontinued\s+it\b",
        r"\bheld\s+it\b",
        r"\bwithheld\s+it\b",
        r"\bnot\s+taking\s+it\b",
        r"\bnot\s+using\s+it\b",
        r"\bno\s+longer\s+taking\s+it\b",
        r"\bno\s+longer\s+using\s+it\b",
    ]

    # Find the earliest other medication after this medication.
    # A pronoun status signal must occur before another medication
    # reference to remain local to this medication.
    next_med_start = len(local_after)

    for other in all_medications:

        if other is None:
            continue

        other_start = other.get("start")

        if other_start is None:
            continue

        if other_start <= med_start:
            continue

        relative = other_start - med_end

        if relative >= 0:
            next_med_start = min(
                next_med_start,
                relative
            )

    pronoun_context = local_after[:next_med_start]

    for pattern in pronoun_negation_patterns:

        if re.search(pattern, pronoun_context):
            return "ABSENT"

    # --------------------------------------------------------
    # Explicit PRESENT patterns local to this medication.
    # --------------------------------------------------------

    present_patterns = [
        r"\bcurrently\s+taking\s+$",
        r"\bcurrently\s+using\s+$",
        r"\bis\s+taking\s+$",
        r"\bis\s+using\s+$",
        r"\bare\s+taking\s+$",
        r"\bare\s+using\s+$",
        r"\breports\s+taking\s+$",
        r"\breports\s+using\s+$",
        r"\buses\s+$",
        r"\busing\s+$",
        r"\btaking\s+$",
    ]

    for pattern in present_patterns:

        if re.search(pattern, local_before):
            return "PRESENT"

    # --------------------------------------------------------
    # Previously-used language immediately before medication
    # is not considered current.
    # --------------------------------------------------------

    previous_patterns = [
        r"\bpreviously\s+(?:used|taking|using|on)\s+$",
        r"\bformerly\s+(?:used|taking|using|on)\s+$",
        r"\bin\s+the\s+past\s+(?:used|taking|using|on)\s+$",
    ]

    for pattern in previous_patterns:

        if re.search(pattern, local_before):
            return "ABSENT"

    return None
'''


source = source[:start] + new_medication_status + source[end:]


# ============================================================
# FIX 3
# Replace extract_tests() with WordPiece-aware merging
# ============================================================

start_marker = "def extract_tests(text: str):"
end_marker = "\n\n# -------------------------------------------\n# Final entity extraction (Task 6)"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start == -1 or end == -1:
    raise RuntimeError(
        "Could not safely locate extract_tests(). "
        "No changes were made."
    )


new_extract_tests = r'''def extract_tests(text: str):

    results = test_ner(text)

    tests = []

    # --------------------------------------------------------
    # Collect consecutive test predictions.
    #
    # The test model can return WordPiece fragments such as:
    #
    #   trans
    #   ##thoracic echocardiogram
    #
    # or:
    #
    #   lip
    #   ##id panel
    #
    # These must be reconstructed before boundary cleaning.
    # --------------------------------------------------------

    current_start = None
    current_end = None

    for entity in results:

        if entity["entity_group"] != "test":

            if current_start is not None:

                raw_text = text[
                    current_start:current_end
                ]

                test_text, clean_start, clean_end = (
                    clean_entity_boundary(
                        text,
                        raw_text,
                        current_start,
                        current_end
                    )
                )

                test_text = test_text.lower()

                test_text = test_text.replace(
                    " x - ray",
                    " x-ray"
                )

                test_text = test_text.replace(
                    "x - ray",
                    "x-ray"
                )

                test_text = test_text.strip(
                    " ,.;:-"
                )

                if test_text:
                    duplicate = any(
                        item["text"].lower() == test_text
                        for item in tests
                    )

                    if not duplicate:
                        tests.append({
                            "text": test_text,
                            "start": clean_start,
                            "end": clean_end
                        })

                current_start = None
                current_end = None

            continue

        start = entity["start"]
        end = entity["end"]

        # ----------------------------------------------------
        # Start first test span
        # ----------------------------------------------------

        if current_start is None:

            current_start = start
            current_end = end
            continue

        # ----------------------------------------------------
        # Merge adjacent / overlapping WordPiece pieces.
        # ----------------------------------------------------

        if start <= current_end + 1:

            current_end = max(
                current_end,
                end
            )

        else:

            raw_text = text[
                current_start:current_end
            ]

            test_text, clean_start, clean_end = (
                clean_entity_boundary(
                    text,
                    raw_text,
                    current_start,
                    current_end
                )
            )

            test_text = test_text.lower()

            test_text = test_text.replace(
                " x - ray",
                " x-ray"
            )

            test_text = test_text.replace(
                "x - ray",
                "x-ray"
            )

            test_text = test_text.strip(
                " ,.;:-"
            )

            if test_text:

                duplicate = any(
                    item["text"].lower() == test_text
                    for item in tests
                )

                if not duplicate:

                    tests.append({
                        "text": test_text,
                        "start": clean_start,
                        "end": clean_end
                    })

            current_start = start
            current_end = end

    # --------------------------------------------------------
    # Flush final test
    # --------------------------------------------------------

    if current_start is not None:

        raw_text = text[
            current_start:current_end
        ]

        test_text, clean_start, clean_end = (
            clean_entity_boundary(
                text,
                raw_text,
                current_start,
                current_end
            )
        )

        test_text = test_text.lower()

        test_text = test_text.replace(
            " x - ray",
            " x-ray"
        )

        test_text = test_text.replace(
            "x - ray",
            "x-ray"
        )

        test_text = test_text.strip(
            " ,.;:-"
        )

        if test_text:

            duplicate = any(
                item["text"].lower() == test_text
                for item in tests
            )

            if not duplicate:

                tests.append({
                    "text": test_text,
                    "start": clean_start,
                    "end": clean_end
                })

    return tests
'''


source = source[:start] + new_extract_tests + source[end:]


# ============================================================
# Write modified file
# ============================================================

FILE.write_text(source, encoding="utf-8")

print("V2.1 patch applied successfully.")
print(f"Modified: {FILE}")
print(f"Backup:   {BACKUP}")