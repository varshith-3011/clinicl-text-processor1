import re
import logging
from transformers import pipeline
import spacy
from negspacy.negation import Negex


logger = logging.getLogger(__name__)


# -----------------------------
# Models
# -----------------------------

DISEASE_MODEL = "alvaroalon2/biobert_diseases_ner"
SYMPTOM_MODEL = "BSC-NLP4BIA/multiclinner-en-symptom-BiomedBERT"
MEDICATION_MODEL = "jackleejm/distilbert-medication-ner"
TEST_MODEL = "sschet/bert-base-uncased_clinical-ner"
ASSERTION_MODEL = "bvanaken/clinical-assertion-negation-bert"


# -----------------------------
# Load models only once
# -----------------------------

disease_ner = pipeline(
    "token-classification",
    model=DISEASE_MODEL,
    aggregation_strategy="simple"
)

symptom_ner = pipeline(
    "token-classification",
    model=SYMPTOM_MODEL,
    aggregation_strategy="first"
)

medication_ner = pipeline(
    "token-classification",
    model=MEDICATION_MODEL,
    aggregation_strategy="simple"
)

test_ner = pipeline(
    "token-classification",
    model=TEST_MODEL,
    aggregation_strategy="simple"
)

assertion_classifier = pipeline(
    "text-classification",
    model=ASSERTION_MODEL,
    truncation=True,
    max_length=512
)


# -----------------------------
# Negation model (spaCy + NegEx)
# -----------------------------

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe(
    "negex",
    config={
        "neg_termset": {
            "pseudo_negations": [],
            "preceding_negations": [
                "no",
                "not",
                "never",
                "without",
                "denies",
                "denied",
                "does not",
                "do not",
                "did not",
                "is not",
                "are not",
                "was not",
                "were not"
            ],
            "following_negations": [],
            "termination": [
                "but",
                "however",
                "although",
                "while"
            ]
        }
    }
)


# -------------------------------------------
# Entity boundary cleanup (Task 1)
# -------------------------------------------
# Strips clearly contextual leading/trailing
# words from NER-extracted entity text.
# Uses the actual source text and character
# offsets so positions stay accurate.
# -------------------------------------------

# Leading words that are NEVER part of a
# medical entity name. These are articles
# and common contextual verbs that NER
# models sometimes absorb into entity spans.

_LEADING_ARTICLES = [
    "the patient's ",
    "the patient's ",
    "patient's ",
    "patient's ",
    "patient ",
    "the ",
    "a ",
    "an ",
]

# Contextual verbs that may precede
# medication entities. These are only
# stripped when followed by remaining text
# that looks like an actual entity name
# (i.e., not just whitespace or punctuation).

_LEADING_CONTEXT_VERBS = [
    "prescribed ",
    "receiving ",
    "taking ",
    "given ",
    "using ",
    "on ",
]

# Trailing words that are never part of
# a medical entity name.

_TRAILING_SUFFIXES = [
    " and",
    " tests",
    " test",
]


def clean_entity_boundary(
    text: str,
    entity_text: str,
    start: int,
    end: int
):
    """
    Clean entity boundaries by removing clearly
    contextual leading/trailing words.

    Uses the original source text at [start:end]
    to stay aligned with character offsets.

    Returns (cleaned_text, new_start, new_end).
    """

    # Work from the actual source text
    # to keep offsets accurate
    raw = text[start:end]
    working = raw

    new_start = start
    new_end = end

    # ----------------------------------
    # Strip leading articles
    # ----------------------------------

    for prefix in _LEADING_ARTICLES:

        if working.lower().startswith(prefix):

            working = working[len(prefix):]
            new_start += len(prefix)
            break  # only strip one article

    # ----------------------------------
    # Strip leading context verbs
    # Only strip if remaining text after
    # the verb is non-empty and looks like
    # an actual entity (not just 1-2 chars)
    # ----------------------------------

    for verb in _LEADING_CONTEXT_VERBS:

        if working.lower().startswith(verb):

            remainder = working[len(verb):].strip()

            # Only strip if there's a real
            # entity name remaining (3+ chars)
            if len(remainder) >= 3:

                working = working[len(verb):]
                new_start += len(verb)
                break

    # ----------------------------------
    # Strip trailing suffixes
    # ----------------------------------

    for suffix in _TRAILING_SUFFIXES:

        if working.lower().endswith(suffix):

            remaining = working[
                :-len(suffix)
            ].strip()

            if len(remaining) >= 2:

                working = remaining
                new_end = new_start + len(working)
                break

    # ----------------------------------
    # Strip outer whitespace/punctuation
    # ----------------------------------

    stripped = working.strip(" ,.;:-")

    if stripped and len(stripped) >= 2:

        # Adjust start for leading chars removed
        lead_removed = len(working) - len(
            working.lstrip(" ,.;:-")
        )
        new_start += lead_removed

        # Adjust end for trailing chars removed
        trail_removed = len(working) - len(
            working.rstrip(" ,.;:-")
        )
        new_end -= trail_removed

        working = stripped

    # Final validation: ensure offsets
    # are sane
    if new_start >= new_end:
        return entity_text.strip(), start, end

    return working, new_start, new_end


# -------------------------------------------
# Uncertainty detection (Task 4)
# -------------------------------------------
# Layered approach for POSSIBLE assertions.
# Detects uncertainty cues near the entity
# while respecting temporal overrides like
# "previously suspected... later confirmed."
# -------------------------------------------

# Cues that indicate uncertainty/possibility
_UNCERTAINTY_CUES = [
    "suspected",
    "possible",
    "probable",
    "may have",
    "might have",
    "could have",
    "likely",
    "cannot be ruled out",
    "can not be ruled out",
    "can't be ruled out",
    "questionable",
    "differential",
    "rule out",
    "r/o",
]

# Cues that indicate definite confirmation
# or definite absence — these override
# uncertainty if they appear as the
# latest/strongest modifier
_CONFIRMATION_CUES = [
    "confirmed",
    "definite",
    "established",
    "diagnosed with",
    "consistent with",
]

_RULED_OUT_CUES = [
    "ruled out",
    "was ruled out",
    "has been ruled out",
    "no evidence of",
    "excluded",
]


def _find_cue_position(text_lower, cues):
    """
    Find the rightmost (latest) occurrence
    of any cue in the text. Returns
    (position, cue) or (-1, None).
    """
    best_pos = -1
    best_cue = None

    for cue in cues:
        pos = text_lower.rfind(cue)
        if pos > best_pos:
            best_pos = pos
            best_cue = cue

    return best_pos, best_cue


def detect_uncertainty(
    text: str,
    entity_start: int,
    entity_end: int
) -> str:
    """
    Check for explicit uncertainty, confirmation,
    or ruled-out cues in the sentence context
    around the entity.

    Returns:
        "POSSIBLE" if uncertainty cue is dominant
        "ABSENT" if ruled-out cue is dominant
        "CONFIRMED" if confirmation cue is dominant
        None if no strong cue detected
    """

    text_lower = text.lower()

    # Find latest/rightmost cue of each type
    unc_pos, unc_cue = _find_cue_position(
        text_lower, _UNCERTAINTY_CUES
    )
    conf_pos, conf_cue = _find_cue_position(
        text_lower, _CONFIRMATION_CUES
    )
    ro_pos, ro_cue = _find_cue_position(
        text_lower, _RULED_OUT_CUES
    )

    # No cues found at all
    if unc_pos == -1 and conf_pos == -1 and ro_pos == -1:
        return None

    # If ruled-out cue is the latest/rightmost
    # and appears clearly, treat as ABSENT
    if ro_pos > unc_pos and ro_pos > conf_pos:
        return "ABSENT"

    # If confirmation cue is later than
    # uncertainty cue (e.g., "previously
    # suspected... later confirmed"), the
    # confirmation wins
    if conf_pos > unc_pos and conf_pos > ro_pos:
        return "CONFIRMED"

    # If uncertainty cue is the latest/strongest
    if unc_pos > conf_pos and unc_pos > ro_pos:
        return "POSSIBLE"

    return None


# -------------------------------------------
# Medication status detection (Task 5)
# -------------------------------------------
# Detects whether a medication is currently
# being taken, was stopped, or was only
# prescribed but not actively taken.
# Uses local clause/context analysis only.
# -------------------------------------------

# Patterns indicating medication is NOT
# currently being taken
_MED_NEGATION_PATTERNS = [
    r"not\s+(?:currently\s+)?taking\s+(?:it|them)",
    r"stopped\s+taking",
    r"discontinued",
    r"not\s+(?:currently\s+)?(?:on|using|taking)\b",
    r"no\s+longer\s+(?:taking|on|using)\b",
    r"was\s+stopped",
    r"has\s+been\s+stopped",
    r"was\s+discontinued",
    r"has\s+been\s+discontinued",
    r"previously\s+(?:taking|on|using)\b",
]

# Patterns indicating medication IS currently
# being taken
_MED_PRESENT_PATTERNS = [
    r"currently\s+taking\b",
    r"is\s+taking\b",
    r"is\s+on\b",
    r"is\s+using\b",
    r"continues?\s+(?:taking|on|using)\b",
    r"actively\s+taking\b",
]


def detect_medication_status(
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


# -------------------------------------------
# Assertion detection (Task 3)
# -------------------------------------------
# Layered approach:
# 1. NegEx for explicit strong negation
# 2. Uncertainty cue detection for POSSIBLE
# 3. Ruled-out cue detection for ABSENT
# 4. Fall through to assertion model
# -------------------------------------------

def get_assertion(
    text: str,
    start: int,
    end: int
) -> str:
    """
    Determine assertion for one entity.

    IMPORTANT:
    Assertion is evaluated using the sentence containing the
    entity rather than the entire clinical document.

    This prevents an uncertainty/negation statement in a later
    sentence from incorrectly affecting an unrelated entity.
    """

    entity = text[start:end]

    # --------------------------------------------------------
    # Find the sentence containing this entity.
    #
    # Keep the character offsets relative to the original text
    # for NegEx, but create a local sentence for the assertion
    # classifier.
    # --------------------------------------------------------

    sentence_start = text.rfind(".", 0, start) + 1

    # Also respect common sentence separators.
    for separator in ["!", "?"]:

        candidate = text.rfind(
            separator,
            0,
            start
        ) + 1

        if candidate > sentence_start:
            sentence_start = candidate

    sentence_end_candidates = [
        position
        for position in [
            text.find(".", end),
            text.find("!", end),
            text.find("?", end)
        ]
        if position != -1
    ]

    sentence_end = (
        min(sentence_end_candidates)
        if sentence_end_candidates
        else len(text)
    )

    sentence = text[
        sentence_start:sentence_end
    ].strip()

    # --------------------------------------------------------
    # Entity position relative to local sentence
    # --------------------------------------------------------

    local_start = start - sentence_start
    local_end = end - sentence_start

    # --------------------------------------------------------
    # Layer 1: NegEx
    #
    # Run NegEx on the original document so that existing
    # behavior and offsets remain unchanged.
    # --------------------------------------------------------

    doc = nlp(text)

    span = doc.char_span(
        start,
        end,
        label="CLINICAL",
        alignment_mode="expand"
    )

    if span is not None:

        doc.ents = [span]

        negex = nlp.get_pipe("negex")
        doc = negex(doc)

        if doc.ents and doc.ents[0]._.negex:
            return "ABSENT"

    # --------------------------------------------------------
    # Layer 2: Uncertainty / ruled-out detection
    #
    # IMPORTANT:
    # detect_uncertainty() must operate on the local sentence.
    # --------------------------------------------------------

    cue_result = detect_uncertainty(
        sentence,
        local_start,
        local_end
    )

    if cue_result == "ABSENT":
        return "ABSENT"

    if cue_result == "POSSIBLE":
        return "POSSIBLE"

    # --------------------------------------------------------
    # Layer 3: Assertion model
    #
    # Feed ONLY the entity's sentence to the classifier.
    # This prevents unrelated sentences from contaminating
    # the assertion.
    # --------------------------------------------------------

    marked_text = (
        sentence[:local_start]
        + "[entity] "
        + entity
        + " [entity]"
        + sentence[local_end:]
    )

    result = assertion_classifier(
        marked_text,
        truncation=True
    )[0]

    return result["label"]


# -----------------------------
# Disease extraction
# -----------------------------

def extract_diseases(text: str):

    results = disease_ner(text)

    diseases = []

    current_start = None
    current_end = None

    for entity in results:

        if entity["entity_group"] != "DISEASE":

            if current_start is not None:

                cleaned, cs, ce = clean_entity_boundary(
                    text,
                    text[current_start:current_end],
                    current_start,
                    current_end
                )

                if cleaned:
                    diseases.append({
                        "text": cleaned.strip(),
                        "start": cs,
                        "end": ce
                    })

                current_start = None
                current_end = None

            continue

        start = entity["start"]
        end = entity["end"]

        # Start a new disease span
        if current_start is None:

            current_start = start
            current_end = end

        # Merge adjacent / overlapping pieces
        elif start <= current_end + 1:

            current_end = max(current_end, end)

        # Separate disease
        else:

            cleaned, cs, ce = clean_entity_boundary(
                text,
                text[current_start:current_end],
                current_start,
                current_end
            )

            if cleaned:
                diseases.append({
                    "text": cleaned.strip(),
                    "start": cs,
                    "end": ce
                })

            current_start = start
            current_end = end

    # Add final disease
    if current_start is not None:

        cleaned, cs, ce = clean_entity_boundary(
            text,
            text[current_start:current_end],
            current_start,
            current_end
        )

        if cleaned:
            diseases.append({
                "text": cleaned.strip(),
                "start": cs,
                "end": ce
            })

    return diseases


# -----------------------------
# Symptom extraction
# -----------------------------

def extract_symptoms(text: str):

    results = symptom_ner(text)

    symptoms = []

    for entity in results:

        if entity["entity_group"] != "SYMPTOM":
            continue

        raw_text = text[
            entity["start"]:entity["end"]
        ].strip()

        cleaned, cs, ce = clean_entity_boundary(
            text,
            raw_text,
            entity["start"],
            entity["end"]
        )

        if cleaned:
            symptoms.append({
                "text": cleaned,
                "start": cs,
                "end": ce
            })

    return symptoms


# -----------------------------
# Medication extraction
# -----------------------------

def extract_medications(
    text: str,
    disease_entities=None,
    test_entities=None
):

    results = medication_ner(text)

    medications = []

    for entity in results:

        # Only keep medication predictions
        if entity["entity_group"] != "DRUG":
            continue

        # Ignore low-confidence predictions
        if entity["score"] < 0.70:
            continue

        raw_text = entity["word"].strip(" ,;")
        raw_start = entity["start"]
        raw_end = entity["end"]

        # ----------------------------------
        # Clean entity boundaries
        # ----------------------------------

        medication, clean_start, clean_end = (
            clean_entity_boundary(
                text,
                raw_text,
                raw_start,
                raw_end
            )
        )

        if not medication:
            continue

        # ----------------------------------
        # Ignore predictions that overlap
        # with detected diseases
        # ----------------------------------

        overlaps_disease = False

        if disease_entities:

            for disease in disease_entities:

                if (
                    clean_start < disease["end"]
                    and clean_end > disease["start"]
                ):

                    overlaps_disease = True
                    break

        if overlaps_disease:
            continue

        # ----------------------------------
        # Ignore predictions that overlap
        # with detected tests
        # ----------------------------------

        overlaps_test = False

        if test_entities:

            for test in test_entities:

                if (
                    clean_start < test["end"]
                    and clean_end > test["start"]
                ):

                    overlaps_test = True
                    break

        if overlaps_test:
            continue

        # ----------------------------------
        # Avoid duplicates
        # ----------------------------------

        duplicate = any(
            medication_item["text"].lower()
            == medication.lower()
            and medication_item["start"]
            == clean_start
            for medication_item in medications
        )

        if duplicate:
            continue

        medications.append({
            "text": medication,
            "start": clean_start,
            "end": clean_end
        })

    return medications


# -----------------------------
# Test extraction
# -----------------------------

def extract_tests(text: str):

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


# -------------------------------------------
# Final entity extraction (Task 6)
# -------------------------------------------
# Internal metadata is richer; public output
# preserves the existing flat-list format.
# -------------------------------------------

def extract_entities(text: str):

    # ----------------------------------
    # Extract entity types
    # ----------------------------------

    disease_entities = extract_diseases(text)

    symptom_entities = extract_symptoms(text)

    # Tests are extracted BEFORE medications
    # so medication predictions can be checked
    # against test predictions.
    test_entities = extract_tests(text)

    medication_entities = extract_medications(
        text,
        disease_entities,
        test_entities
    )

    # ----------------------------------
    # Remove disease predictions that
    # overlap with detected symptoms
    # ----------------------------------

    filtered_diseases = []

    for disease in disease_entities:

        overlaps_symptom = False

        for symptom in symptom_entities:

            if (
                disease["start"] < symptom["end"]
                and disease["end"] > symptom["start"]
            ):

                overlaps_symptom = True
                break

        if not overlaps_symptom:

            filtered_diseases.append(disease)

    # ----------------------------------
    # Internal rich entity metadata
    # ----------------------------------

    _internal_entities = []

    # ----------------------------------
    # Disease assertion
    # ----------------------------------

    final_diseases = []

    for disease in filtered_diseases:

        assertion = get_assertion(
            text,
            disease["start"],
            disease["end"]
        )

        _internal_entities.append({
            "text": disease["text"],
            "category": "disease",
            "start": disease["start"],
            "end": disease["end"],
            "assertion": assertion,
        })

        if assertion == "PRESENT":

            final_diseases.append(
                disease["text"]
            )

    # ----------------------------------
    # Symptom assertion
    # ----------------------------------

    final_symptoms = []

    for symptom in symptom_entities:

        assertion = get_assertion(
            text,
            symptom["start"],
            symptom["end"]
        )

        _internal_entities.append({
            "text": symptom["text"],
            "category": "symptom",
            "start": symptom["start"],
            "end": symptom["end"],
            "assertion": assertion,
        })

        if assertion == "PRESENT":

            final_symptoms.append(
                symptom["text"]
            )

    # ----------------------------------
    # Medication assertion + status
    # ----------------------------------

    final_medications = []

    for medication in medication_entities:

        # First check medication-specific
        # status (prescribed-but-not-taking, etc.)
        med_status = detect_medication_status(
            text,
            medication["text"],
            medication["start"],
            medication["end"],
            medication_entities
        )

        if med_status is not None:
            assertion = med_status
        else:
            assertion = get_assertion(
                text,
                medication["start"],
                medication["end"]
            )

        logger.debug(
            "MEDICATION: %s | ASSERTION: %s",
            medication["text"],
            assertion
        )

        _internal_entities.append({
            "text": medication["text"],
            "category": "medication",
            "start": medication["start"],
            "end": medication["end"],
            "assertion": assertion,
            "medication_status": med_status,
        })

        if assertion == "PRESENT":
            final_medications.append(
                medication["text"]
            )

    # ----------------------------------
    # Test assertion
    # ----------------------------------

    final_tests = []

    for test in test_entities:

        assertion = get_assertion(
            text,
            test["start"],
            test["end"]
        )

        _internal_entities.append({
            "text": test["text"],
            "category": "test",
            "start": test["start"],
            "end": test["end"],
            "assertion": assertion,
        })

        if assertion == "PRESENT":

            final_tests.append(
                test["text"]
            )

    # ----------------------------------
    # Remove duplicates
    # ----------------------------------

    final_diseases = list(
        dict.fromkeys(final_diseases)
    )

    final_symptoms = list(
        dict.fromkeys(final_symptoms)
    )

    final_medications = list(
        dict.fromkeys(final_medications)
    )

    final_tests = list(
        dict.fromkeys(final_tests)
    )

    # ----------------------------------
    # Public output (unchanged format)
    # ----------------------------------

    return {
        "clinical_text": text,
        "diseases": final_diseases,
        "symptoms": final_symptoms,
        "medications": final_medications,
        "tests": final_tests,
        # Internal metadata preserved for
        # future downstream use
        "_internal_entities": _internal_entities,
    }