from pathlib import Path
import shutil


FILE = Path("entity_extractor.py")
BACKUP = Path("entity_extractor_V2_2_ASSERTION_BACKUP.py")


if not FILE.exists():
    raise FileNotFoundError(
        "entity_extractor.py not found."
    )


# ------------------------------------------------------------
# Safety backup
# ------------------------------------------------------------

shutil.copy2(FILE, BACKUP)

print(f"Backup created: {BACKUP}")


source = FILE.read_text(encoding="utf-8")


# ------------------------------------------------------------
# Locate get_assertion()
# ------------------------------------------------------------

start_marker = "def get_assertion("
end_marker = "\n\n# -----------------------------\n# Disease extraction"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start == -1 or end == -1:
    raise RuntimeError(
        "Could not safely locate get_assertion(). "
        "No changes were made."
    )


new_function = r'''def get_assertion(
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
'''


source = (
    source[:start]
    + new_function
    + source[end:]
)


FILE.write_text(
    source,
    encoding="utf-8"
)

print("Assertion-scope patch applied successfully.")
print(f"Modified: {FILE}")
print(f"Backup:   {BACKUP}")