import re


def clean_text(text: str) -> str:
    """
    Cleans clinical text by:
    - converting to lowercase
    - preserving punctuation
    - removing extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text