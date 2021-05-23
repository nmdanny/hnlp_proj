import re
from typing import List

ELEMENT_RE = re.compile("\\b([^,]+)", re.UNICODE)
LAST_ELEMENT_RE = re.compile("\\s+ו([^,]+)", re.UNICODE)


def parse_joined_elements(authors: str) -> List[str]:
    """Given a string containing severals elements separated by asterisks,
    and the final one by ו החיבור, returns each element separately"""
    last_name = None
    match = LAST_ELEMENT_RE.search(authors)
    if match:
        last_name = match.group(1)
        authors = authors[: match.start(0)]
    names = ELEMENT_RE.findall(authors)
    if last_name:
        names.append(last_name)
    return names


def extract_authors(suspected_authors: List[str]) -> List[str]:
    """Given a suspected list of author names, some of which might actually contain
    several authors, transforms it to a list of authors"""
    return [name for sus in suspected_authors for name in parse_joined_elements(sus)]


def combine_texts(texts: List[str]) -> str:
    """Given a list of scraped text elements, combines them into a single string,
    getting rid of extranous whitespace"""
    return " ".join(text.strip() for text in texts)


def flip_hebrew_text(text: str) -> str:
    # check if text contains characters in hebrew block
    hebrew_range = range(0x590, 0x5FF + 1)
    if any(True for c in text if ord(c) in hebrew_range):
        return text[::-1]
    return text
