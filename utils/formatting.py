"""
Utilities
"""
import re


def clean_text(text: str, remove_punctuation=False) -> str:
    """Cleans the inputted text based on the rules given in the comments.
    Code taken from: https://github.com/kk7nc/Text_Classification/

    Args:
        text (str): the text to be cleaned
        remove_punctuation (bool): whether to remove punctuation or not

    Returns:
        the cleaned text
    """
    rules = [
        {r">\s+": u">"},  # remove spaces after a tag opens or closes
        {r"\s+": u" "},  # replace consecutive spaces
        {r"\s*<br\s*/?>\s*": u"\n"},  # newline after a <br>
        {r"</(div)\s*>\s*": u"\n"},  # newline after </p> and </div> and <h1/>...
        {r"</(p|h\d)\s*>\s*": u"\n\n"},  # newline after </p> and </div> and <h1/>...
        {r"<head>.*<\s*(/head|body)[^>]*>": u""},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r"\1"},  # show links instead of texts
        {r"[ \t]*<[^<]*?/?>": u""},  # remove remaining tags
        {r"^\s+": u""},  # remove spaces at the beginning
    ]
    if remove_punctuation:
        rules.append({r"[.,\/#!$%\^&\*;:{}=\-_`~()]": u""})
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()
