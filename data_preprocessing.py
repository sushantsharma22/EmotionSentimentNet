# data_preprocessing.py
import re
import string
import emoji
import spacy
from nltk.tokenize import TweetTokenizer
from config import STOP_WORDS
import en_core_web_sm

# Load spaCy model
nlp = en_core_web_sm.load()

def replace_emoji(token: str) -> str:
    if emoji.is_emoji(token):
        return emoji.demojize(token, delimiters=(" ", " ")).replace(":", "")
    return token

def tokenize_text(text: str) -> list:
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

def spacy_lemmatize(tokens: list) -> list:
    doc = nlp(" ".join(tokens))
    return [token.lemma_.strip() for token in doc if token.lemma_.strip()]

def advanced_clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = tokenize_text(text)
    cleaned_tokens = []
    for token in tokens:
        token = replace_emoji(token)
        token = token.translate(str.maketrans("", "", string.punctuation))
        if token.strip():
            cleaned_tokens.append(token.strip())
    filtered_tokens = [t for t in cleaned_tokens if t not in STOP_WORDS]
    lemmatized_tokens = spacy_lemmatize(filtered_tokens)
    final_tokens = [w for w in lemmatized_tokens if w.isalpha() and len(w) > 1]
    return " ".join(final_tokens)
