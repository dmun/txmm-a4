from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import nltk
import string
import re
from collections import Counter
from nltk.chunk import RegexpParser
from nltk import Tree


POS_TAGS = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
]

PHRASE = {"NP": r"{<DT>?<JJ>*<NN.*>+}", "VP": r"{<VB.*><RB.*>?<VB.*>?}"}


def get_tokenized_data(data: pd.DataFrame):
    new_data = data.copy()
    new_data["tokens"] = data["text"].apply(nltk.word_tokenize)
    return new_data


def get_avg_num_token_chars(tokens: list[str]):
    return sum([len(t) for t in tokens]) / len(tokens)


def get_num_sentences(text: str):
    return len(nltk.sent_tokenize(text))


def get_num_alphabet(text: str):
    return sum(1 for char in text if char.isalpha())


def get_num_punctuation(text: str, n: int = 1):
    r = r"""!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~"""
    pattern = f"[{r}]"
    if n > 1:
        pattern = f"[^{r}]" + f"[{r}]" * n + f"[^{r}]"
    return len(re.findall(pattern, text + " "))


def get_num_contractions(text: str):
    pattern = r"""[a-zA-Z]['"][a-zA-Z]"""
    return len(re.findall(pattern, text))


def get_num_parentheses(text: str):
    return len(re.findall(r"""[\(\)]""", text))


def get_parentheses_frequency(text: str):
    word = text.split()
    return len(re.findall(r"""[\(\)]""", text)) / len(word) if len(word) > 0 else 0


def get_num_emoticons(text: str):
    return len(re.findall(r"[:;]-?[\)\)\[\]]", text))


def get_avg_word_length(tokens: list[str]):
    words = [token for token in tokens if re.match(r"^[a-zA-Z]+$", token)]
    return sum([len(t) for t in words]) / len(words)


def get_num_function_word(text: str):
    words = text.split()
    function_word_count = {
        function_word: words.count(function_word)
        for function_word in ENGLISH_STOP_WORDS
    }
    return pd.Series(function_word_count)


def get_num_letter(text: str):
    letter_count = {char: text.lower().count(char) for char in string.ascii_lowercase}
    return pd.Series(letter_count)


def get_num_no_vowel_words(tokens: list[str]):
    return sum([len(re.findall(r"[aeiou]", text)) for text in tokens])


def get_num_whitespaces(text: str):
    return len(re.findall(r"\w", text))


def get_num_single_quotes(text: str):
    return len(re.findall(r"'", text))


def get_num_double_quotes(text: str):
    return len(re.findall(r'"', text))


# def get_num_tokens(tokens: list[str], word: str):
#     return sum([1 for token in tokens if token == word])


def get_num_pos_tags(tokens: list[str]):
    counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
    return pd.Series({tag: counts.get(tag, 0) for tag in POS_TAGS})


def get_num_phrase(tokens: list[str]):
    pos_tags = nltk.pos_tag(tokens)
    grammar = "\n".join(f"{key}: {value}" for key, value in PHRASE.items())
    chunked = RegexpParser(grammar).parse(pos_tags)
    phrase_counts = Counter()
    for subtree in chunked:
        if isinstance(subtree, Tree):
            phrase_counts[subtree.label()] += 1
    return pd.Series(phrase_counts)


def extract_features(data: pd.DataFrame):
    df = get_tokenized_data(data)
    df["avg_num_chars_per_token"] = df["tokens"].parallel_apply(get_avg_num_token_chars)
    df["num_chars"] = df["text"].parallel_apply(len)
    df["num_sents"] = df["text"].parallel_apply(get_num_sentences)
    df["num_alpha"] = df["text"].parallel_apply(get_num_alphabet)
    df["num_punct"] = df["text"].parallel_apply(get_num_punctuation)
    df["num_punct2"] = df["text"].parallel_apply(get_num_punctuation, n=2)
    df["num_punct3"] = df["text"].parallel_apply(get_num_punctuation, n=3)
    df["num_contractions"] = df["text"].parallel_apply(get_num_contractions)
    df["num_parentheses"] = df["text"].parallel_apply(get_num_parentheses)
    # df["parentheses_frequency"] = df["text"].parallel_apply(get_parentheses_frequency)
    df["num_emoticons"] = df["text"].parallel_apply(get_num_emoticons)
    df["num_no_vowel_words"] = df["tokens"].parallel_apply(get_num_no_vowel_words)
    df["avg_word_length"] = df["tokens"].parallel_apply(get_avg_word_length)
    df["num_single_quotes"] = df["text"].parallel_apply(get_num_single_quotes)
    df["num_double_quotes"] = df["text"].parallel_apply(get_num_double_quotes)
    df["num_whitespaces"] = df["text"].parallel_apply(get_num_whitespaces)
    df = pd.concat([df, df["text"].parallel_apply(get_num_function_word)], axis=1)
    df = pd.concat([df, df["text"].parallel_apply(get_num_letter)], axis=1)
    df = pd.concat([df, df["tokens"].parallel_apply(get_num_pos_tags)], axis=1)
    # df = pd.concat([df, df["tokens"].parallel_apply(get_num_phrase)], axis=1)
    return df.drop(["Unnamed: 0", "text", "tokens", "author"], axis=1)
