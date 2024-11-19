from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import nltk
import string
import re


# %%
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


def get_emoticons_frequency(text: str):
    word = text.split()
    return len(re.findall(r"[:;]-?\)", text)) / len(word) if len(word) > 0 else 0


def get_avg_word_length(text: str):
    words = text.split()
    word_lengths = list(map(lambda w: len(w) if w.isalpha() else 0, words))
    word_lengths = list(filter(lambda w: w > 0, word_lengths))
    return np.mean(word_lengths) if len(word_lengths) > 0 else 0


def get_function_word_frequency(text: str):
    words = text.split()
    total_num = len(words)
    function_word_count = {
        function_word: words.count(function_word)
        for function_word in ENGLISH_STOP_WORDS
    }
    function_word_frequency = {
        function_word: (count / total_num if total_num > 0 else 0)
        for function_word, count in function_word_count.items()
    }
    return pd.Series(function_word_frequency)


def get_letter_frequency(text: str):
    total_num = sum(1 for char in text if char.isalpha())
    letter_count = {char: text.lower().count(char) for char in string.ascii_lowercase}
    letter_frequency = {
        char: (count / total_num if total_num > 0 else 0)
        for char, count in letter_count.items()
    }
    return pd.Series(letter_frequency)


def get_num_no_vowel_words(tokens: list[str]):
    return sum([len(re.findall(r"[aeiou]", text)) for text in tokens])


def get_num_whitespaces(text: str):
    return len(re.findall(r"\w", text))


def get_num_single_quotes(text: str):
    return len(re.findall(r"'", text))


def get_num_double_quotes(text: str):
    return len(re.findall(r'"', text))


def extract_features(data: pd.DataFrame):
    df = get_tokenized_data(data)
    df["avg_num_chars_per_token"] = df["tokens"].apply(get_avg_num_token_chars)
    df["num_chars"] = df["text"].apply(len)
    df["num_sents"] = df["text"].apply(get_num_sentences)
    df["num_alpha"] = df["text"].apply(get_num_alphabet)
    df["num_punct"] = df["text"].apply(get_num_punctuation)
    df["num_punct2"] = df["text"].apply(get_num_punctuation, n=2)
    df["num_punct3"] = df["text"].apply(get_num_punctuation, n=3)
    df["num_contractions"] = df["text"].apply(get_num_contractions)
    df["num_parentheses"] = df["text"].apply(get_num_parentheses)
    # df["num_single_quotes"] = df["text"].apply(get_num_single_quotes)
    # df["num_double_quotes"] = df["text"].apply(get_num_double_quotes)
    # df["num_whitespaces"] = df["text"].apply(get_num_whitespaces)
    # df["parentheses_frequency"] = df["text"].apply(get_parentheses_frequency)
    # df["emoticons_frequency"] = df["text"].apply(get_emoticons_frequency)
    df["num_no_vowel_words"] = df["tokens"].apply(get_num_no_vowel_words)
    df["avg_word_length"] = df["text"].apply(get_avg_word_length)
    df = pd.concat([df, df["text"].apply(get_function_word_frequency)], axis=1)
    df = pd.concat([df, df["text"].apply(get_letter_frequency)], axis=1)
    return df.drop(["text", "tokens", "author"], axis=1)
