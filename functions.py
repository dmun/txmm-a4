from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import nltk
import string
import re
from collections import Counter
from nltk.chunk import RegexpParser
from nltk import Tree


TOP_K = 1000


def get_top_k_words(data, k):
    all_text = " ".join(data['text'].dropna().astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_counts = Counter(words)
    top_k_words = word_counts.most_common(k)
    return [word for word, _ in top_k_words]


# TOP_K_WORDS = get_top_k_words(train_data, TOP_K)

POS_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
    'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
    'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
    'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
    'WP', 'WP$', 'WRB'
]

PHRASE = {
    'NP': r'{<DT>?<JJ>*<NN.*>+}',
    'VP': r'{<VB.*><RB.*>?<VB.*>?}'
}


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
    # return len(re.findall(r"[:;]-?\)", text))
    return len(re.findall(r"[:;]-?[\)\)\[\]]", text))


def get_avg_word_length(tokens: list[str]):
    words = [token for token in tokens if re.match(r"^[a-zA-Z]+$", token)]
    return sum([len(t) for t in words]) / len(words)


def get_num_function_word(text: str):
    words = text.split()
    function_word_count = {
        f'fn:{function_word}': words.count(function_word)
        for function_word in ENGLISH_STOP_WORDS
    }
    return pd.Series(function_word_count)


def get_num_letter(text: str):
    letter_count = {f'let:{char}': text.lower().count(char) for char in string.ascii_letters}
    return pd.Series(letter_count)


def get_num_no_vowel_words(tokens: list[str]):
    return sum(1 for text in tokens if not re.search(r"[aeiou]", text) and re.search(r"\w", text))


def get_num_whitespaces(text: str):
    return len(re.findall(r"\w", text))


def get_num_single_quotes(text: str):
    return len(re.findall(r"'", text))


def get_num_double_quotes(text: str):
    return len(re.findall(r'"', text))


def get_num_uppercase_letters(text: str):
    return sum(1 for char in text if char.isupper())


def get_num_digit(text: str):
    return sum(1 for char in text if char.isdigit())

# def get_num_tokens(tokens: list[str], word: str):
#     return sum([1 for token in tokens if token == word])


def get_num_pos_tags(tokens: list[str]):
    counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
    return pd.Series({f'pos:{tag}': counts.get(tag, 0) for tag in POS_TAGS})


def get_num_phrase(tokens: list[str]):
    pos_tags = nltk.pos_tag(tokens)
    grammar = "\n".join(f"{key}: {value}" for key, value in PHRASE.items())
    chunked = RegexpParser(grammar).parse(pos_tags)
    phrase_counts = Counter()
    for subtree in chunked:
        if isinstance(subtree, Tree):
            phrase_counts[subtree.label()] += 1
    return pd.Series(phrase_counts)

# def get_count_top_k_words(tokens: list[str]):
#     token_counts = Counter(tokens)
#     return pd.Series({word: token_counts.get(word, 0) for word in TOP_K_WORDS})


def extract_features(data: pd.DataFrame):
    df = get_tokenized_data(data)
    df["avg_num_chars_per_token"] = df["tokens"].parallel_apply(get_avg_num_token_chars) # 0.06
    df["num_chars"] = df["text"].parallel_apply(len) #0.059
    df["num_sents"] = df["text"].parallel_apply(get_num_sentences) # 0.12
    df["num_alpha"] = df["text"].parallel_apply(get_num_alphabet) #0.067
    df["num_punct"] = df["text"].parallel_apply(get_num_punctuation) #0.103
    df["num_punct2"] = df["text"].parallel_apply(get_num_punctuation, n=2) #0.082
    df["num_punct3"] = df["text"].parallel_apply(get_num_punctuation, n=3) #0.060
    df["num_contractions"] = df["text"].parallel_apply(get_num_contractions) #0.039
    df["num_parentheses"] = df["text"].parallel_apply(get_num_parentheses) #0.053
    df["parentheses_frequency"] = df["text"].parallel_apply(get_parentheses_frequency) # 0.062
    df["num_emoticons"] = df["text"].parallel_apply(get_num_emoticons) #0.007
    df["num_uppercase_letters"] = df["text"].parallel_apply(get_num_uppercase_letters) # 0.11
    df["num_digit"] = df["text"].parallel_apply(get_num_digit) # 0.11
    df["num_no_vowel_words"] = df["tokens"].parallel_apply(get_num_no_vowel_words) #0.126
    df["avg_word_length"] = df["tokens"].parallel_apply(get_avg_word_length) #0.070
    df["num_single_quotes"] = df["text"].parallel_apply(get_num_single_quotes) #0.025
    df["num_double_quotes"] = df["text"].parallel_apply(get_num_double_quotes) #0.100
    df["num_whitespaces"] = df["text"].parallel_apply(get_num_whitespaces) #0.064
    df = pd.concat([df, df["text"].parallel_apply(get_num_function_word)], axis=1) #0.764
    df = pd.concat([df, df["text"].parallel_apply(get_num_letter)], axis=1) #0.627
    df = pd.concat([df, df["tokens"].parallel_apply(get_num_pos_tags)], axis=1) #0.656
    df = pd.concat([df, df["tokens"].parallel_apply(get_num_phrase)], axis=1) #0.090
    # df = pd.concat([df, df["tokens"].parallel_apply(get_count_top_k_words)], axis=1)
    return df.drop(["Unnamed: 0", "text", "tokens", "author"], axis=1).fillna(0)
