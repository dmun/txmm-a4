from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import functions as F
import string

DEFAULT_MAX_DEPTH = 30
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42

ablations = [
    [],
    "avg_num_chars_per_token",
    "num_chars",
    "num_sents",
    "num_alpha",
    "num_punct",
    "num_punct2",
    "num_punct3",
    "num_contractions",
    "num_parentheses",
    # "parentheses_frequency",
    "num_emoticons",
    "num_no_vowel_words",
    "avg_word_length",
    "num_single_quotes",
    "num_double_quotes",
    "num_whitespaces",
    F.ENGLISH_STOP_WORDS,
    list(string.ascii_lowercase),
    F.POS_TAGS,
]


groups = [
    "0: None",
    "1: Average token length",
    "2: Total number of characters",
    "3: Total number of sentences",
    "4: Total number of alphabets",
    "5: Total number of punctuation marks",
    "6: Total number of double punctuation marks",
    "7: Total number of triple punctuation marks",
    "8: Total number of contractions",
    "9: Total number of parentheses",
    # "10: Parentheses frequency",
    "11: Total number of emoticons",
    "12: Total number of no-vowel words",
    "13: Average word length",
    "14: Total number of single quotes",
    "15: Total number of double quotes",
    "16: Total number of whitespaces",
    "17: Count of each function word",
    "18: Count of each letter",
    "19: Count of each POS tag",
]



filter = [
]


def perform_ablation(ablation: str | list[str],
                     max_depth=DEFAULT_MAX_DEPTH,
                     n_estimators=DEFAULT_N_ESTIMATORS,
                     random_state=DEFAULT_RANDOM_STATE,):

    X_train_filtered = X_train.drop(ablation, axis=1)
    X_test_filtered = X_test.drop(ablation, axis=1)
    X_train_filtered = X_train_filtered.drop(filter, axis=1)
    X_test_filtered = X_test_filtered.drop(filter, axis=1)

    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train_filtered, y_train)
    y_pred = clf.predict(X_test_filtered)

    score = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    return score, acc
