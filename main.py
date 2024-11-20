from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import nltk
import functions as F
import string
from pandarallel import pandarallel
from joblib import Parallel, delayed

# %%
np.random.seed(42)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
pandarallel.initialize(progress_bar=False)

# %%
train_data = pd.read_csv("./data/pan2425_train_data.csv")
dev_data = pd.read_csv("./data/pan2425_dev_data.csv")
test_data = pd.read_csv("./data/pan2425_test_data.csv")
y_train = train_data["author"]
y_dev = dev_data["author"]
y_test = test_data["author"]
print(train_data.head())

# %%
reload(F)
X_train = F.extract_features(train_data).fillna(0)
X_test = F.extract_features(test_data).fillna(0)
# X_dev = F.extract_features(dev_data).fillna(0)

# %%
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
    "None",
    "Average token length",
    "Total number of characters",
    "Total number of sentences",
    "Total number of alphabets",
    "Total number of punctuation marks",
    "Total number of double punctuation marks",
    "Total number of triple punctuation marks",
    "Total number of contractions",
    "Total number of parentheses",
    # "10: Parentheses frequency",
    "Total number of emoticons",
    "Total number of no-vowel words",
    "Average word length",
    "Total number of single quotes",
    "Total number of double quotes",
    "Total number of whitespaces",
    "Count of each function word",
    "Count of each letter",
    "Count of each POS tag",
]


filter = []


def perform_ablation(
    ablation: str | list[str],
    max_depth=DEFAULT_MAX_DEPTH,
    n_estimators=DEFAULT_N_ESTIMATORS,
    random_state=DEFAULT_RANDOM_STATE,
):
    X_train_filtered = X_train.drop(ablation, axis=1)
    X_test_filtered = X_test.drop(ablation, axis=1)
    # X_train_filtered = X_train_filtered.drop(filter, axis=1)
    # X_test_filtered = X_test_filtered.drop(filter, axis=1)

    clf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, random_state=random_state
    )
    clf.fit(X_train_filtered, y_train)
    y_pred = clf.predict(X_test_filtered)

    score = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    return score, acc


def plot_ablation_results(results, groups):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = list(string.ascii_uppercase)[: len(groups)]
    labels = [f"{x}: {group}" for x, group in zip(xs, groups)]
    ax.bar(xs, results[0], label=labels)
    ax.set_ylim(bottom=0.75)
    ax.set_xticks(xs)
    ax.set_xlabel("Remove feature group")
    ax.set_ylabel("F-measure")
    ax.legend()


# %%
results = np.array(Parallel(-1)(delayed(perform_ablation)(x) for x in ablations)).T
plot_ablation_results(results, groups)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
xs = np.arange(len(groups))
ax.bar(xs, results[0], label=groups)

ax.set_ylim(bottom=0.75)
ax.set_xticks(xs)
ax.set_xlabel("Remove feature group")
ax.set_ylabel("F-measure")
ax.legend()
plt.show()

# %%
plt.figure()

params = np.arange(5, 30, 2)
results = np.array(Parallel(-1)(delayed(perform_ablation)([], p) for p in params)).T

plt.plot(params, results[0], label="F-measure")
plt.plot(params, results[1], label="accuracy")
plt.xlabel("max_depth")

plt.grid()
plt.legend()
plt.show()

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"F-score: {f:.2f}")
