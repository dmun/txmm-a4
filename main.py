from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import nltk
import functions as F
from joblib import Parallel, delayed

# %%
np.random.seed(42)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

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

# %%
ablations = [
    [],
    "avg_num_chars_per_token",
    "num_chars",
    # "num_sents",
    "num_alpha",
    "num_punct",
    "num_punct2",
    "num_punct3",
    "num_contractions",
    # "num_parentheses",
    # "parentheses_frequency",
    # "emoticons_frequency",
    "num_no_vowel_words",
    # "avg_word_length",
    F.POS_TAGS,
    F.ENGLISH_STOP_WORDS,
]

groups = [
    "0: None",
    "1: Average token length",
    "2: Total number of characters",
    # "3: Total number of sentences",
    "3: Total alphabet count",
    "4: Total punctuation count",
    "5: Two continuous punctuation count",
    "6: Three continuous punctuation count",
    "7: Total contraction count",
    # "9: Parenthesis count",
    # "10: Parenthesis frequency",
    # "11: Emoticons frequency",
    "8: Number of no vowel words",
    # "13: Average word length",
    "9: POS tags",
    "10: Function Words",
]

filter = [
    "num_sents",
    "num_parentheses",
    "parentheses_frequency",
    "emoticons_frequency",
    "avg_word_length",
]


def perform_ablation(ablation: str | list[str], max_depth=30):
    X_train_filtered = X_train.drop(ablation, axis=1)
    X_test_filtered = X_test.drop(ablation, axis=1)
    X_train_filtered = X_train_filtered.drop(filter, axis=1)
    X_test_filtered = X_test_filtered.drop(filter, axis=1)

    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=42)
    clf.fit(X_train_filtered, y_train)
    y_pred = clf.predict(X_test_filtered)

    score = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    return score, acc


# %%
results = np.array(Parallel(-1)(delayed(perform_ablation)(x) for x in ablations)).T

fig, ax = plt.subplots(figsize=(10, 10))
xs = np.arange(11)
ax.bar(xs, results[0], label=groups)

ax.set_ylim(bottom=0.75)
ax.set_xticks(xs)
ax.set_xlabel("Remove feature group")
ax.set_ylabel("F-measure")
ax.legend()
plt.show()

# %%
plt.figure()

params = np.arange(10, 50, 5)
results = np.array(Parallel(-1)(delayed(perform_ablation)([], p) for p in params)).T

plt.plot(params, results[0], label="F-measure")
plt.plot(params, results[1], label="accuracy")
plt.xlabel("max_depth")

plt.grid()
plt.legend()
plt.show()

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"F-score: {f:.2f}")
